import numpy as np
import scipy.io as sio
import cvxpy as cp
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.layer2 = nn.Linear(hidden_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights and biases
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        
    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.layer2(x)
        return x

def params_to_vec(model):
    """Convert model parameters to a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def vec_to_params(model, vec):
    """Update model parameters using a vector."""
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = vec[pointer:pointer + num_param].view_as(param)
        pointer += num_param

def ProjectOntoL1Ball(v, b):
    """Projects point onto L1 ball of specified radius."""
    if b < 0:
        raise ValueError(f"Radius of L1 ball is negative: {b:.3f}")
    if np.linalg.norm(v, ord=1) <= b:
        return v.copy()
    u = np.sort(np.abs(v))[::-1]
    sv = np.cumsum(u)
    rho = np.where(u > (sv - b) / np.arange(1, len(u) + 1))[0]
    if rho.size == 0:
        theta = 0.0
    else:
        rho_idx = rho[-1]
        theta = (sv[rho_idx] - b) / (rho_idx + 1)
    w = np.sign(v) * np.maximum(np.abs(v) - theta, 0)
    return w

def DBGD(fun_f, grad_f, grad_g, fun_g, TSA, param, x0):
    """Dynamic Barrier Gradient Descent algorithm."""
    stepsize = param['stepsize']
    alpha = param['alpha']
    beta = param['beta']
    lambda_ = param['lam']
    maxiter = param['maxiter']
    maxtime = param['maxtime']
    
    x = x0.copy()
    start_time = time.time()
    
    f_vec1 = []
    g_vec1 = []
    time_vec1 = []
    acc_vec = []
    
    iter_count = 0
    while iter_count <= maxiter:
        iter_count += 1
        
        grad_f_x = grad_f(x)
        grad_g_x = grad_g(x)
        
        g_x = fun_g(x)
        grad_g_norm_sq = np.dot(grad_g_x, grad_g_x)
        phi = min(alpha * g_x, beta * grad_g_norm_sq)
        
        if grad_g_norm_sq == 0:
            weight = 0
        else:
            weight = max((phi - np.dot(grad_f_x, grad_g_x)) / grad_g_norm_sq, 0)
        
        v = grad_f_x + weight * grad_g_x
        x = x - stepsize * v
        x = ProjectOntoL1Ball(x, lambda_)
        
        current_time = time.time() - start_time
        f_vec1.append(fun_f(x))
        g_vec1.append(fun_g(x))
        time_vec1.append(current_time)
        acc_vec.append(TSA(x))
        
        if iter_count % 5000 == 1:
            print(f"Iteration: {iter_count}")
            
        if current_time > maxtime:
            break
    
    return (np.array(f_vec1), np.array(g_vec1), np.array(time_vec1), 
            x, np.array(acc_vec))

def main():
    # Load data
    np.random.seed(123456)
    data = sio.loadmat('wikivital_mathematics.mat')
    A = torch.FloatTensor(data['A'].toarray())
    b = torch.FloatTensor(data['b']).reshape(-1, 1)
    m, n = A.shape

    # Split data
    trp = 0.6
    idx = torch.randperm(m)
    train_end = int(trp * m)
    val_end = train_end + int((m * (1-trp)/2))

    # Training data
    A2 = A[idx[:train_end]]
    b2 = b[idx[:train_end]]

    # Validation data
    A1 = A[idx[train_end:val_end]]
    b1 = b[idx[train_end:val_end]]

    # Test data
    A3 = A[idx[val_end:]]
    b3 = b[idx[val_end:]]

    # Initialize models
    model_f = MLP(input_dim=n)
    model_g = MLP(input_dim=n)

    # Define objective functions
    def fun_f(x):
        vec_to_params(model_f, torch.FloatTensor(x))
        pred = model_f(A1)
        return float(torch.sum((pred - b1)**2)) / 2

    def fun_g(x):
        vec_to_params(model_g, torch.FloatTensor(x))
        pred = model_g(A2)
        return float(torch.sum((pred - b2)**2)) / 2

    def grad_f(x):
        x_tensor = torch.FloatTensor(x)
        x_tensor.requires_grad_(True)
        vec_to_params(model_f, x_tensor)
        
        pred = model_f(A1)
        loss = torch.sum((pred - b1)**2) / 2
        loss.backward()
        
        grad = torch.cat([p.grad.view(-1) for p in model_f.parameters()])
        return grad.detach().numpy()

    def grad_g(x):
        x_tensor = torch.FloatTensor(x)
        x_tensor.requires_grad_(True)
        vec_to_params(model_g, x_tensor)
        
        pred = model_g(A2)
        loss = torch.sum((pred - b2)**2) / 2
        loss.backward()
        
        grad = torch.cat([p.grad.view(-1) for p in model_g.parameters()])
        return grad.detach().numpy()

    def TSA_LS(x):
        vec_to_params(model_f, torch.FloatTensor(x))
        pred = model_f(A3)
        return float(torch.sum((pred - b3)**2)) / 2

    # DBGD parameters
    param = {
        'stepsize': 1e-4,
        'alpha': 1,
        'beta': 1,
        'lam': 1,
        'maxiter': int(1e7),
        'maxtime': 100
    }

    # Initialize
    total_params = sum(p.numel() for p in model_f.parameters())
    x0 = np.zeros(total_params)

    # Run DBGD
    print('DBGD Algorithm starts')
    f_vec, g_vec, time_vec, x_final, tsa_vec = DBGD(
        fun_f, grad_f, grad_g, fun_g, TSA_LS, param, x0
    )
    print('DBGD Solution Achieved!')

    # Plotting
    plt.figure(figsize=(10, 12))


    # Lower-level gap plot
    plt.subplot(3, 1, 1)
    plt.semilogy(time_vec, g_vec, '-')
    plt.ylabel('g(βk)')
    plt.xlabel('time (s)')
    plt.title('Lower-level Optimization ')
    plt.grid(True)

    # Upper-level gap plot
    plt.subplot(3, 1, 2)
    plt.semilogy(time_vec, f_vec, '-')
    plt.ylabel('f(βk)')
    plt.xlabel('time (s)')
    plt.title('Upper-level Optimization ')
    plt.grid(True)
    # Optionally set y-axis limits if values are too small
    # plt.ylim(1e-6, 1e1)  # Adjust these values based on the actual data range

    # Test error plot
    plt.subplot(3, 1, 3)
    plt.semilogy(time_vec, tsa_vec, '-')
    plt.ylabel('Test error')
    plt.xlabel('time (s)')
    plt.title('Test Error')
    plt.grid(True)

    plt.tight_layout()
    # Save the plot before showing it
    plt.savefig('mlp_training_results.png')
    plt.show()

if __name__ == "__main__":
    main()