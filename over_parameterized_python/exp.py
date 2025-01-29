import numpy as np
import scipy.io as sio
import cvxpy as cp
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128):
#         super(MLP, self).__init__()
#         self.layer1 = nn.Linear(input_dim, hidden_dim, bias=True)
#         self.layer2 = nn.Linear(hidden_dim, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()
        
#         nn.init.xavier_normal_(self.layer1.weight)
#         nn.init.xavier_normal_(self.layer2.weight)
#         nn.init.zeros_(self.layer1.bias)
#         nn.init.zeros_(self.layer2.bias)
        
#     def forward(self, x):
#         x = self.sigmoid(self.layer1(x))
#         x = self.layer2(x)
#         return x
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MLP, self).__init__()
        # Option 1: Deeper network with multiple hidden layers
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim, bias=True)  # Additional hidden layer
        self.layer3 = nn.Linear(hidden_dim, 1, bias=True)  # Output layer
        self.sigmoid = nn.Sigmoid()  # or try nn.Tanh()
        
        # Initialize weights
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.xavier_normal_(self.layer3.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)
        
    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        x = self.layer3(x)  # No activation on output layer
        return x

def params_to_vec(model):
    """Convert model parameters to a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu()

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

def DBGD(fun_f, grad_f, grad_g, fun_g, TSA, param, x0, phi_type):
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
    grad_f_vec = []
    grad_g_vec = []
    time_vec1 = []
    acc_vec = []
    d_vec = []
    weight_vec = []
    iter_count = 0
    print(f"Starting DBGD with {phi_type} ")
    while iter_count <= maxiter:
        x_prev = x
        current_time = time.time() - start_time
        if current_time >= maxtime:
            print(f"Stopping at time {current_time:.2f}s")
            break
            
        iter_count += 1
        
        grad_f_x = grad_f(x)
        grad_g_x = grad_g(x)
        grad_f_vec.append(grad_f_x)
        grad_g_vec.append(grad_g_x)
        
        g_x = fun_g(x)
        grad_g_norm_sq = np.dot(grad_g_x, grad_g_x)
        if phi_type == 'DBGD':
            phi = min(alpha * g_x, beta * grad_g_norm_sq)
        elif phi_type == 'BLOOP':
            phi = beta * grad_g_norm_sq
        
        if grad_g_norm_sq == 0:
            weight = 0
        else:
            weight = max((phi - np.dot(grad_f_x, grad_g_x)) / grad_g_norm_sq, 0)
        weight_vec.append(weight)
        
        v = grad_f_x + weight * grad_g_x
        stepsize = stepsize / iter_count**(1/3)
        x = x - stepsize * v
        # x = ProjectOntoL1Ball(x, lambda_)
        
        f_vec1.append(fun_f(x))
        g_vec1.append(fun_g(x))
        d_vec.append(np.linalg.norm(v))
        time_vec1.append(current_time)
        acc_vec.append(TSA(x))
        
        if iter_count % 100 == 0:
            print(f"Iteration: {iter_count}, Time: {current_time:.2f}s")
    
    return (np.array(f_vec1), np.array(g_vec1), np.array(time_vec1), 
            x, np.array(acc_vec), np.array(d_vec), np.array(grad_f_vec), np.array(grad_g_vec), np.array(weight_vec))

def main():
    # Add device configuration at the start
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    np.random.seed(42)
    data = sio.loadmat('wikivital_mathematics.mat')
    # Move tensors to GPU
    A = torch.FloatTensor(data['A'].toarray()).to(device)
    b = torch.FloatTensor(data['b']).reshape(-1, 1).to(device)
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

    # Initialize models and move to GPU
    model_f = MLP(input_dim=n).to(device)
    model_g = MLP(input_dim=n).to(device)

    # Update objective functions to handle GPU tensors
    def fun_f(x):
        vec_to_params(model_f, torch.FloatTensor(x).to(device))
        pred = model_f(A1)
        return float(torch.sum((pred - b1)**2)) / 2

    def fun_g(x):
        vec_to_params(model_g, torch.FloatTensor(x).to(device))
        pred = model_g(A2)
        return float(torch.sum((pred - b2)**2)) / 2

    def grad_f(x):
        x_tensor = torch.FloatTensor(x).to(device)
        
        vec_to_params(model_f, x_tensor)
        x_tensor = torch.cat([p.data.view(-1) for p in model_f.parameters()]).to(device)
        x_tensor.requires_grad_(True)
        # Clear existing gradients
        model_f.zero_grad()
        
        pred = model_f(A1)
        loss = torch.sum((pred - b1)**2) / 2
        loss.backward()
        
        grad = torch.cat([p.grad.view(-1) for p in model_f.parameters()])
        return grad.cpu().detach().numpy()

    def grad_g(x):
        x_tensor = torch.FloatTensor(x).to(device)
        
        vec_to_params(model_g, x_tensor)
        x_tensor = torch.cat([p.data.view(-1) for p in model_g.parameters()]).to(device)
        x_tensor.requires_grad_(True)
        
        # Clear existing gradients
        model_g.zero_grad()
        
        pred = model_g(A2)
        loss = torch.sum((pred - b2)**2) / 2
        loss.backward()
        
        grad = torch.cat([p.grad.view(-1) for p in model_g.parameters()])
        return grad.cpu().detach().numpy()

    def TSA_LS(x):
        vec_to_params(model_f, torch.FloatTensor(x).to(device))
        pred = model_f(A3)
        return float(torch.sum((pred - b3)**2)) / 2

    # DBGD parameters
    stepsize = 1e-4
    param = {
        'stepsize': stepsize,
        'alpha': 1,
        'beta': 1,
        'lam': 1,
        'maxiter': int(1e7),
        'maxtime': 10
    }

    phi_type = 'BLOOP'
    # Initialize
    total_params = sum(p.numel() for p in model_f.parameters())
    init_scale = 0
    # x0 = init_scale * np.ones(total_params)
    x0 = np.random.randn(total_params) * 0.01

    # Run DBGD
    print('DBGD Algorithm starts')
    
    f_vec, g_vec, time_vec, x_final, tsa_vec, d_vec, grad_f_vec, grad_g_vec, weight_vec = DBGD(
        fun_f, grad_f, grad_g, fun_g, TSA_LS, param, x0, phi_type
    )
    print('DBGD Solution Achieved!')
    # Calculate norm of grad_g at each iteration
    grad_g_norm = [np.linalg.norm(g) for g in grad_g_vec]
    grad_f_norm = [np.linalg.norm(f) for f in grad_f_vec]
    # weight_norm = [np.linalg.norm(w) for w in weight_vec]
    

    # Plotting
    plt.figure(figsize=(10, 12))


    plt.subplot(6, 1, 1)
    plt.semilogy(time_vec, d_vec, '-')
    plt.ylabel('||d_k||')
    plt.xlabel('time (s)')
    plt.title('||x_k - x_{k-1}||')
    plt.grid(True)
    plt.xlim(0, param['maxtime'])
    print(d_vec)

    plt.subplot(6, 1, 2)
    plt.semilogy(time_vec, grad_f_norm, '-')
    plt.ylabel('||grad_f(βk)||')
    plt.xlabel('time (s)')
    plt.title('||grad_f(βk)||')
    plt.grid(True)
    plt.xlim(0, param['maxtime'])


    plt.subplot(6, 1, 3)
    plt.semilogy(time_vec, grad_g_norm, '-')
    plt.ylabel('||grad_g(βk)||')
    plt.xlabel('time (s)')
    plt.title('||grad_g(βk)||')
    plt.grid(True)
    plt.xlim(0, param['maxtime'])



    plt.subplot(6, 1, 4)
    plt.semilogy(time_vec, f_vec, '-')
    plt.ylabel('f(βk)')
    plt.xlabel('time (s)')
    plt.title('f(βk)')
    plt.grid(True)
    plt.xlim(0, param['maxtime'])

    plt.subplot(6, 1, 5)
    plt.semilogy(time_vec, g_vec, '-')
    plt.ylabel('g(βk)')
    plt.xlabel('time (s)')
    plt.title('g(βk)')
    plt.grid(True)
    plt.xlim(0, param['maxtime'])

    plt.subplot(6, 1, 6)
    plt.semilogy(time_vec, weight_vec, '-')
    plt.ylabel('weight')
    plt.xlabel('time (s)')
    plt.title('weight')
    plt.grid(True)
    plt.xlim(0, param['maxtime'])

    # # Lower-level gap plot
    # plt.subplot(3, 1, 1)
    # plt.semilogy(time_vec, g_vec, '-')
    # plt.ylabel('g(βk)')
    # plt.xlabel('time (s)')
    # plt.title('Lower-level Optimization ')
    # plt.grid(True)
    # plt.xlim(0, param['maxtime'])

    # # Upper-level gap plot
    # plt.subplot(3, 1, 2)
    # plt.semilogy(time_vec, f_vec, '-')
    # plt.ylabel('f(βk)')
    # plt.xlabel('time (s)')
    # plt.title('Upper-level Optimization ')
    # plt.grid(True)
    # plt.xlim(0, param['maxtime'])

    # Test error plot
    # plt.subplot(3, 1, 3)
    # plt.semilogy(time_vec, tsa_vec, '-')
    # plt.ylabel('Test error')
    # plt.xlabel('time (s)')
    # plt.title('Test Error')
    # plt.grid(True)
    # plt.xlim(0, param['maxtime'])

    plt.tight_layout()
    # Save the plot with hidden_dim and stepsize in the filename
    hidden_dim = model_f.layer1.out_features  # Get hidden dimensions
    stepsize = param['stepsize']
    print(hidden_dim)
    print(stepsize)
    import os
    dir_name = f'random_init'
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(os.path.join(dir_name, f'init{init_scale}_hidden{hidden_dim}_lr{str(stepsize)}_phi{phi_type}.png'))
    print(f"Saved plot to {dir_name}/init{init_scale}_hidden{hidden_dim}_lr{str(stepsize)}_phi{phi_type}.png")
    plt.show()

if __name__ == "__main__":
    main()