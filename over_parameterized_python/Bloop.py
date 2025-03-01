import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import hydra
import wandb
from omegaconf import OmegaConf
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, cfg.mlp.hidden_dim)
        self.fc2 = nn.Linear(cfg.mlp.hidden_dim, cfg.mlp.hidden_dim)
        self.fc3 = nn.Linear(cfg.mlp.hidden_dim, 10)
        
        self.dropout = nn.Dropout(cfg.mlp.dropout)
        self.batch_norm = cfg.mlp.batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(cfg.mlp.hidden_dim)
            self.bn2 = nn.BatchNorm1d(cfg.mlp.hidden_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

# Define the Lipschitz upper-bound auxiliary loss
def lipschitz_loss(model):
    return sum(torch.log(torch.norm(layer.weight.view(-1), p=2)) 
              for layer in [model.fc1, model.fc2, model.fc3])

def l2_norm_loss(model):
    total = 0.5 * sum(torch.sum(layer.weight ** 2) for layer in [model.fc1, model.fc2, model.fc3])
    print(f"L2 norm loss value: {total.item()}")  # Debug print
    return total

def test_loss(model, test_loader, criterion, device):
    """Compute test loss while maintaining the computational graph."""
    model.eval()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_samples = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        total_loss = total_loss + batch_loss * images.size(0)
        total_samples += images.size(0)
    
    model.train()
    return total_loss / total_samples  # Returns a tensor with gradient information

# DBGD algorithm functions
def params_to_vec(model):
    """Convert model parameters to a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()

def vec_to_params(model, vec, device):
    """Update model parameters using a vector."""
    vec_tensor = torch.FloatTensor(vec).to(device)
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = vec_tensor[pointer:pointer + num_param].view_as(param)
        pointer += num_param

def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    model.train()
    return accuracy

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(cfg):
    # Initialize wandb if logging is enabled
    if cfg.wandb.log:
        run = wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
            resume=cfg.wandb.resume,
            name=cfg.run_name
        )

    # Set random seed
    torch.manual_seed(cfg.rseed)
    np.random.seed(cfg.rseed)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # Determine batch size based on full_batch setting
    if cfg.full_batch:
        # Use the entire datasets as single batches
        train_batch_size = len(train_dataset)
        test_batch_size = len(test_dataset)
        print(f"Using full-batch mode - Train batch size: {train_batch_size}, Test batch size: {test_batch_size}")
    else:
        train_batch_size = cfg.batch_size
        test_batch_size = cfg.batch_size
        print(f"Using mini-batch mode - Batch size: {train_batch_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=cfg.learning_rate,
    )

    # Select the auxiliary loss function based on configuration
    print(f"Using auxiliary loss: {cfg.auxiliary_loss}")
    if cfg.auxiliary_loss == "lipschitz":
        aux_loss_fn = lambda m: lipschitz_loss(m)
    elif cfg.auxiliary_loss == "l2_norm":
        aux_loss_fn = lambda m: l2_norm_loss(m)
    elif cfg.auxiliary_loss == "test_loss":
        aux_loss_fn = lambda m: test_loss(m, test_loader, criterion, device)
    else:
        print(f"Unknown auxiliary loss type: {cfg.auxiliary_loss}, defaulting to lipschitz")
        aux_loss_fn = lambda m: lipschitz_loss(m)

    # Bloop algorithm parameters
    ema_decay = cfg.optimization.ema_decay
    lambda_aux = cfg.optimization.lambda_aux
    print(f"EMA decay: {ema_decay}")
    print(f"Lambda aux: {lambda_aux}")
    g_main_ema = None

    # Training loop
    main_losses = [] 
    aux_losses = [] 
    g_norms = []  # Renamed from g_main_norms
    train_accuracies = []
    test_accuracies = []
    weight_values = []
    grad_f_norm_values = []
    param_diff_norms = []
    global_step = 0
    
    # Store initial parameters
    prev_params = params_to_vec(model)
    
    # Print the optimization mode
    print(f"Optimization mode: {cfg.mode}")
    if cfg.mode == "dbgd":
        print(f"DBGD parameters - Alpha: {cfg.optimization.dbgd.alpha}, Beta: {cfg.optimization.dbgd.beta}, "
              f"Stepsize: {cfg.optimization.dbgd.stepsize}, Phi type: {cfg.optimization.dbgd.phi_type}")
    elif cfg.mode == "baseline":
        print(f"Baseline parameters - Auxiliary loss weight: {cfg.optimization.baseline.aux_weight}")

    for epoch in range(cfg.epochs):
        epoch_main_loss = 0
        epoch_aux_loss = 0
        num_batches = 0
        epoch_g_norm = 0  # Renamed from epoch_g_main_norm
        epoch_weight = 0  # For DBGD
        epoch_grad_f_norm_sq = 0  # For DBGD
        epoch_param_diff_norm = 0  # For parameter difference tracking
        
        # Training loop
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            if cfg.mode == "dbgd":
                # DBGD optimization step with selected auxiliary loss
                # Modify DBGD_step to use the selected auxiliary loss
                model.zero_grad()
                outputs = model(images)
                loss_main = criterion(outputs, labels)
                loss_aux = aux_loss_fn(model)
                
                # Compute gradient of main loss
                loss_main.backward(retain_graph=True)
                grad_f_x = torch.cat([p.grad.view(-1) for p in model.parameters()]).cpu().numpy()
                grad_f_norm_sq = np.dot(grad_f_x, grad_f_x)
                # Compute gradient of auxiliary loss
                model.zero_grad()
                loss_aux.backward()
                # Handle None gradients properly and move to CPU before converting to NumPy
                grad_g_x = torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) 
                                    for p in model.parameters()]).cpu().numpy()
                
                # Debug prints
                if cfg.auxiliary_loss == "l2_norm":
                    print(f"Raw gradients: {[p.grad.norm().item() if p.grad is not None else 0.0 for p in model.parameters()]}")
                    print(f"grad_g_x norm: {np.linalg.norm(grad_g_x)}")
                
                # DBGD algorithm parameters
                alpha = cfg.optimization.dbgd.alpha
                beta = cfg.optimization.dbgd.beta
                stepsize = cfg.optimization.dbgd.stepsize
                phi_type = cfg.optimization.dbgd.phi_type
                
                # Compute auxiliary function value
                g_x = loss_aux.item()
                grad_g_norm_sq = np.dot(grad_g_x, grad_g_x)
                
                # Compute phi based on the selected type
                if phi_type == 'DBGD':
                    phi = min(alpha * g_x, beta * grad_g_norm_sq)
                elif phi_type == 'BLOOP':
                    phi = beta * grad_g_norm_sq
                
                # Compute weight
                if grad_g_norm_sq == 0:
                    weight = 0
                else:
                    weight = max((phi - np.dot(grad_f_x, grad_g_x)) / grad_g_norm_sq, 0)
                
                # Compute update direction
                v = grad_f_x + weight * grad_g_x
                
                # Adaptive stepsize as in the original implementation
                adaptive_stepsize = stepsize / (global_step**(1/3) if global_step > 0 else 1)
                
                # Update parameters
                x = params_to_vec(model)
                x_new = x - adaptive_stepsize * v
                vec_to_params(model, x_new, device)
                
                # Compute parameter difference norm
                curr_params = params_to_vec(model)
                param_diff_norm = np.sum((curr_params - prev_params) ** 2)
                epoch_param_diff_norm += param_diff_norm * batch_size
                prev_params = curr_params
                
                loss_main = loss_main.item()
                loss_aux = loss_aux.item()
                
                epoch_main_loss += loss_main * batch_size
                epoch_aux_loss += loss_aux * batch_size
                epoch_g_norm += grad_g_norm_sq * batch_size  # Renamed from epoch_g_main_norm
                epoch_grad_f_norm_sq += grad_f_norm_sq * batch_size
                epoch_weight += weight * batch_size
                num_batches += batch_size
                
                # Log metrics to wandb every 20 batches
                if cfg.wandb.log and global_step % 20 == 0:
                    batch_train_acc = compute_accuracy(model, train_loader, device)
                    batch_test_acc = compute_accuracy(model, test_loader, device)
                    
                    wandb.log({
                        "main_loss": loss_main,
                        "aux_loss": loss_aux,
                        "g_norm": grad_g_norm_sq,
                        "grad_f_norm_sq": grad_f_norm_sq,
                        "weight": weight,
                        "train_accuracy": batch_train_acc,
                        "test_accuracy": batch_test_acc,
                        "step": global_step
                    })
            else:
                # Original optimization (standard, bilevel, or baseline)
                optimizer.zero_grad()
                outputs = model(images)
                loss_main = criterion(outputs, labels)
                loss_aux = aux_loss_fn(model)

                epoch_main_loss += loss_main.item() * batch_size
                epoch_aux_loss += loss_aux.item() * batch_size
                num_batches += batch_size

                # Log metrics to wandb every 20 batches
                if cfg.wandb.log and global_step % 20 == 0:
                    # Calculate accuracies
                    batch_train_acc = compute_accuracy(model, train_loader, device)
                    batch_test_acc = compute_accuracy(model, test_loader, device)
                    
                    wandb.log({
                        "main_loss": loss_main.item(),
                        "aux_loss": loss_aux.item(),
                        "train_accuracy": batch_train_acc,
                        "test_accuracy": batch_test_acc,
                        "step": global_step
                    })

                if cfg.mode == "standard":
                    # Standard gradient descent - only use main loss
                    loss_main.backward()
                    # Compute gradient norm
                    grad_norm = sum(torch.norm(p.grad, p=2)**2 for p in model.parameters() if p.grad is not None).item()
                    epoch_g_norm += grad_norm * batch_size
                    
                    # Store parameters before update
                    prev_params = params_to_vec(model)
                    
                    # Log gradient norm to wandb
                    if cfg.wandb.log and global_step % 20 == 0:
                        wandb.log({
                            "g_norm": grad_norm,
                            "step": global_step
                        })
                    
                    # Scale gradients by learning rate
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is not None:
                                # Debug print for gradient and parameter values
                                if epoch == 0 and global_step == 0:
                                    print(f"Standard mode - Param shape: {p.shape}, Grad norm: {torch.norm(p.grad).item()}, Param norm: {torch.norm(p.data).item()}")
                                    print(f"Learning rate: {cfg.learning_rate}")
                                p.data.add_(p.grad, alpha=-cfg.learning_rate * 100)  # Increase learning rate by 100x for testing
                    
                    # Compute parameter difference norm after update
                    curr_params = params_to_vec(model)
                    param_diff_norm = np.sum((curr_params - prev_params) ** 2)
                    # Debug print for parameter difference
                    if epoch == 0 and global_step == 0:
                        print(f"Standard mode - Param diff norm: {param_diff_norm}")
                    epoch_param_diff_norm += param_diff_norm * batch_size
                    prev_params = curr_params
                    
                    optimizer.zero_grad()
                    
                elif cfg.mode == "baseline":
                    # Baseline mode - optimize weighted sum of losses
                    total_loss = loss_main + cfg.optimization.baseline.aux_weight * loss_aux
                    total_loss.backward()
                    # Compute gradient norm
                    grad_norm = sum(torch.norm(p.grad, p=2)**2 for p in model.parameters() if p.grad is not None).item()
                    epoch_g_norm += grad_norm * batch_size
                    
                    # Store parameters before update
                    prev_params = params_to_vec(model)
                    
                    # Log gradient norm to wandb
                    if cfg.wandb.log and global_step % 20 == 0:
                        wandb.log({
                            "g_norm": grad_norm,
                            "step": global_step
                        })
                    
                    # Scale gradients by learning rate
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is not None:
                                # Debug print for gradient and parameter values
                                if epoch == 0 and global_step == 0:
                                    print(f"Baseline mode - Param shape: {p.shape}, Grad norm: {torch.norm(p.grad).item()}, Param norm: {torch.norm(p.data).item()}")
                                    print(f"Learning rate: {cfg.learning_rate}")
                                p.data.add_(p.grad, alpha=-cfg.learning_rate * 100)  # Increase learning rate by 100x for testing
                    
                    # Compute parameter difference norm after update
                    curr_params = params_to_vec(model)
                    param_diff_norm = np.sum((curr_params - prev_params) ** 2)
                    # Debug print for parameter difference
                    if epoch == 0 and global_step == 0:
                        print(f"Baseline mode - Param diff norm: {param_diff_norm}")
                    epoch_param_diff_norm += param_diff_norm * batch_size
                    prev_params = curr_params
                    
                    optimizer.zero_grad()

                if cfg.mode == "bilevel":
                    # Bilevel optimization mode
                    loss_main.backward(retain_graph=True)
                    g_main = [p.grad.clone() for p in model.parameters()]
                    
                    g_main_norm = sum(torch.norm(g, p=2)**2 for g in g_main).item() 
                    epoch_g_norm += g_main_norm * batch_size

                    # Log g_main_norm to wandb every 20 batches
                    if cfg.wandb.log and global_step % 20 == 0:
                        wandb.log({
                            "g_norm": g_main_norm,
                            "step": global_step
                        })

                    # Rest of bilevel optimization code...
                    if g_main_ema is None:
                        g_main_ema = [g.clone() for g in g_main]

                    optimizer.zero_grad()
                    loss_aux.backward()
                    g_aux = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
                    
                    with torch.no_grad():
                        for p, g_m, g_a, g_m_ema in zip(model.parameters(), g_main, g_aux, g_main_ema):
                            g_a_flat = g_a.view(-1)
                            g_m_ema_flat = g_m_ema.view(-1)
                            dot_product = torch.dot(g_a_flat, g_m_ema_flat)
                            norm_squared = torch.dot(g_m_ema_flat, g_m_ema_flat)
                            proj_g_aux = g_a - (dot_product / norm_squared) * g_m_ema
                            p.grad = g_m + lambda_aux * proj_g_aux
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    g_main_ema = [(1 - ema_decay) * g_ema + ema_decay * g for g_ema, g in zip(g_main_ema, g_main)]
                    
                    # Compute parameter difference norm after update for bilevel mode
                    curr_params = params_to_vec(model)
                    param_diff_norm = np.sum((curr_params - prev_params) ** 2)
                    
                    # Debug print for parameter difference
                    if epoch == 0 and global_step == 0:
                        print(f"Bilevel mode - Param diff norm: {param_diff_norm}")
                    
                    epoch_param_diff_norm += param_diff_norm * batch_size
                    prev_params = curr_params

            global_step += 1

        # Calculate epoch-level metrics (for printing only)
        train_acc = compute_accuracy(model, train_loader, device)
        test_acc = compute_accuracy(model, test_loader, device)
        
        main_losses.append(epoch_main_loss / num_batches)
        aux_losses.append(epoch_aux_loss / num_batches)
        g_norms.append(epoch_g_norm / num_batches)  # Renamed from g_main_norms
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if cfg.mode == "dbgd":
            weight_values.append(epoch_weight / num_batches)
            grad_f_norm_values.append(epoch_grad_f_norm_sq / num_batches)

        # Add average parameter difference norm for this epoch
        param_diff_norms.append(epoch_param_diff_norm / num_batches)
        
        # Update print statements to include parameter difference norm
        if cfg.mode == "dbgd":
            print(f"Epoch [{epoch+1}/{cfg.epochs}], "
                  f"Main Loss: {main_losses[-1]:.4f}, "
                  f"Aux Loss: {aux_losses[-1]:.4f}, "
                  f"Grad f Norm: {grad_f_norm_values[-1]:.4f}, "
                  f"Grad g Norm: {g_norms[-1]:.4f}, "  # Renamed from g_main_norms
                  f"Weight: {weight_values[-1]:.4f}, "
                  f"Param Diff Norm: {param_diff_norms[-1]:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{cfg.epochs}], "
                  f"Main Loss: {main_losses[-1]:.4f}, "
                  f"Aux Loss: {aux_losses[-1]:.4f}, "
                  f"Grad Norm: {g_norms[-1]:.4f}, "  # Renamed from g_main_norms
                  f"Param Diff Norm: {param_diff_norms[-1]:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")

    # Create a directory for plots based on the mode
    plot_dir = f"plots_{cfg.mode}_{cfg.auxiliary_loss}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save metrics data for comparison plots
    metrics_data = {
        'main_losses': main_losses,
        'aux_losses': aux_losses,
        'g_norms': g_norms,  # Renamed from g_main_norms
        'grad_f_norm_values': grad_f_norm_values if cfg.mode == "dbgd" else [],
        'weight_values': weight_values if cfg.mode == "dbgd" else [],
        'param_diff_norms': param_diff_norms,
        'auxiliary_loss_type': cfg.auxiliary_loss
    }
    np.save(f'{plot_dir}/metrics_{cfg.auxiliary_loss}.npy', metrics_data)
    
    # Different plotting for DBGD and other methods
    if cfg.mode == "dbgd":
        # 1. Main Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(main_losses) + 1), main_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Main Loss')
        plt.title(f'Main Loss vs Epoch (Aux: {cfg.auxiliary_loss})')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(f'{plot_dir}/main_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Auxiliary Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(aux_losses) + 1), aux_losses, 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Auxiliary Loss')
        plt.title(f'Auxiliary Loss ({cfg.auxiliary_loss}) vs Epoch')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(f'{plot_dir}/aux_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Gradient f Norm Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(grad_f_norm_values) + 1), grad_f_norm_values, 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('||grad_f||²')
        plt.title('||grad_f||² vs Epoch')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(f'{plot_dir}/grad_f_norm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Gradient g Norm Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(g_norms) + 1), g_norms, 'm-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('||grad_g||²')
        plt.title('||grad_g||² vs Epoch')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(f'{plot_dir}/grad_g_norm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Weight Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(weight_values) + 1), weight_values, 'c-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.title('DBGD Weight vs Epoch')
        plt.grid(True)
        plt.savefig(f'{plot_dir}/weight.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Train Accuracy Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Train Accuracy (%)')
        plt.title('Training Accuracy vs Epoch')
        plt.grid(True)
        plt.savefig(f'{plot_dir}/train_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Test Accuracy Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Testing Accuracy vs Epoch')
        plt.grid(True)
        plt.savefig(f'{plot_dir}/test_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combined accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='Train Accuracy')
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'r-', label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Testing Accuracy vs Epoch')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{plot_dir}/combined_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Loss trajectory plot
        fig, ax = plt.subplots(figsize=(10, 8))
        norm = plt.Normalize(0, len(main_losses))
        scatter = ax.scatter(main_losses, aux_losses, c=np.arange(len(main_losses)), 
                            cmap='viridis', norm=norm, alpha=0.8, s=100)
        ax.plot(main_losses, aux_losses, '-', color='gray', alpha=0.5, linewidth=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Main Loss (log scale)')
        ax.set_ylabel('Auxiliary Loss (log scale)')
        ax.set_title(f'Training Trajectory: Main Loss vs {cfg.auxiliary_loss} Loss')
        ax.grid(True)
        fig.colorbar(scatter, ax=ax, label='Training Progress')
        plt.savefig(f'{plot_dir}/loss_trajectory.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Parameter difference norm plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(param_diff_norms) + 1), param_diff_norms, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('||x_{k+1} - x_k||²')
        plt.title('Parameter Difference Norm vs Epoch')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(f'{plot_dir}/param_diff_norm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Standard plotting for non-DBGD methods
        # Loss trajectory plot
        fig, ax = plt.subplots(figsize=(10, 8))
        norm = plt.Normalize(0, len(main_losses))
        scatter = ax.scatter(main_losses, aux_losses, c=np.arange(len(main_losses)), 
                            cmap='viridis', norm=norm, alpha=0.8, s=100)
        ax.plot(main_losses, aux_losses, '-', color='gray', alpha=0.5, linewidth=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Main Loss (log scale)')
        ax.set_ylabel('Auxiliary Loss (log scale)')
        ax.set_title(f'Training Trajectory: Main Loss vs {cfg.auxiliary_loss} Loss')
        ax.grid(True)
        fig.colorbar(scatter, ax=ax, label='Training Progress')
        plt.savefig(f'{plot_dir}/loss_trajectory.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Parameter difference norm plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(param_diff_norms) + 1), param_diff_norms, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('||x_{k+1} - x_k||²')
        plt.title('Parameter Difference Norm vs Epoch')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(f'{plot_dir}/param_diff_norm.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Gradient norm plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(g_norms) + 1), g_norms, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('||g_main||²')
        plt.title('||g_main||² vs Epoch')
        plt.grid(True)
        plt.yscale('log')  
        plt.savefig(f'{plot_dir}/gradient_norm.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='Train Accuracy')
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'r-', label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Testing Accuracy vs Epoch')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{plot_dir}/accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

    if cfg.wandb.log:
        wandb.finish()

if __name__ == "__main__":
    main()

# To run with different optimization modes:
# Standard mode: python Bloop.py mode=standard
# Bilevel optimization (BLOOP): python Bloop.py mode=bilevel
# DBGD: python Bloop.py mode=dbgd
# Baseline mode: python Bloop.py mode=baseline
# 
# To customize DBGD parameters:
# python Bloop.py mode=dbgd optimization.dbgd.alpha=1.0 optimization.dbgd.beta=1.0 optimization.dbgd.stepsize=1e-4 optimization.dbgd.phi_type=BLOOP
#
# To use full-batch training:
# python Bloop.py full_batch=true
#
# To select different auxiliary losses:
# python Bloop.py auxiliary_loss=lipschitz  # Default Lipschitz norm loss
# python Bloop.py auxiliary_loss=l2_norm    # L2 norm of weights
# python Bloop.py auxiliary_loss=test_loss  # Loss on test dataset
#
# To customize baseline parameters:
# python Bloop.py mode=baseline optimization.baseline.aux_weight=0.1
#
# To run a sweep (only for bilevel optimization):
# wandb sweep config/train/sweep.yaml