import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Lipschitz upper-bound auxiliary loss
def lipschitz_loss(model):
    return sum(torch.log(torch.norm(layer.weight.view(-1), p=2)) 
              for layer in [model.fc1, model.fc2, model.fc3])


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Bloop algorithm parameters
lambda_aux = 0.2  # Trade-off hyperparameter
ema_decay = 0.05  # EMA decay parameter
g_main_ema = None  # Exponential Moving Average of the training gradient

# Training loop
num_epochs = 100
start_epoch = 0
main_losses = [] 
aux_losses = [] 
g_main_norms = [] 

# Load checkpoint if it exists
checkpoint_path = 'bloop_checkpoint.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    g_main_ema = checkpoint['g_main_ema']
    main_losses = checkpoint['main_losses']
    aux_losses = checkpoint['aux_losses']
    g_main_norms = checkpoint.get('g_main_norms', [])  
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")

for epoch in range(start_epoch, num_epochs):
    epoch_main_loss = 0
    epoch_aux_loss = 0
    num_batches = 0
    epoch_g_main_norm = 0 
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss_main = criterion(outputs, labels)
        loss_aux = lipschitz_loss(model)

        epoch_main_loss += loss_main.item() * batch_size
        epoch_aux_loss += loss_aux.item() * batch_size
        num_batches += batch_size

        loss_main.backward(retain_graph=True)
        g_main = [p.grad.clone() for p in model.parameters()]
        
   
        g_main_norm = sum(torch.norm(g, p=2)**2 for g in g_main).item() 
        epoch_g_main_norm += g_main_norm * batch_size

        loss_aux.backward()
        g_aux = [p.grad.clone() for p in model.parameters()]
        optimizer.zero_grad()  
        
       
        if g_main_ema is None:
            g_main_ema = [g.clone() for g in g_main]
        else:
            g_main_ema = [(1 - ema_decay) * g_ema + ema_decay * g for g_ema, g in zip(g_main_ema, g_main)]

      
        with torch.no_grad():
            for p, g_m, g_a, g_m_ema in zip(model.parameters(), g_main, g_aux, g_main_ema):
                g_a_flat = g_a.view(-1)
                g_m_ema_flat = g_m_ema.view(-1)
                dot_product = torch.dot(g_a_flat, g_m_ema_flat)
                norm_squared = torch.dot(g_m_ema_flat, g_m_ema_flat)
                proj_g_aux = g_a - (dot_product / norm_squared) * g_m_ema
                p.grad = g_m + lambda_aux * proj_g_aux
        
        optimizer.step()
    

    main_losses.append(epoch_main_loss / num_batches)
    aux_losses.append(epoch_aux_loss / num_batches)
    g_main_norms.append(epoch_g_main_norm / num_batches)
    print(f"Epoch [{epoch+1}/{num_epochs}], Main Loss: {main_losses[-1]:.4f}, "
          f"Aux Loss: {aux_losses[-1]:.4f}, Grad Norm: {g_main_norms[-1]:.4f}")
    
  
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'g_main_ema': g_main_ema,
            'main_losses': main_losses,
            'aux_losses': aux_losses,
            'g_main_norms': g_main_norms,  
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")


final_model_path = 'bloop_model_final.pt'
torch.save(model.state_dict(), final_model_path)

checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'g_main_ema': g_main_ema,
    'main_losses': main_losses,
    'aux_losses': aux_losses,
    'g_main_norms': g_main_norms, 
}
torch.save(checkpoint, checkpoint_path)
print("Final model and checkpoint saved")


fig, ax = plt.subplots(figsize=(10, 8))

norm = plt.Normalize(0, len(main_losses))

scatter = ax.scatter(main_losses, aux_losses, c=np.arange(len(main_losses)), 
                    cmap='viridis', norm=norm, alpha=0.8, s=100)

ax.plot(main_losses, aux_losses, '-', color='gray', alpha=0.5, linewidth=1)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Main Loss (log scale)')
ax.set_ylabel('Auxiliary Loss (log scale)')
ax.set_title('Training Trajectory: Main Loss vs Auxiliary Loss')
ax.grid(True)


fig.colorbar(scatter, ax=ax, label='Training Progress')


plt.savefig('loss_trajectory.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(g_main_norms) + 1), g_main_norms, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('||g_main||²')
plt.title('||g_main||² vs Epoch')
plt.grid(True)
plt.yscale('log')  
plt.savefig('gradient_norm.png', dpi=300, bbox_inches='tight')
plt.close()
