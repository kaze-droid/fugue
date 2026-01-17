import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Config
GRID_SIZE = 16  # 16x16 = 256 blocks
LATENT_DIM = 32

class DefragVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 256 -> 128 -> 64 -> Latent
        self.encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, LATENT_DIM)
        self.fc_logvar = nn.Linear(64, LATENT_DIM)
        
        # Decoder: Latent -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid() # Output values between 0 and 1
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def generate_data(batch_size=64):
    """
    Creates messy grids (input) and sorted grids (target).
    0.0 = Empty, 0.5 = User, 1.0 = System
    """
    inputs = []
    targets = []
    for _ in range(batch_size):
        grid = np.zeros(256)
        num_sys = np.random.randint(5, 20)
        num_user = np.random.randint(20, 100)
        
        # Fill messy input
        indices = np.random.choice(256, num_sys + num_user, replace=False)
        grid[indices[:num_sys]] = 1.0
        grid[indices[num_sys:]] = 0.5
        inputs.append(grid)
        
        # Create sorted target
        target = np.zeros(256)
        target[:num_sys] = 1.0
        target[num_sys:num_sys+num_user] = 0.5
        targets.append(target)
        
    return torch.FloatTensor(np.array(inputs)), torch.FloatTensor(np.array(targets))

# Training Loop
model = DefragVAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(1000):
    inputs, targets = generate_data(128)
    optimizer.zero_grad()
    
    recon, mu, logvar = model(inputs)
    
    # Loss = Reconstruction Error + KL Divergence
    recon_loss = criterion(recon, targets)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + (kl_loss * 0.0001) # Keep KL weight low to prioritize recon
    
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Export to ONNX
dummy_input = torch.randn(1, 256)
torch.onnx.export(model, dummy_input, "memory_vae.onnx", 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
