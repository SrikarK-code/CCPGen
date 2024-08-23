import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import ot
from torchdiffeq import odeint

class RBFMetric(nn.Module):
    def __init__(self, dim, K, kappa=0.5, epsilon=1e-4):
        super().__init__()
        self.dim = dim
        self.K = K
        self.kappa = kappa
        self.epsilon = epsilon
        self.centers = nn.Parameter(torch.randn(K, dim))
        self.log_omega = nn.Parameter(torch.zeros(dim, K))
        self.log_lambda = nn.Parameter(torch.zeros(dim, K))

    def forward(self, x):
        omega = F.softplus(self.log_omega)
        lambda_sq = F.softplus(self.log_lambda)
        dists = torch.cdist(x, self.centers, p=2)
        h = torch.sum(omega.T * torch.exp(-lambda_sq.T * dists**2), dim=1)
        return torch.diag(1.0 / (h + self.epsilon)**8)

    def train_metric(self, data):
        # K-means clustering for centers (simplified)
        with torch.no_grad():
            self.centers.data = data[torch.randperm(len(data))[:self.K]]
        
        optimizer = Adam(self.parameters(), lr=1e-3)
        for _ in range(1000):
            dists = torch.cdist(data, self.centers, p=2)
            omega = F.softplus(self.log_omega)
            lambda_sq = F.softplus(self.log_lambda)
            h = torch.sum(omega.T * torch.exp(-lambda_sq.T * dists**2), dim=1)
            loss = torch.mean((1 - h)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

class MFMInterpolant(nn.Module):
    def __init__(self, scvi_dim, prot_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scvi_dim + prot_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prot_dim)
        )

    def forward(self, x0, x1, t):
        t = t.view(-1, 1)
        input_tensor = torch.cat([x0, x1, t], dim=1)
        return (1 - t) * x0 + t * x1 + t * (1 - t) * self.net(input_tensor)

class VectorField(nn.Module):
    def __init__(self, scvi_dim, prot_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scvi_dim + prot_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prot_dim)
        )

    def forward(self, x, t):
        t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=1))

class MetricFlowMatching(nn.Module):
    def __init__(self, scvi_dim, prot_dim, hidden_dim, K):
        super().__init__()
        self.scvi_dim = scvi_dim
        self.prot_dim = prot_dim
        self.interpolant = MFMInterpolant(scvi_dim, prot_dim, hidden_dim)
        self.vector_field = VectorField(scvi_dim, prot_dim, hidden_dim)
        self.metric = RBFMetric(prot_dim, K)

    def compute_geodesic_loss(self, x0, x1):
        t = torch.rand(x0.size(0), 1, device=x0.device)
        xt = self.interpolant(x0, x1, t)
        
        x_dot = x1 - x0 + t*(1-t)*self.interpolant.net(torch.cat([x0, x1, t], dim=1)) + \
                (1-2*t)*self.interpolant.net(torch.cat([x0, x1, t], dim=1))
        
        G = self.metric(xt)
        return torch.mean(torch.sum(x_dot * (G @ x_dot), dim=1))

    def compute_mfm_loss(self, x0, x1):
        t = torch.rand(x0.size(0), 1, device=x0.device)
        xt = self.interpolant(x0, x1, t)
        
        x_dot = x1 - x0 + t*(1-t)*self.interpolant.net(torch.cat([x0, x1, t], dim=1)) + \
                (1-2*t)*self.interpolant.net(torch.cat([x0, x1, t], dim=1))
        
        vt = self.vector_field(xt, t)
        G = self.metric(xt)
        return torch.mean(torch.sum((vt - x_dot) * (G @ (vt - x_dot)), dim=1))

def compute_ot_coupling(scvi_latents, prot_latents):
    cost_matrix = torch.cdist(scvi_latents, prot_latents, p=2)
    a = torch.ones(scvi_latents.shape[0]) / scvi_latents.shape[0]
    b = torch.ones(prot_latents.shape[0]) / prot_latents.shape[0]
    return torch.tensor(ot.emd(a, b, cost_matrix.cpu().numpy()))

def train_mfm(mfm, scvi_latents, prot_latents, n_epochs=1000, batch_size=128, lr=1e-4):
    optimizer = Adam(mfm.parameters(), lr=lr)
    
    mfm.metric.train_metric(prot_latents)
    coupling = compute_ot_coupling(scvi_latents, prot_latents)
    
    for epoch in range(n_epochs):
        # Step 1: Train interpolants
        indices = torch.multinomial(coupling.flatten(), batch_size, replacement=True)
        i, j = indices // coupling.shape[1], indices % coupling.shape[1]
        x0, x1 = scvi_latents[i], prot_latents[j]
        
        loss_geodesic = mfm.compute_geodesic_loss(x0, x1)
        optimizer.zero_grad()
        loss_geodesic.backward()
        optimizer.step()
        
        # Step 2: Train vector field
        loss_mfm = mfm.compute_mfm_loss(x0, x1)
        optimizer.zero_grad()
        loss_mfm.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Geodesic Loss: {loss_geodesic.item():.4f}, MFM Loss: {loss_mfm.item():.4f}")

class ODEFunc(nn.Module):
    def __init__(self, mfm):
        super().__init__()
        self.mfm = mfm
    
    def forward(self, t, x):
        return self.mfm.vector_field(x, t.reshape(1, 1))

def sample_mfm(mfm, scvi_latent, t_span=[0, 1], method='dopri5'):
    ode_func = ODEFunc(mfm)
    t = torch.linspace(t_span[0], t_span[1], 100)
    traj = odeint(ode_func, scvi_latent, t, method=method)
    return traj[-1]

# Usage example
scvi_dim = 50  # Example dimension
prot_dim = 100  # Example dimension
hidden_dim = 256
K = 100  # Number of centers for RBF metric

mfm = MetricFlowMatching(scvi_dim, prot_dim, hidden_dim, K)

# Assume scvi_latents and prot_latents are already encoded
train_mfm(mfm, scvi_latents, prot_latents)

# Generate new ProtT5 latents from scVI latents
new_scvi_latent = scvi_encoder(new_scvi_data)  # Encode new scVI data
new_prot_latent = sample_mfm(mfm, new_scvi_latent)

# Decode new ProtT5 latent to sequence
new_protein_sequence = prot_decoder(new_prot_latent)
