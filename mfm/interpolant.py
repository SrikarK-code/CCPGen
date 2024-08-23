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
        return (1 - t) * x0 + t * x1 + t * (1 - t) * self.net(torch.cat([x0, x1, t], dim=1))
