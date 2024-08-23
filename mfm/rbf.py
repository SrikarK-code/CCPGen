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
        return torch.diag(1.0 / (h + self.epsilon))

    def train_metric(self, data):
        # K-means clustering for centers
        kmeans = KMeans(n_clusters=self.K)
        kmeans.fit(data.cpu().numpy())
        self.centers.data = torch.tensor(kmeans.cluster_centers_, device=data.device)
        
        # Compute bandwidths
        cluster_labels = kmeans.predict(data.cpu().numpy())
        for k in range(self.K):
            cluster_points = data[cluster_labels == k]
            if len(cluster_points) > 0:
                bandwidth = 1 / (2 * (self.kappa / len(cluster_points) * torch.sum(torch.cdist(cluster_points, self.centers[k].unsqueeze(0), p=2)**2)))
                self.log_lambda.data[:, k] = torch.log(torch.full((self.dim,), bandwidth.item()))

        # Train weights
        optimizer = Adam(self.parameters(), lr=1e-3)
        for _ in range(1000):
            h = self(data)
            loss = torch.mean((1 - torch.diag(h))**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
