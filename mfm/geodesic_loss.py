def compute_geodesic_loss(self, x0, x1):
    t = torch.rand(x0.size(0), 1, device=x0.device)
    xt = self.interpolant(x0, x1, t)
    
    x_dot = x1 - x0 + t*(1-t)*self.interpolant.net(torch.cat([x0, x1, t], dim=1)) + \
            (1-2*t)*self.interpolant.net(torch.cat([x0, x1, t], dim=1)) + \
            t*(1-t)*(1-2*t)*torch.autograd.grad(
                self.interpolant.net(torch.cat([x0, x1, t], dim=1)).sum(),
                t,
                create_graph=True
            )[0]
    
    G = self.metric(xt)
    return torch.mean(torch.sum(x_dot * (G @ x_dot), dim=1))
