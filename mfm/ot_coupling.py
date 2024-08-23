def compute_mfm_loss(self, x0, x1):
    t = torch.rand(x0.size(0), 1, device=x0.device)
    xt = self.interpolant(x0, x1, t)
    
    x_dot = x1 - x0 + t*(1-t)*self.interpolant.net(torch.cat([x0, x1, t], dim=1)) + \
            (1-2*t)*self.interpolant.net(torch.cat([x0, x1, t], dim=1)) + \
            t*(1-t)*(1-2*t)*torch.autograd.grad(
                self.interpolant.net(torch.cat([x0, x1, t], dim=1)).sum(),
                t,
                create_graph=True
            )[0]
    
    vt = self.vector_field(xt, t)
    G = self.metric(xt)
    return torch.mean(torch.sum((vt - x_dot) * (G @ (vt - x_dot)), dim=1))

def compute_ot_coupling(scvi_latents, prot_latents):
    cost_matrix = torch.cdist(scvi_latents, prot_latents, p=2)
    a = torch.ones(scvi_latents.shape[0]) / scvi_latents.shape[0]
    b = torch.ones(prot_latents.shape[0]) / prot_latents.shape[0]
    return torch.tensor(ot.emd(a, b, cost_matrix.cpu().numpy()))
