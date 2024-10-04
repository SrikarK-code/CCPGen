import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FlowMatchingTrainer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        init_type="gaussian",
        noise_scale=1.0,
        reflow_t_schedule="uniform",
        use_ode_sampler="euler",
        sigma_var=0.0,
        ode_tol=1e-5,
        sample_N=25,
    ):
        super().__init__()
        self.model = model
        self.init_type = init_type
        self.noise_scale = noise_scale
        self.reflow_t_schedule = reflow_t_schedule
        self.use_ode_sampler = use_ode_sampler
        self.sigma_var = sigma_var
        self.ode_tol = ode_tol
        self.sample_N = sample_N
        self.T = 1
        self.eps = 1e-3
        self.sigma_t = lambda t: (1.0 - t) * sigma_var

    def forward(self, x_0, prot_target):
        # x_0: X_diffmap embeddings [batch_size, diffmap_dim]
        # prot_target: ProtT5 embeddings [batch_size, seq_len, 1024]
        
        # Expand x_0 to match prot_target's sequence length
        x_0_expanded = x_0.unsqueeze(1).expand(-1, prot_target.shape[1], -1)
        
        t = torch.rand(x_0.shape[0], device=x_0.device) * (self.T - self.eps) + self.eps
        t_expand = t.view(-1, 1, 1).repeat(1, prot_target.shape[1], prot_target.shape[2])
        
        noise = torch.randn_like(prot_target)
        perturbed_target = t_expand * prot_target + (1 - t_expand) * noise
        
        model_out = self.model(x_0_expanded, t * 999)
        loss = F.mse_loss(model_out, prot_target, reduction="none").mean([1, 2]).mean()
        
        return loss

    @torch.no_grad()
    def euler_sample(self, cond, shape, guidance_scale):
        device = next(self.model.parameters()).device
        cond = cond.to(device)
        batch_size, seq_len, _ = shape
        x = torch.randn(batch_size, seq_len, self.model.out_channels, device=device)
        dt = 1.0 / self.sample_N
        eps = 1e-3
        for i in range(self.sample_N):
            num_t = i / self.sample_N * (self.T - eps) + eps
            t = torch.ones(batch_size, device=device) * num_t

            model_out = self.model(torch.cat([x] * 2), torch.cat([t.unsqueeze(1)] * 2), cond)
            noise_pred_uncond, noise_pred_text = model_out.chunk(2)
            pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            sigma_t = self.sigma_t(num_t)
            pred_sigma = pred + (sigma_t**2) / (2 * (self.noise_scale**2) * ((1.0 - num_t) ** 2)) * (
                0.5 * num_t * (1.0 - num_t) * pred - 0.5 * (2.0 - num_t) * x
            )

            x = x + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(x)

        return x, self.sample_N

class ProteinFlowMatching(nn.Module):
    def __init__(self, flow_matching, decoder):
        super().__init__()
        self.flow_matching = flow_matching
        self.decoder = decoder

    def forward(self, x, target):
        return self.flow_matching(x, target)  # training: loss from the flowmatching module

    def generate(self, x, num_steps=200):
        device = next(self.parameters()).device
        x = x.to(device)
        latent = self.flow_matching.euler_sample(x, (x.shape[0], num_steps, self.flow_matching.model.out_channels), guidance_scale=3.0)[0]
        protein_sequence = self.decoder(latent, max_length=num_steps)
        return protein_sequence
