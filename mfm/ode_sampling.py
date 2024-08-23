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
