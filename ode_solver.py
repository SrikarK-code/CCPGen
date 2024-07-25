from scipy.integrate import solve_ivp
import numpy as np

class ODESolverModule:
    def __init__(self, model):
        self.model = model

    def ode_func(self, t, y, *args):
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        t_tensor = torch.tensor([t], dtype=torch.float32)
        with torch.no_grad():
            dy_dt = self.model(y_tensor, t_tensor, *args).squeeze().numpy()
        return dy_dt

    def solve(self, y0, t_span, *args, method='RK45', **kwargs):
        solution = solve_ivp(
            fun=lambda t, y: self.ode_func(t, y, *args),
            t_span=t_span,
            y0=y0,
            method=method,
            **kwargs
        )
        return solution