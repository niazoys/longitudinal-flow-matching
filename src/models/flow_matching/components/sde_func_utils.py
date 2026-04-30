import ipdb
import torch

def _sde_solver(sde, initial_state, tau_span,dim=4096):
    dt = tau_span[1] - tau_span[0]
    current_state = initial_state
    trajectory = [current_state[:,:dim]]
    noise_trajectory = []
    for t in tau_span[1:]:
        # drift = sde.f(t, current_state)
        # diffusion = sde.g(t, current_state)
        # if sde.time_scaler is not None:
        #     dt = sde.time_scaler(dt)
        drift, score,diffusion = sde.f_g(t, current_state)
        noise = torch.randn_like(diffusion) * diffusion * (dt)
        diffusion= torch.tensor(1.0).to(diffusion.device)
        tmp_current_state = current_state[:,:dim] + (drift  + score*(diffusion)/2)*dt #+ noise
        trajectory.append(tmp_current_state)
        noise_trajectory.append(diffusion * noise)
        current_state = torch.cat([tmp_current_state, current_state[:, dim:]], 1)
    return torch.stack(trajectory), torch.stack(noise_trajectory)


def new_sde_solver(sde, initial_state, tau_span,dim=4096,int_type='ode',uncert_w=0.1):
    dt = tau_span[1] - tau_span[0]
    current_state = initial_state
    trajectory = [current_state[:,:dim]]
    noise_trajectory = []
    for t in tau_span[1:]:
        # drift = sde.f(t, current_state)
        # diffusion = sde.g(t, current_state)
        # if sde.time_scaler is not None:
        #     dt = sde.time_scaler(dt)
        drift, score,diffusion = sde.f_g(t, current_state)
        
        if int_type == 'ode':
            score = torch.zeros_like(score).to(score.device)
            diffusion = torch.zeros_like(diffusion).to(diffusion.device)
        elif int_type == 'sde':
            diffusion = torch.ones_like(diffusion).to(diffusion.device)
        elif int_type == 'sde_uncertainty':
            diffusion= diffusion * torch.tensor(uncert_w).to(diffusion.device)
        else:
            raise ValueError(f"Unknown int_type: {int_type}")
        
        noise = torch.randn_like(diffusion) * diffusion * dt
        tmp_current_state = current_state[:,:dim] + (drift  + score*(diffusion)/2)*dt + noise
        
        
        trajectory.append(tmp_current_state)
        noise_trajectory.append(diffusion * noise)
        current_state = torch.cat([tmp_current_state, current_state[:, dim:]], 1)
    return torch.stack(trajectory), torch.stack(noise_trajectory)




class SDE_func_solver(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, noise=None, dim=4096,time_scaler=None,reverse=False):
        super().__init__()
        self.drift = ode_drift
        self.reverse = reverse
        self.noise = noise
        self.dim = dim
        self.time_scaler = time_scaler

    def f_g(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        out=self.drift(x)
        # ipdb.set_trace()
        return out[:, :self.dim], out[:, self.dim:2*self.dim],out[:, 2*self.dim:3*self.dim] 

    def f(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        return self.drift(x)[:, :self.dim]

    def g(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        if self.noise is None:
            noise_result= self.drift(x)[:, self.dim:2*self.dim]
        else:
            noise_result = self.noise(x)

        return noise_result #* torch.sqrt(torch.tensor(0.1)) #This due to osccilating schedule of sigma during training #torch.sqrt(t * (1 - t)) #sigma need to change?
    





    
class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, noise=1.0, reverse=False):
        super().__init__()
        self.drift = ode_drift
        self.reverse = reverse
        self.noise = noise

    def f(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        
        return self.drift(x)

    def g(self, t, y):
        return torch.ones_like(t) * torch.ones_like(y) * self.noise