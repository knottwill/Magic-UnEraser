from typing import Dict
import torch
import numpy as np

def linear_schedule(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """Returns pre-computed schedules for DDPM sampling with a linear noise schedule."""
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))  # Cumprod in log-space (better precision)

    return {"beta_t": beta_t, "alpha_t": alpha_t}

# def linear_schedule(beta_1: float, beta_T: float, T: int) -> Dict[str, torch.Tensor]:
#     """Returns pre-computed schedules for DDPM sampling with a linear noise schedule."""
#     assert 0 < beta_1 < beta_T < 1.0, "beta_1 and beta_T must be in (0, 1)"

#     beta_t = (beta_T - beta_1) * torch.arange(0, T, dtype=torch.float32) / T + beta_1
#     beta_t = torch.cat([torch.tensor([0.]), beta_t], dim=0)
#     alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))  # Cumprod in log-space (better precision)

#     return {"beta_t": beta_t, "alpha_t": alpha_t}

def cosine_schedule(T: int, s: float = 0.002) -> dict:
    """Returns pre-computed schedules for DDPM sampling with a cosine noise schedule.

    https://arxiv.org/pdf/2102.09672.pdf
    
    We clip beta_t to be no larger than 0.999 to prevent singularities at the end of the diffusion process near t = T. 

    We use a small offset s to prevent beta_t from being too small
    near t = 0, since we found that having tiny amounts of
    noise at the beginning of the process made it hard for the
    network to predict the noise accurately enough. In particular, we
    selected s such that sqrt(beta_1) was slightly smaller than the pixel
    bin size 1/255, which gives 0.002. 

    Should we also clip alpha_t to be no smaller than 0.001?
    """

    def f(t):
        return torch.pow(torch.cos((t/T + s) / (1 + s) * np.pi / 2), 2)
    
    
    timesteps = torch.arange(1, T + 1, dtype=torch.int64)

    alpha_t = f(timesteps)/f(torch.tensor(0))
    alpha_t = torch.cat([torch.tensor([1.0]), alpha_t], dim=0)

    beta_t = 1 - alpha_t[timesteps]/alpha_t[timesteps - 1]
    beta_t = torch.cat([torch.tensor([0.0]), beta_t], dim=0)

    # clip beta_t
    beta_t = torch.clamp(beta_t, 0.0, 0.999)

    # alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))

    return {"beta_t": beta_t, "alpha_t": alpha_t}