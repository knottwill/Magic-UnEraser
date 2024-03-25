from typing import Dict
import torch
import torch.nn as nn
from .degradation.base import DegredationOperator


class DDPM(nn.Module):
    def __init__(
        self,
        net,
        noise_schedule: Dict[str, torch.Tensor],
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.net = net

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.beta_t  # Exists! Set by register_buffer
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])
        self.alpha_t

        self.T = len(self.beta_t) - 1 # beta_0 is not used
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Algorithm 18.1 in Prince
        x ~ (batch_size, 1, H W)
        """

        t = torch.randint(1, self.T + 1, (x.size(0),), device=x.device)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting

        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t. Loss is what we return.

        return self.criterion(eps, self.net(z_t, t / self.T))

    def sample(self, n_sample: int, img_shape = (1, 28, 28), device = 'cpu', visualise=False) -> torch.Tensor:
        """Algorithm 18.2 in Prince"""

        _one = torch.ones(n_sample, device=device)
        z_t = torch.randn(n_sample, *img_shape, device=device)
        Zs = torch.tensor([])
        for i in range(self.T, 0, -1):
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.net(z_t, (i/self.T) * _one)
            z_t /= torch.sqrt(1 - beta_t)

            if visualise:
                Zs = torch.cat((Zs, z_t.detach().cpu()), dim=1)

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of the algorithm)

        if visualise:
            return Zs
        
        return z_t
    

class ColdDiffusion(nn.Module):

    def __init__(
        self,
        reconstructor: nn.Module,
        degrador: DegredationOperator,
        criterion: nn.Module = nn.MSELoss(),
        T: int = 1000,
    ) -> None:
        super().__init__()

        self.degrador = degrador
        self.reconstructor = reconstructor

        self.T = T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        t = torch.randint(1, self.T + 1, (x.size(0),), device=x.device)
        s = t / self.T # severity

        z_t = self.degrador(x, s) # encoder
        x_recon = self.reconstructor(z_t, s) # decoder

        return nn.MSELoss()(x_recon, x)
    
    def sample(self, n_sample: int, img_shape = (1, 28, 28), device = 'cpu', cold_algorithm: bool = True, visualise: bool = False) -> torch.Tensor:

        self.degrador.sampling(True) # set degrador sampling mode to True

        _ones = torch.ones(n_sample, device=device)
        z_T = self.degrador.latent_sampler(n_sample, img_shape, device)

        z_t_sub_one = z_T
        Zs = torch.tensor([])
        for t in range(self.T, 0, -1):
            z_t = z_t_sub_one

            # detach z_t and concatenate to Z
            if visualise:
                Zs = torch.cat((Zs, z_t.detach().cpu()), dim=1)
                
            x_recon = self.reconstructor(z_t, (t * _ones)/self.T)

            if t > 1:

                d_t_sub_one = self.degrador(x_recon, ((t - 1) * _ones)/self.T)
                if cold_algorithm:
                    d_t = self.degrador(x_recon, (t * _ones)/self.T)
                    z_t_sub_one = z_t - d_t + d_t_sub_one
                else:
                    z_t_sub_one = d_t_sub_one
            
        self.degrador.sampling(False) # set degrador sampling mode to False

        if visualise:
            Zs = torch.cat((Zs, x_recon.detach().cpu()), dim=1)
            return Zs
        
        return x_recon
