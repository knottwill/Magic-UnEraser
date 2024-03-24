"""!
@brief Contains all degradation operators used for Cold Diffusion
"""
import torch 
import torch.nn as nn
from .eraser_utils import all_pixels_in_trajectory, eraserhead_masks

class DegredationOperator(nn.Module):
    """!
    @brief Base class for all degradation operators
    """
    def __init__(self):
        super().__init__()
        self.sampling_mode = False
    
    def sampling(self, mode: bool = True):
        """!
        @brief Set the sampling mode of the degradation operator
        """
        self.sampling_mode = mode

    def train(self, mode: bool = True):
        super().train(mode)
        if mode: # if we are in training mode, we are not sampling
            self.sampling_mode = False

class Eraser(DegredationOperator):
    """!
    @brief Eraser Degredation Operator

    """
    def __init__(self, trajectory, sigma: float = 2, size: int = 28):
        super().__init__()

        pixels = all_pixels_in_trajectory(trajectory)
        single_masks = eraserhead_masks(pixels, sigma=sigma, size=size)
        masks = torch.exp(torch.cumsum(torch.log(single_masks), dim=0))
        self.register_buffer("masks", masks)
        
    def __call__(self, x: torch.Tensor, s: torch.Tensor):
        """
        s - severity (t/T) in [0, 1]
        """

        B, C, H, W = x.shape

        # if we are sampling, the colour and noise should be fixed
        if self.sampling_mode: 
            colours = self.mask_colours
            noise = self.noise
        else:
            colours = torch.rand((B,), device=x.device) - 0.5
            noise = torch.randn_like(x, device=x.device) * 0.02

        # get mask
        indices = torch.round(self.masks.shape[0]*s)
        indices = indices.clamp(0, self.masks.shape[0] - 1).int()
        mask = self.masks[indices, :, :, :]

        # apply mask
        degraded_x = x * mask + colours.view(-1, 1, 1, 1) * (1 - mask)

        # round all pixel values to the 8 most significant digits 
        # to avoid information leakage due to floating point errors
        degraded_x = (degraded_x * 1e8).round() / 1e8

        # add a small amount of noise to the fully-masked pixels
        noise = noise * (mask == 0)

        degraded_x = degraded_x + noise

        return degraded_x
    
    def latent_sampler(self, n_sample, img_shape, device):

        colours = torch.rand((n_sample,), device=device) - 0.5
        noise = torch.randn(n_sample, *img_shape, device=device) * 0.02

        # if we are sampling, we fix the colour and noise for the whole process
        if self.sampling_mode:
            self.mask_colours = colours
            self.noise = noise

        empty_img = torch.full((n_sample, *img_shape), -0.5, device=device)
        final_mask = self.masks[-1].repeat(n_sample, 1, 1, 1)
        base = empty_img * final_mask + colours.view(-1, 1, 1, 1) * (1 - final_mask)

        return base + noise
