"""!@file eraser.py
@brief Contains the Eraser Degredation Operator

@details The idea is that the operator simulates the effect of an eraser 'rubbing out the image', where the eraser head
is moved across the image in a zigzag pattern. This operator applies a mask to the input image, with severity controlled
by the variable `s`. The mask is generated by a series of individual Eraser head masks, which are solid disks of radius
`eraserhead_radius` with a Gaussian falloff at the edges with standard deviation `sigma`. The trajectory of the eraser head
is defined by the `trajectory` parameter, which is a list of pixel coordinates at the edges of the image, that the eraser
head moves between in straight lines. First, we calculate all pixels along this trajectory using the `all_pixels_in_trajectory`
function. The mask is generated by applying the individual eraser head masks at each pixel in the trajectory, and multiplying
them together. The severity parameter `s` (= t/T) is a float in the range [0, 1], which controls how far along the trajectory the
eraser head has moved. For `s = 0`, the mask is the identity, and for `s = 1`, the mask is the final mask, leaving no trace
of the original image. To ensure that the latent space is diverse, we make it so that the background image (the image left
behind by the eraser) is a solid random colour with noise added. This noise is gaussian with standard deviation `noise_std`.
The `__call__` method applies the degradation operator to the input image `x` with severity `s`.

Degradation operators are required in the training and sampling process of a cold diffusion model. If we are
sampling from the model, we need to fix the background image for the whole generation process. The sampling mode is set
using the `sampling` method from the DegredationOperator base class. The `latent_sampler` method
generates a batch of background images for the given batch size and image shape. If we are in sampling mode, then it fixes
these images as the background for the whole process.
"""
import torch
from .base import DegredationOperator
from .eraser_utils import all_pixels_in_trajectory, eraserhead_masks


class Eraser(DegredationOperator):
    """!
    @brief Eraser Degredation Operator
    """

    def __init__(self, trajectory, eraserhead_radius: int = 3, sigma: float = 3, noise_std: float = 0.02, size: int = 28):
        super().__init__()

        # generate masks
        pixels = all_pixels_in_trajectory(trajectory)
        single_masks = eraserhead_masks(pixels, eraserhead_radius, sigma, size)
        masks = torch.exp(torch.cumsum(torch.log(single_masks), dim=0))
        self.register_buffer("masks", masks)
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor, s: torch.Tensor):
        """
        s - severity (t/T) in [0, 1]

        The index of the mask to apply is given by s * M, where M is the number of pixels/masks in the trajectory.
        """

        batch_size, C, H, W = x.shape

        # if we are sampling, the background should already be specified
        if self.sampling_mode:
            background = self.background
        else:
            colours = torch.rand((batch_size,), device=x.device) - 0.5
            noise = torch.randn_like(x, device=x.device) * self.noise_std
            background = colours.view(-1, 1, 1, 1) + noise

        # get mask
        indices = torch.round(self.masks.shape[0] * s)
        indices = indices.clamp(0, self.masks.shape[0] - 1).int()
        mask = self.masks[indices, :, :, :]

        # apply mask
        degraded_x = x * mask + background * (1 - mask)

        # round all pixel values to the 8 most significant digits
        # to avoid information leakage due to floating point errors
        degraded_x = (degraded_x * 1e8).round() / 1e8

        return degraded_x

    def latent_sampler(self, n_sample, img_shape, device):
        colours = torch.rand((n_sample,), device=device) - 0.5
        noise = torch.randn((n_sample, *img_shape), device=device) * self.noise_std
        background = colours.view(-1, 1, 1, 1) + noise

        # if we are sampling, we fix the background for the whole process
        if self.sampling_mode:
            self.background = background

        return background
