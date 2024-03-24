"""!
@brief Contains all utility functions for the Eraser Degredation Operator
"""
from skimage.draw import line 
# from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_zigzag_trajectory(left, top, bottom, right, size=28):
    trajectory = []
    for r, c in zip(left, top):
        trajectory += [(r, 0), (0, c)]
    for r, c in zip(right, bottom):
        trajectory += [(size - 1, c), (r, size - 1)]
    return trajectory

def eraser_trajectory(radius, size=28):
    increment = 2*radius+1
    if radius == 2:
        left = np.arange(2, size, increment)
        top = np.clip(left + 1, 0, size-1)
        bottom = np.arange(increment, size, increment)
        right = bottom
    elif radius == 3:
        left = np.arange(3, size, increment)
        top = left + 1
        bottom = top 
        right = left + 2
    else:
        raise ValueError("Disk radius not supported")
    return get_zigzag_trajectory(left, top, bottom, right, size)

def all_pixels_in_trajectory(trajectory: list):
    all_pixels = []
    for i in range(len(trajectory) - 1):
        r0, c0 = trajectory[i]
        r1, c1 = trajectory[i + 1]
        rr, cc = line(r0, c0, r1, c1)
        all_pixels.extend(list(zip(rr, cc)))
    # Remove duplicates while preserving order
    final_pixels = []
    [final_pixels.append(x) for x in all_pixels if x not in final_pixels]
    final_pixels = torch.tensor(final_pixels)
    return final_pixels


def disk_masks(pixels: torch.Tensor, radius, size=28):
    """!
    @brief Generates a solid disk mask for the center of the eraser head.
    
    @details For each pixel in `pixels`, a solid disk of radius `radius` is generated.
    """

    N = pixels.shape[0]
    rr, cc = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    
    # Expand rr and cc to shape (N, 28, 28) for broadcasting with pixels
    rr_expanded = rr.unsqueeze(0).expand(N, size, size)
    cc_expanded = cc.unsqueeze(0).expand(N, size, size)
    
    # Calculate squared distance from each center to each point in the grid
    circle = (rr_expanded - pixels[:, 0].unsqueeze(1).unsqueeze(2))**2 + \
             (cc_expanded - pixels[:, 1].unsqueeze(1).unsqueeze(2))**2
    
    mask = (circle > radius**2).float()
    return mask.unsqueeze(1)

def eraserhead_masks(pixels: torch.Tensor, radius: int = 2, sigma: float = 1.0, size: int = 28) -> torch.Tensor:
    """!
    @brief Generates a mask for the head of the eraser. (for each pixel in `pixels`)

    @details The mask consists of a solid disk of radius `radius`, with a Gaussian falloff around the edge.
    `pixels` has shape (N, 2), where N is the number of masks to generate.
    For each pixel in `pixels`, a separate mask is generated, centered on that pixel.

    @param pixels: A tensor of shape (N, 2) containing the coordinates of the center pixels.
    @param radius: The radius of the solid disk mask.
    @param sigma: The standard deviation of the Gaussian falloff.
    @param size: The size of the image.

    @return A tensor of shape (N, 1, size, size) containing the masks.
    """
    r = torch.arange(size).float().view(1, 1, -1, 1)  # Add two new dimensions
    c = torch.arange(size).float().view(1, 1, 1, -1)  # Add two new dimensions

    r = r - pixels[:, 0].view(-1, 1, 1, 1)
    c = c - pixels[:, 1].view(-1, 1, 1, 1)

    # Calculate the Gaussian falloff
    masks = torch.exp(-0.5 * (r**2 + c**2) / sigma**2)
    masks = 1 - masks

    # extra disk mask to set surrounding pixels to 0 also 
    masks = masks * disk_masks(pixels, radius, size)

    return masks