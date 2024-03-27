"""!@file eraser_utils.py

@brief Contains all utility functions for the Eraser Degredation Operator
"""
from skimage.draw import line

# from skimage.morphology import disk
import numpy as np
import torch


def get_zigzag_trajectory(left, top, bottom, right, size=28):
    """!
    @brief Generates a zigzag trajectory for the eraser head.

    @details The trajectory consists of a series of points on the edges of the image that the eraser moves between
    in straight lines. The points are defined by intersections with `left`, `top`, `bottom`, and `right` edges.
    The eraser moves from the first point in `left` to the first point in `top`, then to the
    second point in `left`, and so on until the last point in `top`. Then it moves to the first point in `bottom`,
    then `right`, and so on until it has completed the full trajectory.
    """
    trajectory = []
    for r, c in zip(left, top):
        trajectory += [(r, 0), (0, c)]
    for r, c in zip(right, bottom):
        trajectory += [(size - 1, c), (r, size - 1)]
    return trajectory


def eraser_trajectory(radius):
    """!
    @brief Generates a zigzag trajectory for the eraser head with a given radius.

    @details The trajectory consists of a series of points on the edge of the image that the eraser moves between.
    We need to ensure that the solid disk of the eraser head will move across every pixel in the image, to ensure
    that the image is fully erased. This involved calculation by hand to determine the correct points for the zigzag pattern,
    so we only allow `radius` to be 2 or 3.
    """
    if radius == 2:
        # Points on the edges of the image that the eraser head moves between
        # (calculated by hand)
        left = np.array([2, 7, 12, 17, 22, 27])
        top = np.array([3, 8, 13, 18, 23, 27])
        bottom = np.array([5, 10, 15, 20, 25])
        right = bottom
    elif radius == 3:
        left = np.array([3, 10, 17, 24])
        top = left + 1
        bottom = top
        right = left + 2
    else:
        raise ValueError("Disk radius not supported")
    return get_zigzag_trajectory(left, top, bottom, right)


def all_pixels_in_trajectory(trajectory: list):
    """!
    @brief Generates all pixels in the trajectory of the eraser head.

    @details The trajectory is a list of pixel coordinates that the eraser head moves between in straight lines.
    This function generates all pixels along that trajectory by using the `line` function from skimage.
    """
    all_pixels = []
    for i in range(len(trajectory) - 1):
        r0, c0 = trajectory[i]
        r1, c1 = trajectory[i + 1]
        rr, cc = line(r0, c0, r1, c1)
        all_pixels.extend(list(zip(rr, cc)))

    # Remove duplicates while preserving order
    final_pixels = []
    [final_pixels.append(x) for x in all_pixels if x not in final_pixels]

    # Return as tensor
    final_pixels = torch.tensor(final_pixels)
    return final_pixels


def disk_mask(pixel, radius, size):
    """!
    @brief Generates a solid disk mask for the center of the eraser head.
    The mask has zeros inside the disk and ones outside.

    @param pixel: A tensor (row, column) representing the center of the disk.
    @param radius: The radius of the disk.
    @param size: The size of the image.

    @return: A tensor of shape (1, size, size) containing the mask.
    """
    # create meshgrid
    rr, cc = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")

    # calculate distance from center
    distance = ((rr - pixel[0]) ** 2 + (cc - pixel[1]) ** 2).float().sqrt()

    # create mask
    mask = (distance > radius).float().unsqueeze(0)
    return mask


def gaussian_mask(pixel, sigma, size):
    """!
    @brief Generates a Gaussian mask centered at a specified pixel location.
    The mask's values increase with distance from the center according to a Gaussian distribution.

    @param pixel: A tensor (row, column) representing the center of the mask.
    @param sigma: The standard deviation of the Gaussian distribution.
    @param size: The size of the image.

    @return: A tensor of shape (1, size, size) containing the Gaussian mask.
    """

    # create meshgrid
    rr, cc = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")

    # calculate gaussian mask
    dY = rr - pixel[0]
    dX = cc - pixel[1]
    exp_part = torch.exp(-0.5 * (dX**2 + dY**2) / (sigma**2))
    gaussian_mask = 1 - exp_part
    return gaussian_mask.unsqueeze(0)


def eraserhead_masks(pixels: torch.Tensor, radius: int = 3, sigma: float = 3, size: int = 28) -> torch.Tensor:
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

    masks = torch.tensor([])
    for pixel in pixels:
        disk = disk_mask(pixel, radius, size)  # solid disk
        gaussian = gaussian_mask(pixel, sigma, size)  # gaussian falloff
        mask = disk * gaussian  # combine the two masks
        masks = torch.cat((masks, mask.unsqueeze(0)))

    return masks
