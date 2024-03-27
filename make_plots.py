"""!@file make_plots.py
@brief Script to make plots for the report

@details We make the following plots:
- Training and test loss over epochs for each model
- Samples from the models at epochs 1, 5, 10, 30
- Visualisation of the eraser degradation operator

Example usage:
`python make_plots.py --models ./models --output_dir ./plots`
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import argparse
from os.path import join

import torch
from torchvision import transforms
from torchvision.datasets import MNIST

from src.degradation.eraser_utils import eraser_trajectory, all_pixels_in_trajectory, eraserhead_masks
from src.degradation.eraser import Eraser

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, help="Directory containing the models")
parser.add_argument("--output_dir", type=str, help="Directory to output plots in")
args = parser.parse_known_args()[0]


def plot_loss(metric_logger):
    """!@brief Plot the training and test loss over epochs"""
    fig, ax = plt.subplots(figsize=(7, 4))

    epochs = [i for i in range(1, len(metric_logger["train_loss"]) + 1)]
    ax.plot(epochs, metric_logger["train_loss"], label="train")
    ax.plot(epochs, metric_logger["test_loss"], label="test")
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_xticks(range(1, len(metric_logger["train_loss"]) + 1, 2))
    ax.legend()
    return fig


def save_fig(fig, save_path):
    """!@brief Save the figure to the specified path"""
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Figure saved at {save_path}")


####################
# Training visualisations
####################

# models to analyse
models = ["ddpm_default", "ddpm_high", "magic_uneraser"]

for model_name in models:
    # loss plots
    with open(join(args.models, model_name, "log.pkl"), "rb") as f:
        metric_logger = pickle.load(f)
        fig = plot_loss(metric_logger)
        save_fig(fig, join(args.output_dir, f"{model_name}_loss.png"))

        # print final train and test loss
        print(f"Model: {model_name}")
        print(f"Final train loss: {metric_logger['train_loss'][-1]:.4f}")
        print(f"Final test loss: {metric_logger['test_loss'][-1]:.4f}\n")

    # plotting samples over the training process at epochs 1, 5, 10, 30

    epochs = [1, 5, 10, 25]
    letters = ["(a)", "(b)", "(c)", "(d)"]
    fig, axs = plt.subplots(1, len(epochs), figsize=(10, 5))
    for i, epoch in enumerate(epochs):
        path = join(args.models, model_name, "samples", f"epoch_{epoch - 1:04d}.png")
        img = Image.open(path)
        axs[i].imshow(img)
        axs[i].axis("off")
        axs[i].text(0.02, 1.05, letters[i], fontsize=14, transform=axs[i].transAxes, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, f"plots/{model_name}_training.png")


####################
# Visualisation of how the eraser degradation works
####################

trajectory = eraser_trajectory(3)
pixels = all_pixels_in_trajectory(trajectory)

# define severity of degradation (s = t/T)
s = torch.tensor([0.16])  # severity = t/T

# generate individual eraserheads and full masks
eraserheads = eraserhead_masks(pixels, radius=3, sigma=3, size=28)
masks = torch.exp(torch.cumsum(torch.log(eraserheads), dim=0))

# get the index of the eraserhead and mask corresponding to severity s
index = torch.round(masks.shape[0] * s)
index = index.clamp(0, masks.shape[0] - 1).int()

# create a visualisation of the trajectory
trajectory_img = np.zeros((28, 28))
trajectory_img[pixels[:, 0], pixels[:, 1]] = 0.5
trajectory_img[pixels[:index, 0], pixels[:index, 1]] = 1

# get the eraserhead and mask corresponding to severity s
eraserhead = eraserheads[index]
mask = masks[index]

# get a sample from MNIST
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
dataset = MNIST("./data", train=True, download=True, transform=tf)
x = dataset[22][0].unsqueeze(0)

# create a degrador object and fix the background
degrador = Eraser(trajectory, eraserhead_radius=3, sigma=3, size=28)
degrador.sampling(True)
colour = torch.tensor([0.3])
noise = torch.randn_like(x) * 0.05
degrador.background = colour.view(-1, 1, 1, 1) + noise

# plot the trajectory, eraserhead, mask and degraded image
fig, axs = plt.subplots(1, 4, figsize=(10, 7))
axs = axs.flatten()

axs[0].imshow(trajectory_img, cmap="gray")
axs[1].imshow(eraserhead.squeeze(), cmap="gray")
axs[2].imshow(mask.squeeze(), cmap="gray")
axs[3].imshow(degrador(x, s).squeeze().detach().cpu().numpy(), cmap="gray")

# get rid of x and y ticks
for i, letter in enumerate(["(a)", "(b)", "(c)", "(d)"]):
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].text(0.02, 1.05, letter, fontsize=14, transform=axs[i].transAxes, fontweight="bold")

plt.tight_layout()
save_fig(fig, join(args.output_dir, "eraser_demonstration.png"))
