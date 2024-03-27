"""!@file eval.py

@brief Evaluation script for trained diffusion model

@details This script generates samples from the trained model and
computes the FID score between the generated samples and the MNIST test set.
However many samples are generated, the FID score is computed on the same number of samples
from the MNIST test set.

Example usage:
`python eval.py --model_dir ./models/ddpm_default --output_dir ./plots`
"""

import numpy as np
import torch
import argparse
from os.path import join
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
from time import time

from src.config_parsing import parse_config

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

t0 = time()

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="Directory containing model config and state_dict.pth file")
parser.add_argument("--output_dir", type=str, help="Directory to output plots in")
parser.add_argument("--n_samples", default=100, type=int, help="Number of samples to draw from the model")
args = parser.parse_known_args()[0]

device = torch.device("cpu")  # CPU is sufficient for evaluation

# load model
cfg, model, _ = parse_config(join(args.model_dir, "config.ini"))
model.to(device)
state_dict = torch.load(join(args.model_dir, "state_dict.pth"), map_location=device)
model.load_state_dict(state_dict)

# print number of model parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model parameters: {n_params}\n")

# unconditional generation
print(f"Generating {args.n_samples} samples...")
model.eval()
with torch.no_grad():
    x_gen = model.sample(args.n_samples, (1, 28, 28), device)

    # Save samples to `./contents` directory
    grid = make_grid(x_gen, nrow=int(np.sqrt(args.n_samples)))
    save_path = join(args.output_dir, f"{cfg['model_name']}_samples.png")
    save_image(grid, save_path)

print(f"{args.n_samples} Samples saved to {save_path}\n")

############
# FID Score
############

# https://arxiv.org/pdf/1706.08500.pdf

# Load MNIST test set
tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0)),
    ]
)
testset = MNIST("./data", train=False, download=True, transform=tf)
dataloader = DataLoader(testset, batch_size=500, shuffle=True)

# (N, 1, 28, 28) in [-0.5, 0.5] -> (N, 3, 299, 299) in [-1, 1]
transform_for_inceptionv3 = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Resize the images to 299x299
        transforms.Lambda(lambda x: x.mul(255).byte()),  # Convert to 0-255
        transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Repeat 1-channel to 3-channel
    ]
)

# feature = 64, 192, 768 or 2048 (2048 is default, and used in cold diffusion paper)
print("Computing FID score...")
fid = FrechetInceptionDistance(feature=2048)

for real_images, _ in dataloader:
    real_images = transform_for_inceptionv3(real_images)
    fid.update(real_images, real=True)
    break

fid.update(transform_for_inceptionv3(x_gen), real=False)

fid_score = fid.compute()
print(f"FID score: {fid_score}")

print(f"Time taken: {time()-t0: .2f} seconds")
