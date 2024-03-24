import numpy as np
import torch
import argparse
from os.path import join
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from torchmetrics.image.fid import FrechetInceptionDistance

from src.config_parsing import parse_config

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="Directory containing model config and state_dict.pth file")
parser.add_argument("--n_samples", default=100, type=int, help="Number of samples to draw from the model")
args = parser.parse_known_args()[0]

# load model
cfg, model, _ = parse_config(join(args.model_dir, 'config.ini'))
model.load_state_dict(torch.load(join(args.model_dir, 'state_dict.pth')))

# unconditional generation
model.eval()
with torch.no_grad():
    x_gen = model.sample(args.n_samples, (1, 28, 28), torch.device('cpu'))  # Can get device explicitly with `accelerator.device`

    # Save samples to `./contents` directory
    grid = make_grid(x_gen, nrow=int(np.sqrt(args.n_samples)))
    save_image(grid, join(args.model_dir, 'final_samples.png'))

print(f"{args.n_samples} Samples saved to {join(args.model_dir, 'final_samples.png')}\n")

############
# FID Score
############

# https://arxiv.org/pdf/1706.08500.pdf

# Load MNIST test set
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0)),])
testset = MNIST("./data", train=False, download=True, transform=tf)
dataloader = DataLoader(testset, batch_size=args.n_samples, shuffle=True)

# (N, 1, 28, 28) in [-0.5, 0.5] -> (N, 3, 299, 299) in [-1, 1]
transform_for_inceptionv3 = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize the images to 299x299
    transforms.Lambda(lambda x: x.mul(255).byte()), # Convert to 0-255
    transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Repeat 1-channel to 3-channel
])

# feature = 64, 192, 768 or 2048 (2048 is default, and used in cold diffusion paper)
fid = FrechetInceptionDistance(feature=2048)

for real_images, _ in dataloader:
    real_images = transform_for_inceptionv3(real_images)
    fid.update(real_images, real=True)
    break

fid.update(transform_for_inceptionv3(x_gen), real=False)

fid_score = fid.compute()
print(f"FID score: {fid_score}")