"""!@file train.py

@brief Script to train diffusion model on MNIST dataset

@details The model is specified in the config file. It can be cold diffusion or DDPM.
"""

import sys 
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from accelerate import Accelerator
from PIL import Image

from src.config_parsing import parse_config

# parse config file
config_file = sys.argv[1]
cfg, model, optim = parse_config(config_file)

# load data
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
trainset = MNIST("./data", train=True, download=True, transform=tf)
testset = MNIST("./data", train=False, download=True, transform=tf)
train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size=cfg['batch_size'], shuffle=False, drop_last=False)

# create directory to save model state dict, samples, logs and config file
results_dir = os.path.join("./models", cfg['model_name'])
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "samples"), exist_ok=True)

# save config file in results directory for safety
with open(os.path.join(results_dir, "config.ini"), "w") as f:
    f.write(open(config_file, 'r').read())

# use accelerator if specified in config file
if cfg['use_accelerator']:
    accelerator = Accelerator()
    model, optim, train_loader, test_loader = accelerator.prepare(model, optim, train_loader, test_loader)
    device = accelerator.device
else:
    device = torch.device('cpu')

# train model
metric_logger = {'train_loss': [], 'test_loss': []}
losses = []
for i in range(cfg['n_epochs']):
    model.train()

    total_loss = 0
    for x, _ in tqdm(train_loader):
        optim.zero_grad()

        loss = model(x)

        loss.backward()
        # ^Technically should be `accelerator.backward(loss)` but not necessary for local training
        optim.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    metric_logger['train_loss'].append(train_loss)
    print(f"Epoch {i+1}/{cfg['n_epochs']}: Train Loss: {train_loss}")

    model.eval()
    with torch.no_grad():

        # obtain average test loss
        total_loss = 0
        for x, _ in tqdm(test_loader):
            loss = model(x)
            total_loss += loss.item()
        test_loss = total_loss / len(test_loader)

        # sample from model
        xh = model.sample(n_sample=16, img_shape=(1, 28, 28), device=device) 
        grid = make_grid(xh, nrow=4)

        # save samples
        save_image(grid, os.path.join(results_dir, 'samples', f"epoch_{i:04d}.png"))

        # save model
        torch.save(model.state_dict(), os.path.join(results_dir, "state_dict.pth"))

    # log and save losses
    metric_logger['test_loss'].append(test_loss)
    print(f"Epoch {i+1}/{cfg['n_epochs']}: Test Loss: {test_loss}")
    with open(os.path.join(results_dir, "log.pkl"), "wb") as f:
        pickle.dump(metric_logger, f)

