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

from src.config_parsing import load_config
from src.cnn import CNN
from src.models import DDPM

# load config file
cfg_file = sys.argv[1]
cfg = load_config(cfg_file)

# create cnn, diffusion model and optimizer
net = CNN(in_channels=cfg['img_shape'][0], expected_shape=cfg['img_shape'][1:], n_hidden=cfg["n_hidden"], act=cfg["act"])
ddpm = DDPM(net, noise_schedule=cfg['noise_schedule'], criterion=cfg["criterion"])
optim = cfg['optimizer'](ddpm.parameters(), **cfg['optim_hparams'])

# load data
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
dataset = MNIST("./data", train=True, download=True, transform=tf)
dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)

# create directory to save model state dict, samples, logs and config file
results_dir = os.path.join("./models", cfg['model_name'])
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "samples"), exist_ok=True)

# save config file in results directory for safety
with open(os.path.join(results_dir, "config.ini"), "w") as f:
    f.write(open(cfg_file, 'r').read())

# use accelerator if specified in config file
if cfg['use_accelerator']:
    accelerator = Accelerator()
    ddpm, optim, dataloader = accelerator.prepare(ddpm, optim, dataloader)
    device = accelerator.device
else:
    device = torch.device('cpu')

# train model
metric_logger = {}
losses = []
for i in range(cfg['n_epochs']):
    ddpm.train()

    pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar
    for x, _ in pbar:
        optim.zero_grad()

        loss = ddpm(x)

        loss.backward()
        # ^Technically should be `accelerator.backward(loss)` but not necessary for local training

        losses.append(loss.item())
        avg_loss = np.average(losses[min(len(losses)-100, 0):]) # average of last 100 losses
        pbar.set_description(f"{i} loss: {avg_loss:.3g}")  # Show running average of loss in progress bar

        optim.step()

    ddpm.eval()
    with torch.no_grad():
        xh = ddpm.sample(16, (1, 28, 28), device) 
        grid = make_grid(xh, nrow=4)

        # save samples
        save_image(grid, os.path.join(results_dir, 'samples', f"epoch_{i:04d}.png"))

        # save model
        torch.save(ddpm.state_dict(), os.path.join(results_dir, f"{cfg['model_name']}.pth"))

        # save losses
        metric_logger['loss'] = losses
        with open(os.path.join(results_dir, "log.pkl"), "wb") as f:
            pickle.dump(metric_logger, f)

