# Magic Un-Eraser

<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.2.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.12.2-blue.svg?logo=python&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>

## Description

In this project we train a diffusion model to reverse the custom 'Eraser' image degradation process.

<details open>
<summary><b>Toggle Degradation/Reconstruction</b></summary>
<br>
<div style="display: flex; justify-content: flex-start; gap: 100px;">
    <div style="text-align: center;">
        <p style="font-size: 18px; font-family: Arial, sans-serif; color: red; margin-bottom: 20px;">Degradation</p>
        <img id="degradationGif" src="plots/degradation.gif" alt="Degradation" style="width: 200px;">
    </div>
    <div style="text-align: center;">
        <p style="font-size: 18px; font-family: Arial, sans-serif; color: green; margin-bottom: 20px;">Generation</p>
        <img id="reconstructionGif" src="plots/reconstruction.gif" alt="Reconstruction" style="width: 200px;">
    </div>
</div>
</details>

Project structure
- `configs/` - Contains model configurations
- 

## Usage / Re-Production

#### 1. Set-up

#### 2. Training

We trained three models: DDPM model with default hyperparameters. DDPM model with twice the model capacity. Cold diffusion model using the 'Eraser' degradation strategy (& otherwise the same hyperparameters as the default DDPM). 

```bash
$ python train.py ./configs/ddpm_default.ini    # default DDPM
$ python train.py ./configs/ddpm_high.ini       # high capacity DDPM
$ python train.py ./configs/magic_uneraser.ini  # Magic Uneraser :)
```

To evaluate the models, we generate 100 samples and calculate the FID score between the samples 

```bash
$ python eval.py --model_dir ./models/ddpm_default --output_dir ./plots
$ python eval.py --model_dir ./models/ddpm_high --output_dir ./plots
$ python eval.py --model_dir ./models/magic_uneraser --output_dir ./plots
```

To make the plots

```bash
$ python make_plots.py --models ./models --output_dir ./plots
```

## Timing

I ran all scripts on my personal laptop. The `train.py` script used the `mps` device, which is essentially the macbook GPU. The specifications are:
- Operating System: macOS Sonoma v14.0

CPU:
- Chip:	Apple M1 Pro
- Total Number of Cores: 8 (6 performance and 2 efficiency)
- Memory (RAM): 16 GB

GPU (`mps`):
- Chipset Model: Apple M1 Pro
- Type: GPU
- Bus: Built-In
- Total Number of Cores: 14
- Metal Support: Metal 3
