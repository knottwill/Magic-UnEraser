# Magic Un-Eraser

<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.2.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.12.2-blue.svg?logo=python&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>

## Description

<details open>
<summary><b>Toggle Degradation/Reconstruction</b></summary>
<br>
<div style="display: flex; justify-content: flex-start; gap: 100px;">
    <div style="text-align: center;">
        <p style="font-size: 18px; font-family: Arial, sans-serif; color: red; margin-bottom: 20px;">Degradation</p>
        <img id="degradationGif" src="plots/degradation.gif" alt="Degradation" style="width: 200px;">
    </div>
    <div style="text-align: center;">
        <p style="font-size: 18px; font-family: Arial, sans-serif; color: green; margin-bottom: 20px;">Reconstruction</p>
        <img id="reconstructionGif" src="plots/reconstruction.gif" alt="Reconstruction" style="width: 200px;">
    </div>
</div>
</details>


## Usage / Re-production

```bash
$ python train.py ./configs/ddpm_default.ini
$ python train.py ./configs/ddpm_cos.ini
$ python train.py ./configs/cold.ini
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
