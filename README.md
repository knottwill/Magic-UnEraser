# Magic Un-Eraser

<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.2.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.12.2-blue.svg?logo=python&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>

## Description

In this project we train a cold diffusion model to generate handwritten digits by reversing the custom 'Eraser' image degradation process. As a baseline, we also train two denoising diffusion probabilistic models (DDPM - standard diffusion).

<details open>
<summary><b>Toggle Degradation/Generation</b></summary>
<br>

| Degradation | Generation |
|:-:|:-:|
| ![Degradation](plots/degradation.gif) | ![Generation](plots/reconstruction.gif) |
</details>

<br>
<b>Project structure</b>

- `configs/` - Contains model configurations of the 3 models trained in this project.
- `data/` - Contains the MNIST dataset.
- `docs/` - Contains documentation for the project.
- `models/` - Contains subdirectories for each trained model (automatically generated by the `train.py` script). Each sub-directory contains the model state dictionary, the configuration used to train it, the metric logger, and a subdirectory of samples generated throughout the training process.
- `plots/` - Contains the plots used in the report, and other visualisations.
- `src/` - Contains the source code for the project. (Re-usable components that are used by the scripts in the root directory)
- `eval.py` - Script for evaluating the trained models.
- `make_plots.py` - Script for generating plots used in the report.
- `train.py` - Script for training the models.
- `.gitignore` - Tells git which files to ignore
- `.pre-commit-config.yaml` - Specifies pre-commit hooks to protect the `main` branch
- `Dockerfile` - Dockerfile to generate docker image
- `requirements.txt` - List of packages/versions to re-create the environment for the project.
- `LICENSE` - MIT license.

## Usage / Re-Production

Note: All commands assume they are being run from the root directory of the project.

### 1. Set-up

To re-create the environment used for the project, use the `requirements.txt` file. This can be done with `pip`, `conda` or docker. The docker container will not naturally have access to the `mps` device on Mac laptops (which was used to train the models), thus for best performance it is recommended to use `pip` or `conda` to re-createe the environment.

<b>pip</b>

```bash
$ pip install -r requirements.txt
```

<b>Using conda</b>

```bash
$ conda create --name <env-name> python
$ conda activate <env-name>
$ pip install -r requirements.txt
```

<b>Docker (not recommended)</b>

```bash
$ docker build -t <image-name> .
$ docker run -ti <image-name> bash
```

### 2. Training

We trained three models: DDPM model with default hyperparameters (low-capacity), DDPM model with high-capacity (twice the number of trainable parameters) and a Cold diffusion model (termed the 'Magic Un-Eraser') using the 'Eraser' degradation strategy. The Cold diffusion model had all the same hyperparameters as the default DDPM.

```bash
$ python train.py ./configs/ddpm_default.ini    # default DDPM model
$ python train.py ./configs/ddpm_high.ini       # high capacity DDPM model
$ python train.py ./configs/magic_uneraser.ini  # Cold diffusion model ("Magic Un-Eraser")
```

### 3. Evaluation & Plotting

To evaluate the models, we generate 100 samples and calculate the FID score between the samples and 500 images from the MNIST test set. Use the following commands:

```bash
$ python eval.py --model_dir ./models/ddpm_default --output_dir ./plots
$ python eval.py --model_dir ./models/ddpm_high --output_dir ./plots
$ python eval.py --model_dir ./models/magic_uneraser --output_dir ./plots
```

To make the plots, use

```bash
$ python make_plots.py --models ./models --output_dir ./plots
```

## Timing

Times to run each script:
- `train.py` - Using the `mps` device, it took ~25 minutes to train the high-capacity DDPM model, and ~15 minutes to train the other two models.
- `eval.py` - No more than 200 seconds for each trained model. (Used the `cpu` only)
- `make_plots.py` - No more than 1-2 minutes.

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
