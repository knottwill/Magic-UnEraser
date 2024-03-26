"""!@file config_parsing.py

@brief Functions for parsing the model configuration files.
"""

from configparser import ConfigParser
import torch
import torch.nn as nn
from .noise_schedules import linear_schedule, cosine_schedule
from .models import DDPM, ColdDiffusion
from .cnn import CNN 
from .degradation.eraser import Eraser
from .degradation.eraser_utils import eraser_trajectory

def parse_degradation_config(config):
    """!
    @brief Parse the degradation strategy from the config file.

    @details If the diffusion type is 'ddpm', the noise schedule is parsed.
    If the diffusion type is 'cold', the degradation operator is parsed.

    @param config ConfigParser object containing the configuration file.

    @return Degradation operator or noise schedule.
    """

    diffusion_type = config['fundamental']['diffusion_type'] 
    img_shape = eval(config.get('fundamental', 'img_shape', fallback="(1, 28, 28)"))
    T = config.getint('hyperparameters', 'T') # number of diffusion steps

    # noise schedule
    if diffusion_type == "ddpm": # only ddpm has noise schedule

        # noise schedule can be linear or cosine (only linear is used in report)
        if config['noise_schedule']['type'] == "linear":
            beta1 = config.getfloat('noise_schedule', 'beta1', fallback=1e-4)
            beta2 = config.getfloat('noise_schedule', 'beta2', fallback=0.02)
            noise_schedule = linear_schedule(beta1, beta2, T)
        elif config['noise_schedule']['type'] == "cosine":
            s = config.getfloat('noise_schedule', 'offset', fallback=0.002)
            noise_schedule = cosine_schedule(T, s)
        else:
            raise ValueError("Invalid noise schedule type")
        
        return noise_schedule # noise schedule
        
    # degradation operator
    elif diffusion_type == "cold":
        if config['degradation']['type'] == 'eraser': # only eraser degradation is supported

            # eraserhead_radius: radius of the central disk of the eraserhead
            eraserhead_radius = config.getint('degradation', 'eraserhead_radius', fallback=3)

            # sigma: standard deviation of the Gaussian mask used to blur the eraserhead
            sigma = config.getfloat('degradation', 'sigma', fallback=3)

            # noise_std: standard deviation of the noise left behind by the eraser
            noise_std = config.getfloat('degradation', 'noise_std', fallback=0.02)

            # generate eraser trajectory
            trajectory = eraser_trajectory(eraserhead_radius)

            # create degradation operator
            degrador = Eraser(trajectory, eraserhead_radius, sigma, noise_std=noise_std, size=img_shape[-1])
        else:
            raise ValueError("Degradation type not supported")
        
        return degrador # degradation operator
    
    else:
        raise ValueError("Invalid diffusion type")
        
def parse_config(config_file):
    """!
    @brief Parse the model configuration file.

    @details Parse the configuration file to create the model and optimizer, as well as 
    dictionary containing other important configuration parameters.

    @param config_file Path to the configuration file.

    @return Configuration dictionary, model and optimizer.
    """

    config = ConfigParser()
    config.read(config_file)

    T = config.getint('hyperparameters', 'T') # number of degradation steps
    img_shape = eval(config.get('fundamental', 'img_shape', fallback="(1, 28, 28)")) # image shape
    diffusion_type = config['fundamental']['diffusion_type'] # diffusion type

    # activation function
    activation_func = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
    }[config.get('fundamental', 'activation', fallback="gelu")]

    # channels in the hidden layers of the CNN
    n_hidden = eval(config.get('hyperparameters', 'n_hidden', fallback="(16, 32, 32, 16)"))

    # restoration network
    net = CNN(in_channels=img_shape[0], expected_shape=img_shape[1:], n_hidden=n_hidden, act=activation_func)

    # loss function
    criterion = {
        'MSE': nn.MSELoss(),
        'L1': nn.L1Loss()
    }[config.get('fundamental', 'criterion', fallback="MSE")]

    # degradation operator (cold diffusion) or noise schedule (ddpm)
    degradation = parse_degradation_config(config)

    # diffusion model
    if diffusion_type == "ddpm":
        model = DDPM(net, noise_schedule=degradation, criterion=criterion)
    elif diffusion_type == "cold":
        model = ColdDiffusion(reconstructor=net, degrador=degradation, criterion=criterion, T=T)
    else:
        raise ValueError("Invalid diffusion type")
    
    # optimizer (Adam or SGD)
    if config['optimizer']['type'] == "adam":
        lr = config.getfloat('optimizer', 'lr', fallback=2e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config['optimizer']['type'] == "sgd":
        lr = config.getfloat('optimizer', 'lr', fallback=1e-3)
        momentum = config.getfloat('optimizer', 'momentum', fallback=0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError("Invalid optimizer type")
    
    # other important configuration parameters
    cfg = {
        'model_name': config['fundamental']['model_name'],
        'use_accelerator': config.getboolean('fundamental', 'use_accelerator', fallback=True),
        'T': T,
        'n_epochs': config.getint('hyperparameters', 'n_epochs'),
        'batch_size': config.getint('hyperparameters', 'batch_size', fallback=128),
        'degradation': degradation,
    }

    # return configuration, model and optimizer
    return cfg, model, optimizer
