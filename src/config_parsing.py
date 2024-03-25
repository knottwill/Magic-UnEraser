from configparser import ConfigParser
import torch
import torch.nn as nn
from .noise_schedules import linear_schedule, cosine_schedule
from .models import DDPM, ColdDiffusion
from .cnn import CNN 
from .degradation.eraser import Eraser
from .degradation.eraser_utils import eraser_trajectory

def parse_degradation_config(config):

    diffusion_type = config['fundamental']['diffusion_type']
    img_shape = eval(config.get('fundamental', 'img_shape', fallback="(1, 28, 28)"))
    T = config.getint('hyperparameters', 'T')

    # noise schedule
    if diffusion_type == "ddpm": # only ddpm has noise schedule

        if config['noise_schedule']['type'] == "linear":
            beta1 = config.getfloat('noise_schedule', 'beta1', fallback=1e-4)
            beta2 = config.getfloat('noise_schedule', 'beta2', fallback=0.02)
            noise_schedule = linear_schedule(beta1, beta2, T)
        elif config['noise_schedule']['type'] == "cosine":
            s = config.getfloat('noise_schedule', 'offset', fallback=0.002)
            noise_schedule = cosine_schedule(T, s)
        else:
            raise ValueError("Invalid noise schedule type")
        
        return noise_schedule
        
    elif diffusion_type == "cold":
        if config['degradation']['type'] == 'eraser':
            eraserhead_radius = config.getint('degradation', 'eraserhead_radius', fallback=3)
            sigma = config.getfloat('degradation', 'sigma', fallback=3)
            noise_std = config.getfloat('degradation', 'noise_std', fallback=0.02)

            trajectory = eraser_trajectory(eraserhead_radius)
            degrador = Eraser(trajectory, eraserhead_radius, sigma, noise_std=noise_std, size=img_shape[-1])
        else:
            raise ValueError("Degradation type not supported")
        
        return degrador
    
    else:
        raise ValueError("Invalid diffusion type")
        
def parse_config(config_file):

    config = ConfigParser()
    config.read(config_file)

    # number of degredation steps
    T = config.getint('hyperparameters', 'T')

    # activation function
    activation_func = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
    }[config.get('fundamental', 'activation', fallback="gelu")]

    img_shape = eval(config.get('fundamental', 'img_shape', fallback="(1, 28, 28)"))
    n_hidden = eval(config.get('hyperparameters', 'n_hidden', fallback="(16, 32, 32, 16)"))

    # reconstructor network
    net = CNN(in_channels=img_shape[0], expected_shape=img_shape[1:], n_hidden=n_hidden, act=activation_func)

    # loss function
    criterion = {
        'MSE': nn.MSELoss(),
        'L1': nn.L1Loss()
    }[config.get('fundamental', 'criterion', fallback="MSE")]

    # diffusion type ('ddpm' or 'cold' diffusion)
    diffusion_type = config['fundamental']['diffusion_type']

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
    
    # other important arguments
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
