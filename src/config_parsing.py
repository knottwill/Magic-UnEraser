from configparser import ConfigParser
import torch
import torch.nn as nn
from .noise_schedules import linear_schedule, cosine_schedule
from .models import DDPM 
from .cnn import CNN 

def parse_config(config_file, project_root='.'):

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

    # diffusion model
    if diffusion_type == "ddpm":
        model = DDPM(net, noise_schedule=noise_schedule, criterion=criterion)
    elif diffusion_type == "cold":
        raise NotImplementedError("Cold Diffusion not yet implemented")
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
    
    # model configuration
    cfg = {
        'model_name': config['fundamental']['model_name'],
        'use_accelerator': config.getboolean('fundamental', 'use_accelerator', fallback=True),
        'T': T,
        'n_epochs': config.getint('hyperparameters', 'n_epochs'),
        'batch_size': config.getint('hyperparameters', 'batch_size', fallback=128),
    }

    # return configuration, model and optimizer
    return cfg, model, optimizer
