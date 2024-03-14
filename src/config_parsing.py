from configparser import ConfigParser
import torch
import torch.nn as nn
from .noise_schedules import linear_schedule, cosine_schedule
from .models import DDPM 
from .cnn import CNN 

def load_config(config_file, project_root='.'):

    config = ConfigParser()
    config.read(config_file)

    T = config.getint('hyperparameters', 'T')

    # activation function
    activation_func = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
    }[config.get('fundamental', 'activation', fallback="gelu")]

    # loss function
    criterion = {
        'MSE': nn.MSELoss(),
        'L1': nn.L1Loss()
    }[config.get('fundamental', 'criterion', fallback="MSE")]

    # noise schedule
    if config['noise_schedule']['type'] == "linear":
        beta1 = config.getfloat('noise_schedule', 'beta1', fallback=1e-4)
        beta2 = config.getfloat('noise_schedule', 'beta2', fallback=0.02)
        noise_schedule = linear_schedule(beta1, beta2, T)
    elif config['noise_schedule']['type'] == "cosine":
        s = config.getfloat('noise_schedule', 'offset', fallback=0.002)
        noise_schedule = cosine_schedule(T, s)
    else:
        raise ValueError("Invalid noise schedule type")

    # optimizer (Adam or SGD)
    if config['optimizer']['type'] == "adam":
        optimizer = torch.optim.Adam
        optim_hparams = {
            'lr': config.getfloat('optimizer', 'lr', fallback=2e-4),
        }
    elif config['optimizer']['type'] == "sgd":
        optimizer = torch.optim.SGD
        optim_hparams = {
            'lr': config.getfloat('optimizer', 'lr', fallback=1e-3),
            'momentum': config.getfloat('optimizer', 'momentum', fallback=0.9),
        }
    else:
        raise ValueError("Invalid optimizer type")

    # model configuration
    cfg = {
        'model_name': config['fundamental']['model_name'],
        'diffusion_type': config['fundamental']['diffusion_type'], # 'ddpm' or 'cold'
        'img_shape': eval(config.get('fundamental', 'img_shape', fallback="(1, 28, 28)")),
        'use_accelerator': config.getboolean('fundamental', 'use_accelerator', fallback=True),
        'T': T,
        'n_epochs': config.getint('hyperparameters', 'n_epochs'),
        'batch_size': config.getint('hyperparameters', 'batch_size', fallback=64),
        'n_hidden': eval(config.get('hyperparameters', 'n_hidden', fallback="(16, 32, 32, 16)")),
        'act': activation_func,
        'criterion': criterion,
        'noise_schedule': noise_schedule,
        'optimizer': optimizer,
        'optim_hparams': optim_hparams
    }

    return cfg
    
def parse_config(config_file):
    """Parse a config file and return a model with the specified hyperparameters."""

    cfg = load_config(config_file)

    net = CNN(in_channels=cfg['img_shape'][0], expected_shape=cfg['img_shape'][1:], n_hidden=cfg["n_hidden"], act=cfg["act"])

    if cfg['diffusion_type'] == "ddpm":
        model = DDPM(net, noise_schedule=cfg['noise_schedule'], criterion=cfg["criterion"])
    
    return model