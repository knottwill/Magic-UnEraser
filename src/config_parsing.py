from configparser import ConfigParser
import torch
import torch.nn as nn
from .noise_schedules import linear_schedule, cosine_schedule

def load_config(config_file):

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
    }[config.get('main', 'activation', fallback="gelu")]

    # loss function
    criterion = {
        'MSE': nn.MSELoss,
        'L1': nn.L1Loss
    }[config.get('main', 'criterion', fallback="MSE")]

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

    cfg = {
        'model_name': config['main']['model_name'],
        'img_shape': eval(config.get('main', 'img_shape', fallback="(1, 28, 28)")),
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
    
def get_config_string(config_file):
    with open(config_file, 'r') as f:
        return f.read()