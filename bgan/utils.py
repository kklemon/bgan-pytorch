import torch
import torch.nn as nn

from pathlib import Path
from datetime import datetime


def update_average(model_tgt, model_src, beta):
    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)


def get_default_device(device=None):
    if device:
        return device
    else:
        return 'cuda' if torch.cuda.is_available() else 'cpu'


def create_result_dir(run_name):
    result_dir = Path(f'results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{run_name}')
    result_dir.mkdir(parents=True)

    samples_dir = result_dir / 'samples'
    samples_dir.mkdir()

    checkpoint_dir = result_dir / 'checkpoints'
    checkpoint_dir.mkdir()

    return result_dir, samples_dir, checkpoint_dir


def get_activation_by_name(name):
    if name == 'relu':
        return nn.ReLU
    if name == 'leaky_relu':
        return lambda: nn.LeakyReLU(0.2)
    if name == 'elu':
        return nn.ELU
    raise ValueError(f'Unknown activation function \'{name}\'')


def apply_spectral_norm(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.utils.spectral_norm(m)


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0, 0.02)