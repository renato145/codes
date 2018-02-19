import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import init
from torch.utils.data.dataset import random_split

def V(x, cuda=True, **kwargs):
    v = torch.autograd.Variable(x, **kwargs)
    return v.cuda() if cuda else v

def split_dataset(ds, splits=[0.2]):
    n = ds.data_tensor.size(0)
    splits = [int(n*s) for s in splits]
    splits = [n-sum(splits)] + splits
    return random_split(ds, splits)

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None: init.constant(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None: init.constant(m.bias, 0)

def get_img(x):
    dims = len(x.size())
    x = x.cpu().data.numpy() if isinstance(x, torch.autograd.Variable) else x.cpu().numpy()
    if dims == 2:
        return x
    elif dims == 3:
        if x.shape[0] == 3:
            return np.rollaxis(x, 0, 3)
        elif x.shape[-1] == 3:
            return x

    raise Exception('Uh oh check dims.')
