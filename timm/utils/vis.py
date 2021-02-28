import torch
import torch.nn as nn
import visdom
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Visualizer():
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = 1

    def plot_curves(self, d, iters, title='loss', xlabel='iters', ylabel='accuracy'):
        name = list(d.keys())
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        self.vis.line(Y=y,
                      X=np.array([self.index]),
                      win=title,
                      opts=dict(legend=name, title = title, xlabel=xlabel, ylabel=ylabel),
                      update=None if self.index == 0 else 'append')
        self.index = iters
