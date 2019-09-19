from torch import nn
from .fix_modules import register_fix_module

register_fix_module(nn.Conv2d)
register_fix_module(nn.Linear)
register_fix_module(nn.BatchNorm1d)
register_fix_module(nn.BatchNorm2d)
