import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
import math

from models.modules.norm import RMSNorm

def exists(x):
    return x is not None

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.silu(gate) * x

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        time_cond_dim = None,
        dropout = 0.,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        inner_dim = int(dim * mult * 2 / 3)
        dim_out = dim

        self.time_cond = None
        self.dropout = nn.Dropout(dropout)

        if dropout > 0:
            self.net = nn.Sequential(
                nn.Linear(dim, inner_dim*2),
                SwiGLU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out)
            ) 
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, inner_dim*2),
                SwiGLU(),
                nn.Linear(inner_dim, dim_out)
            )

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 3),
                Rearrange('b d -> b 1 d')
            )

            zero_init_(self.time_cond[-2])
        else:
            zero_init_(self.net[-1])


    def forward(self, x, time = None):
        x = self.norm(x)
        if exists(self.time_cond):
            assert exists(time)
            scale, shift, gate = self.time_cond(time).chunk(3, dim = 2)
            x = (x * (scale + 1)) + shift

        x = self.net(x)

        if exists(self.time_cond):
            x = x*gate

        return x
