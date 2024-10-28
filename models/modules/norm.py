import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = F.normalize(x, dim = -1) * self.scale * self.gamma
        return out
    
class ConvRMSNorm(RMSNorm):
    """
    Convolution-friendly RMSNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, dim):
        super().__init__(dim)

    def forward(self, x):
        x = rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = rearrange(x, 'b t ... -> b ... t')
        return x

# use layernorm without bias, more stable

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)