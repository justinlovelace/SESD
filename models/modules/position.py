import torch
import torch.nn as nn
import math
from einops import rearrange, einsum, repeat

from models.modules.blocks import FeedForward

def exists(val):
    return val is not None

class VariationalFourierFeatures(nn.Module):
    """ following https://arxiv.org/abs/2107.00630 """

    def __init__(self, n_min=0, n_max=6, step=1):
        super().__init__()
        assert n_min <= n_max
        self.n_min = n_min
        self.n_max = n_max
        self.step = step

    def forward(self, x):
        # Create Base 2 Fourier features
        w = 2.**torch.arange(self.n_min, self.n_max+1, self.step, device = x.device, dtype = x.dtype) * 2 * math.pi

        if len(x.shape) == 3:
            w = repeat(w, 'f -> b l f', b = x.shape[0], l = x.shape[1])    
            freqs = einsum(x, w, 'b l d, b l f -> b l d f')
            freqs = rearrange(freqs, 'b l d f -> b l (d f)')
            fouriered = torch.cat([x, freqs.sin(), freqs.cos()], dim=-1)
        elif len(x.shape) == 1:
            w = repeat(w, 'f -> l f', l = x.shape[0])
            freqs = einsum(x, w, 'l, l f -> l f')
            x = rearrange(x, 'l -> l ()')
            fouriered = torch.cat([x, freqs.sin(), freqs.cos()], dim=-1)
        return fouriered
    
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, dim, n_min=0, n_max=6, n_layers=3):
        super().__init__()
        self.variational_fourier_features = VariationalFourierFeatures(n_min=n_min, n_max=n_max)

        self.init_proj = nn.Linear((1+(n_max-n_min))*2+1, dim)

        self.layers = nn.ModuleList([FeedForward(dim) for _ in range(n_layers)])

    def forward(self, x):
        fourier_emb = self.init_proj(self.variational_fourier_features(x))
        for layer in self.layers:
            fourier_emb = layer(fourier_emb) + fourier_emb
        
        return fourier_emb
    
class RelativePositionalEmbedding(nn.Module):
    def __init__(self, dim, n_min=0, n_max=6, n_layers=3):
        super().__init__()
        self.variational_fourier_features = VariationalFourierFeatures(n_min=n_min, n_max=n_max)

        self.init_proj = nn.Linear((1+(n_max-n_min))*2+1, dim)

        self.layers = nn.ModuleList([FeedForward(dim) for _ in range(n_layers)])

    def forward(self, x, attention_mask):
        position_indices = torch.arange(x.shape[1], device = x.device)
        # Need to handle masked contexts from classifier free guidance
        context_len = torch.sum(attention_mask, dim=-1)
        # Replace 0s with 1s to avoid divide by 0
        context_len = torch.where(context_len == 0, torch.ones_like(context_len), context_len)
        relative_position = repeat(position_indices, 'l -> b l', b = x.shape[0])/(context_len.unsqueeze(-1))
        relative_position = rearrange(relative_position, 'b l -> b l ()')
        relative_position_emb = self.init_proj(self.variational_fourier_features(relative_position))
        for layer in self.layers:
            relative_position_emb = layer(relative_position_emb) + relative_position_emb
        
        return relative_position_emb

    
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.emb.weight, std=.01)

    def forward(self, x, pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

# From https://github.com/lucidrains/x-transformers/blob/c7cc22268c8ebceef55fe78343197f0af62edf18/x_transformers/x_transformers.py#L272
class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth=3, log_distance = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.init_proj = nn.Linear(1, dim)

        self.layers = nn.ModuleList([FeedForward(dim) for _ in range(depth)])

        self.out = nn.Linear(dim, heads)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        pos_emb = self.init_proj(pos)
        for layer in self.layers:
            pos_emb = layer(pos_emb) + pos_emb
        pos_biases = self.out(pos_emb)

        # get position biases        
        bias = pos_biases[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias
