import math
import torch
from torch import nn
from einops import rearrange, reduce
from functools import partial

from models.modules import RMSNorm, ConvRMSNorm, ConditionableTransformer, LayerNorm, MaskedConv1d
# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerWrapper(nn.Module):
    def __init__(
        self,
        dim,
        text_dim,
        channels = 128,
        position_aware_cross_attention=False,
        inpainting_embedding = False,
        num_transformer_layers = 8,
        dropout=0.0,
    ):
        super().__init__()

    
        self.channels = channels
        self.out_dim = channels
        
        input_channels = channels

        self.init_conv = nn.Conv1d(input_channels, dim, 1)


        if inpainting_embedding:
            self.inpainting_embedding = nn.Embedding(2, dim)
        else:
            self.inpainting_embedding = None

        # time embeddings

        time_dim = dim * 2

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.transformer = ConditionableTransformer(dim, dim_context=text_dim, num_layers=num_transformer_layers, time_cond_dim=time_dim, dropout=dropout, position_aware_cross_attention=position_aware_cross_attention)
        
        self.final_conv = nn.Sequential(
            ConvRMSNorm(dim),
            nn.SiLU(),
            nn.Conv1d(dim, self.out_dim, 1)
        )
        zero_init_(self.final_conv[-1])

        
        self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )

    def forward(self, x, time_cond, text_cond=None, text_cond_mask=None, inpainting_mask=None, audio_mask=None):
        if not exists(audio_mask):
            audio_mask = torch.ones((x.shape[0], x.shape[2]), dtype=torch.bool, device=x.device)
        x = self.init_conv(x)
        if exists(self.inpainting_embedding):
            assert exists(inpainting_mask)
            inpainting_emb = self.inpainting_embedding(inpainting_mask)
            x = x + rearrange(inpainting_emb, 'b l c -> b c l')

        mean_pooled_context = masked_mean(text_cond, dim=1, mask=text_cond_mask)
        text_mean_cond = self.to_text_non_attn_cond(mean_pooled_context)

        # Rescale continuous time [0,1] to similar range as Ho et al. 2020
        t = self.time_mlp(time_cond*1000) 

        t = t + text_mean_cond

        x = rearrange(x, 'b c l -> b l c')

        x = self.transformer(x, context=text_cond, context_mask=text_cond_mask, time=t, audio_mask=audio_mask)
        x = rearrange(x, 'b l c -> b c l')

        return self.final_conv(x)