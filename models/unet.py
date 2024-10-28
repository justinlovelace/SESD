import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from models.modules import RMSNorm, ConvRMSNorm, ConditionableTransformer, LayerNorm, MaskedConv1d

ENCODEC_DIM = 128

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        MaskedConv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return MaskedConv1d(dim, default(dim_out, dim), 4, 2, 1)
    

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

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, input_connection=False, input_dim=None):
        super().__init__()
        self.input_connection = input_connection
        if input_connection:
            assert exists(input_dim)
            self.proj = MaskedConv1d(dim + input_dim, dim_out, 3, padding = 1)
        else:
            self.proj = MaskedConv1d(dim, dim_out, 3, padding = 1)
        self.norm = ConvRMSNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None, audio_mask=None, input_data=None):
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        if self.input_connection:
            assert exists(input_data)
            x = torch.cat((x, input_data), dim=1)

        x = self.proj(x, audio_mask)
        
        return x
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, input_connection=False, input_dim=None, time_emb_dim = None, groups = 32):
        super().__init__()
        if input_connection:
            assert exists(input_dim)
        self.input_connection = input_connection
        self.mlp = None
        if exists(time_emb_dim):
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 3),
                Rearrange('b c -> b c 1')
            )
            zero_init_(self.mlp[-2])
        
        if input_connection:
            self.block1 = Block(dim, dim_out, groups = groups, input_connection=True, input_dim=input_dim)
        else:
            self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, audio_mask=None, input_data=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb)
            scale_shift = self.mlp(time_emb)
            scale, shift, gate = scale_shift.chunk(3, dim = 1)
            scale_shift = (scale, shift)


        h = self.block1(x, audio_mask=audio_mask, input_data=input_data)

        h = self.block2(h, scale_shift=scale_shift, audio_mask=audio_mask)

        if exists(self.mlp):
            h = h*gate
            
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        zero_init_(self.to_out)

    def forward(self, x, audio_mask=None):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        if exists(audio_mask):
            mask_value = -torch.finfo(q.dtype).max
            mask = audio_mask[:, None, None, :]
            k = k.masked_fill(~mask, mask_value)
            v = v.masked_fill(~mask, 0.)
            del mask

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)
    
def l2norm(t):
    return F.normalize(t, dim = -1)


def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ConvRMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
    
def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

# model

class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        text_dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 128,
        position_aware_cross_attention=False,
        inpainting_embedding = False,
        resnet_block_groups = 32,
        scale_skip_connection=False,
        num_transformer_layers = 3,
        dropout=0.0,
        num_transformer_registers=8,
        input_connection=False,
    ):
        super().__init__()

    
        self.channels = channels
        
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 1)

        dims = [init_dim, *map(lambda m: int(dim * m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups, input_connection=input_connection, input_dim=channels)

        if inpainting_embedding:
            self.inpainting_embedding = nn.Embedding(2, init_dim)
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

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.input_connection = input_connection
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else MaskedConv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.transformer = ConditionableTransformer(mid_dim, dim_context=text_dim, num_layers=num_transformer_layers, time_cond_dim=time_dim, dropout=dropout, position_aware_cross_attention=position_aware_cross_attention, num_registers=num_transformer_registers)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else MaskedConv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.out_dim = channels

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        
        self.final_conv = nn.Sequential(
            ConvRMSNorm(dim),
            nn.SiLU(),
            nn.Conv1d(dim, self.out_dim, 1)
        )
        zero_init_(self.final_conv[-1])

        # Accelerates convergence for image diffusion models
        # Use it by default, but haven't ablated 
        self.scale_skip_connection = (2 ** -0.5) if scale_skip_connection else 1
        
        self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )

    def forward(self, x, time_cond, text_cond=None, text_cond_mask=None, inpainting_mask=None, audio_mask=None):
        assert x.shape[-1] % (2**(len(self.downs)-1)) == 0, f'Length of the audio latent must be a factor of {2**(len(self.downs)-1)}'
        if not exists(audio_mask):
            audio_mask = torch.ones((x.shape[0], x.shape[2]), dtype=torch.bool, device=x.device)
        assert torch.remainder(audio_mask.sum(dim=1), (2**(len(self.downs)-1))).sum().item()==0, f'Length of audio mask must be a factor of {2**(len(self.downs)-1)}'
        
        if self.input_connection:
            input_data = x.clone()
            # Zero out masked values
            input_data = input_data.masked_fill(~audio_mask[:, None, :], 0.)
        else:
            input_data = None
        x = self.init_conv(x)
        if exists(self.inpainting_embedding):
            assert exists(inpainting_mask)
            inpainting_emb = self.inpainting_embedding(inpainting_mask)
            x = x + rearrange(inpainting_emb, 'b l c -> b c l')

        r = x.clone()

        mean_pooled_context = masked_mean(text_cond, dim=1, mask=text_cond_mask)
        text_mean_cond = self.to_text_non_attn_cond(mean_pooled_context)

        # Rescale continuous time [0,1] to similar range as Ho et al. 2020
        t = self.time_mlp(time_cond*1000) 

        t = t + text_mean_cond

        h = []
        audio_mask_list = [audio_mask]
        input_data_list = [input_data]
        for idx, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t, audio_mask=audio_mask_list[-1], input_data=input_data_list[-1])
            h.append(x)

            x = block2(x, t, audio_mask=audio_mask_list[-1], input_data=input_data_list[-1])
            x = attn(x, audio_mask=audio_mask_list[-1])
            h.append(x)


            x_prev_shape = x.shape
            x = downsample(x, audio_mask_list[-1])
            if x.shape[-1] != x_prev_shape[-1]:
                downsampled_mask = reduce(audio_mask_list[-1], 'b (l 2) -> b l', reduction='max')
                audio_mask_list.append(downsampled_mask)
                if self.input_connection:
                    # Einops to mean pool input data
                    input_data_list.append(reduce(input_data_list[-1], 'b c (l 2) -> b c l', reduction='mean'))
        x = rearrange(x, 'b c l -> b l c')

        x = self.transformer(x, context=text_cond, context_mask=text_cond_mask, time=t, audio_mask=audio_mask_list[-1])
        x = rearrange(x, 'b l c -> b c l')

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()*(self.scale_skip_connection)), dim = 1)
            x = block1(x, t, audio_mask_list[-1], input_data_list[-1])

            x = torch.cat((x, h.pop()*(self.scale_skip_connection)), dim = 1)
            x = block2(x, t, audio_mask_list[-1], input_data_list[-1])
            x = attn(x, audio_mask_list[-1])

            # Awkward implementation to maintain backwards compatibility with previous checkpoints
            if isinstance(upsample, nn.Sequential):
                # Need to cast to float32 for upsampling
                # Upsample operation
                x = upsample[0](x.float())
                audio_mask_list.pop()
                # Masked conv operation
                x = upsample[1](x, audio_mask_list[-1])

                if self.input_connection:
                    input_data_list.pop()
            else:
                x = upsample(x, audio_mask_list[-1])
            
        x = torch.cat((x, r), dim = 1)
        
        x = self.final_res_block(x, t, audio_mask, input_data)
        return self.final_conv(x)