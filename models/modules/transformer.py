import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
import math

from models.modules.norm import RMSNorm
from models.modules.position import RelativePositionalEmbedding, DynamicPositionBias
from models.modules.blocks import FeedForward   

def exists(x):
    return x is not None

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)
            
class Attention(nn.Module):
    def __init__(
            self, 
            dim, 
            dim_head = 32,
            time_cond_dim = None,
            dropout=0.
            ):
        super().__init__()
        assert dim % dim_head == 0, 'Dimension must be divisible by the head dimension'
        self.heads = dim // dim_head

        self.dropout = dropout
        self.time_cond = None

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim)

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 3),
                Rearrange('b d -> b 1 d')
            )
            zero_init_(self.time_cond[-2])
        else:
            zero_init_(self.to_out)

    def forward(self, x, attn_bias, time=None, audio_mask=None):
        b, c, n = x.shape

        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift, gate = self.time_cond(time).chunk(3, dim = 2)
            x = (x * (scale + 1)) + shift
        
        qkv = self.to_qkv(x).chunk(3, dim = 2)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads).contiguous(), qkv)

        attn_bias = repeat(attn_bias, 'h i j -> b h i j', b=b)

        if exists(audio_mask):
            mask_value = -torch.finfo(q.dtype).max
            mask = rearrange(audio_mask, 'b l -> b () () l')
            attn_bias = attn_bias.masked_fill(~mask, mask_value)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0., attn_mask=attn_bias)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        if exists(self.time_cond):
            out = out*gate

        return out
    
class CrossAttention(nn.Module):
    def __init__(self, dim, dim_context, dim_head = 32, time_cond_dim=None, dropout=0., position_aware=False, cross_attn_pos_dim=None,):
        super().__init__()
        assert dim % dim_head == 0, 'Dimension must be divisible by the head dimension'
        self.heads = dim // dim_head
        self.dropout = dropout
        self.norm = RMSNorm(dim)
        self.time_cond = None
        self.time_cond = None
        self.position_aware = position_aware

        if self.position_aware:
            assert exists(cross_attn_pos_dim)
            self.pos_to_k = nn.Linear(cross_attn_pos_dim, dim, bias=False)


        self.norm_context = nn.LayerNorm(dim_context)

        self.null_kv = nn.Parameter(torch.randn(2, dim))
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 3),
                Rearrange('b d -> b 1 d')
            )
            zero_init_(self.time_cond[-2])
        else:
            zero_init_(self.to_out)

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

    def forward(self, x, context, context_mask, time=None, context_pos=None,):
        '''
        x: [B, L_audio, d_unet]
        context: [B, L_text, d_lm]
        context_mask: [B, L_text]
        '''
        b, c, n = x.shape
        x = self.norm(x)
        if exists(self.time_cond):
            assert exists(time)
            scale, shift, gate = self.time_cond(time).chunk(3, dim = 2)
            x = (x * (scale + 1)) + shift
        context = self.norm_context(context)
    
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
            
        if self.position_aware:
            assert exists(context_pos)
            k = k + self.pos_to_k(context_pos)
            
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads).contiguous(), (q, k, v))

        # Null value for classifier free guidance
        nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b = b, h=self.heads), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        query_len = q.shape[2]

        # RMSNorm Trick for stability
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Masking pad tokens 
        context_mask = F.pad(context_mask, (1, 0), value = True)
        context_mask = repeat(context_mask, 'b j -> b h q_len j', h=self.heads, q_len=query_len)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=context_mask, dropout_p=self.dropout if self.training else 0.)
        # attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        # attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
        # out = attn_weight @ v

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        if exists(self.time_cond):
            out = out*gate

        return out


class ConditionableTransformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        *,
        num_layers,
        time_cond_dim,
        dim_head = 64,
        ff_mult = 4,
        dropout=0.0,
        position_aware_cross_attention=False,
        num_registers=8,
        dense_connections=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.position_aware = position_aware_cross_attention
        if self.position_aware:
            cross_attn_pos_dim = dim//4
            self.context_pos_emb = RelativePositionalEmbedding(cross_attn_pos_dim, n_min=0, n_max=6)
        else:
            cross_attn_pos_dim = None

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, dropout=dropout),
                CrossAttention(dim = dim, dim_head = dim_head, dim_context=dim_context, dropout=dropout, cross_attn_pos_dim=cross_attn_pos_dim, position_aware=position_aware_cross_attention),
                FeedForward(dim=dim, mult=ff_mult, time_cond_dim=time_cond_dim, dropout=dropout)
            ]))

        self.dynamic_pos_bias = DynamicPositionBias(dim = dim // 4, heads = dim // dim_head, log_distance = False, depth = 2)

        self.has_registers = num_registers > 0
        if num_registers > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_registers, dim))

        if dense_connections:
            assert num_layers % 2 == 0, 'number of layers must be divisible by 2 for dense connections'
            self.dense_blocks = nn.ModuleList([nn.Linear(dim*2, dim) for i in range(num_layers // 2)])
        self.dense_connections = dense_connections

    def forward(
        self,
        x,
        *,
        time,
        context,
        context_mask,
        audio_mask,
    ):

        if self.position_aware:
            context_pos = self.context_pos_emb(context, context_mask)
            
        else:
            context_pos = None

        if self.has_registers:
            mem = repeat(self.memory_tokens, 'l d -> b l d', b = x.shape[0])
            x, mem_packed_shape = pack((mem, x), 'b * d')

            mem_attn_mask = torch.ones_like(mem[:,:,0], dtype=torch.bool)
            audio_mask = torch.cat((mem_attn_mask, audio_mask), dim=1)


        i = j = x.shape[1]
        attn_bias = self.dynamic_pos_bias(i, j)

        hiddens = []
        for idx, (attn, cross_attn, ff) in enumerate(self.layers):
            if self.dense_connections:
                if self.has_registers:
                    if idx < (self.num_layers // 2):
                        # store hidden states for dense connections
                        hiddens.append(x[:, mem.shape[1]:, :]) 
                    else:
                        concat_feats = torch.cat((x[:, mem.shape[1]:, :], hiddens.pop()), dim=-1)
                        x[:, mem.shape[1]:, :] = self.dense_blocks[idx - (self.num_layers // 2)](concat_feats)
                else:
                    if idx < (self.num_layers // 2):
                        # store hidden states for dense connections
                        hiddens.append(x) 
                    else:
                        concat_feats = torch.cat((x, hiddens.pop()), dim=-1)
                        x = self.dense_blocks[idx - (self.num_layers // 2)](concat_feats)
            res = x
            x = attn(x, attn_bias=attn_bias, audio_mask=audio_mask) + res

            res = x
            x = cross_attn(x, context = context,  context_mask=context_mask, context_pos=context_pos) + res

            res = x
            x = ff(x, time=time) + res

        if self.has_registers:
            mem, x = unpack(x, mem_packed_shape, 'b * d')

        return x
