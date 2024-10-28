import math
import typing as tp
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, reduce, repeat


class MaskedConv1d(nn.Conv1d):
    """Wrapper around Conv1d to provide masking functionality
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, audio_mask=None):
        if audio_mask is not None:
            assert audio_mask.shape[-1] == x.shape[-1]
            conv_mask = rearrange(audio_mask, 'b l -> b () l')
            x = torch.where(conv_mask, x, 0)
        x = super().forward(x)
        return x