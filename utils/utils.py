import torch
from datetime import datetime
import os
from pathlib import Path
import argparse

def compute_grad_norm(parameters):
    # implementation adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2).item()
    return total_norm

def get_output_dir(args):
    model_dir = f'{args.dataset_name}/{args.run_name}/'
    output_dir = os.path.join(args.save_dir, model_dir)
    return output_dir

def parse_float_tuple(dim_mults_str):
    try:
        dim_mults = tuple(map(float, dim_mults_str.split(',')))
        return dim_mults
    except ValueError:
        raise argparse.ArgumentTypeError('dim_mults must be a comma-separated list of integers')
