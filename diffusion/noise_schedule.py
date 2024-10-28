import torch
import math
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import json

# Avoid log(0)
def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def beta_linear_schedule(t, clip_min = 1e-9):
    return torch.exp(-1e-4 - 10 * (t ** 2)).clamp(min = clip_min, max = 1.)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

def sigmoid_k_weighting(k, gamma):
    return torch.sigmoid(-gamma + k)
    
class V_Weighting:
    def __init__(self, objective='pred_v'):
        self.objective = objective
    
    def v_loss_weighting(self, gamma):
        return 1/torch.cosh(-(gamma/2))

    def eps_loss_weighting(self, gamma):
        return torch.exp(-gamma/(2))
        
class LogNormal_V_Weighting:
    def __init__(self, gamma_mean=0.0, gamma_std=0.0, objective='pred_v'):
        self.gamma_mean = gamma_mean
        self.gamma_std = gamma_std
        self.min_gamma = -15
        self.max_gamma = 15

        assert objective == 'pred_v'

        self.normal_dist = torch.distributions.normal.Normal(self.gamma_mean, self.gamma_std)

        self.log_max_weighting = self.normal_dist.log_prob(torch.tensor(gamma_mean))
    
    def v_loss_weighting(self, gamma):
        return torch.exp(self.normal_dist.log_prob(gamma) - self.log_max_weighting)
    
    def v_weighting(self, gamma):
        return self.v_loss_weighting(gamma) 

class LogCauchy_V_Weighting:
    def __init__(self, gamma_mean=0.0, gamma_std=0.0, objective='pred_v'):
        self.gamma_mean = gamma_mean
        self.gamma_std = gamma_std
        self.min_gamma = -15
        self.max_gamma = 15

        assert objective == 'pred_v'

        self.cauchy_dist = torch.distributions.cauchy.Cauchy(self.gamma_mean, self.gamma_std)

        self.log_max_weighting = self.cauchy_dist.log_prob(torch.tensor(gamma_mean))
    
    def v_loss_weighting(self, gamma):
        return torch.exp(self.cauchy_dist.log_prob(gamma) - self.log_max_weighting)
    
    def v_weighting(self, gamma):
        return self.v_loss_weighting(gamma) 
    
class Asymmetric_LogNormal_V_Weighting:
    def __init__(self, gamma_mean=0.0, gamma_std=0.0, std_mult=2.0, objective='pred_v'):
        self.gamma_mean = gamma_mean
        self.gamma_std = gamma_std
        self.min_gamma = -15
        self.max_gamma = 15

        assert objective == 'pred_v'

        self.neg_normal_v_weighting = LogCauchy_V_Weighting(gamma_mean, gamma_std*std_mult, objective='pred_v')
        self.normal_v_weighting = LogNormal_V_Weighting(gamma_mean, gamma_std, objective='pred_v')
    
    def v_loss_weighting(self, gamma):
        # Use normal weighting for gamma >= self.gamma_mean
        normal_weighting = self.normal_v_weighting.v_loss_weighting(gamma)
        # Use neg_normal weighting for gamma < self.gamma_mean
        neg_normal_weighting = self.neg_normal_v_weighting.v_loss_weighting(gamma)
        return torch.where(gamma < self.gamma_mean, neg_normal_weighting, normal_weighting)

    def v_weighting(self, gamma):
        # Use cauchy weighting for gamma < self.gamma_mean
        neg_normal_weighting = self.neg_normal_v_weighting.v_weighting(gamma)
        # Use normal weighting for gamma >= self.gamma_mean
        normal_weighting = self.normal_v_weighting.v_weighting(gamma)
        return torch.where(gamma < self.gamma_mean, neg_normal_weighting, normal_weighting)

# converting gamma to alpha, sigma or logsnr
def log_snr_to_alpha(log_snr):
    alpha = torch.sigmoid(log_snr)
    return alpha

# Log-SNR shifting (https://arxiv.org/abs/2301.10972)
def alpha_to_shifted_log_snr(alpha, scale = 1):
    return (log(alpha) - log(1 - alpha)).clamp(min=-20, max=20) + 2*np.log(scale).item()

def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale = scale)
    return log_snr_to_alpha(shifted_log_snr)

def plot_noise_schedule(unscaled_sampling_schedule, name, y_value):
    assert y_value in {'alpha^2', 'alpha', 'log(SNR)'}
    t = torch.linspace(0, 1, 100)  # 100 points between 0 and 1
    scales = [.2, .5, 1.0]
    for scale in scales:
        sampling_schedule = partial(time_to_alpha, alpha_schedule=unscaled_sampling_schedule, scale=scale)
        alphas = sampling_schedule(t)  # Obtain noise schedule values for each t
        if y_value == 'alpha^2':
            y_axis_label = r'$\alpha^2_t$'
            y = alphas
        elif y_value == 'alpha':
            y_axis_label = r'$\alpha_t$'
            y = alphas.sqrt()
        elif y_value == 'log(SNR)':
            y_axis_label = r'$\log(\lambda_t)$'
            y = alpha_to_shifted_log_snr(alphas, scale=1)

        plt.plot(t.numpy(), y.numpy(), label=f'Scale: {scale:.1f}')
    if y_value == 'log(SNR)':
        plt.ylim(-15, 15)
    plt.xlabel('t')
    plt.ylabel(y_axis_label)
    plt.title(f'{name}')
    plt.legend()
    plt.savefig(f'viz/{name.lower()}_{y_value}.png')
    plt.clf()


def plot_cosine_schedule():
    t = torch.linspace(0, 1, 100)  # 100 points between 0 and 1
    sampling_schedule = cosine_schedule
    alphas = sampling_schedule(t)  # Obtain noise schedule values for each t
    y = alphas
    plt.plot(t.numpy(), y.numpy())
    plt.xlabel('t')
    plt.ylabel(f'alpha^2')
    plt.title(f'Cosine Noise Schedule')
    plt.savefig(f'viz/standard_cosine.png')
    plt.clf()

def plot_weighting_functions():
    gamma = torch.linspace(-15, 15, 1000)


    # Plot v weighting
    v_obj = V_Weighting(gamma_shift=0.0, objective='pred_eps')
    v_weighting = v_obj.v_loss_weighting(gamma)

    # Log-normal V weighting with mean=0.0, std=2.0
    v_obj = LogNormal_V_Weighting(gamma_mean=-1.0, gamma_std=2.4, objective='pred_v')
    v_weighting_lognormal = v_obj.v_loss_weighting(gamma)

    # Log-cauchyNormal V weighting with mean=0.0, std=2.0
    v_obj = Asymmetric_LogNormal_V_Weighting(gamma_mean=-1.0, gamma_std=2.4, objective='pred_v', std_mult=2.0)
    v_weighting_logcauchynormal = v_obj.v_loss_weighting(gamma)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gamma.numpy(), v_weighting_logcauchynormal.numpy(), label='Asymmetric Weighting')
    ax.plot(gamma.numpy(), v_weighting_lognormal.numpy(), label='Symmetric Weighting')
    ax.plot(gamma.numpy(), v_weighting.numpy(), label='V-Weighting (VoiceBox)')
    ax.set_xlabel(r'Log-SNR ($\lambda_t$)', fontsize=14)
    ax.set_ylabel(r'Loss Weight ($w(\lambda_t)$)', fontsize=14)
    ax.set_title('Loss Weighting Across Noise Levels', fontsize=14)
    # Create legend 
    ax.legend(loc='upper right', ncol=1, fontsize=12)
    plt.savefig('viz/v_weighting_pred.png', bbox_inches='tight')
    plt.clf()

    # Save log-scale version
    fig, ax = plt.subplots()
    ax.plot(gamma.numpy(), v_weighting.numpy(), label='V Weighting')
    ax.plot(gamma.numpy(), v_weighting_lognormal.numpy(), label='Log-Normal V Weighting (mean=-1.0, std=2.4)')
    ax.plot(gamma.numpy(), v_weighting_logcauchynormal.numpy(), label='Log-CauchyNormal V Weighting (mean=0.0, std=2.4)')
    ax.set_xlabel(r'$\lambda_t$')
    ax.set_ylabel('Weighting (V-Space)')
    ax.set_title('V Weighting with pred_v Objective')
    ax.legend()
    ax.set_yscale('log')
    plt.savefig('viz/v_weighting_pred_v_log2.png')
    plt.clf()

# Make sure to call this function in your main visualization routine
if __name__ == '__main__':
    plot_weighting_functions()
    plot_cosine_schedule()
