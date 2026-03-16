import sys, os
import pathlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
import random, time
import soundfile as sf


import math
from typing import Optional, Tuple, List
from itertools import chain

import torch
from torch import Tensor
import numpy as np

import torch.nn as nn
import torch.nn.init as init

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

import torch.nn.functional as F

from torch.amp import autocast # guard
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.generators.apex_gan_modules import PchipF0UpsamplerTorch, FusedGeoSaw, Snake


def apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]):
    return tensor * mask if mask is not None else tensor

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def remove_weight_norm_legacy_safe(module):
    if is_parametrized(module, "weight"):
        remove_parametrizations(module, "weight", leave_parametrized=True)
    else:
        remove_weight_norm(module)

def create_ups_convtranspose1d_layer(in_channels, out_channels, kernel_size, stride):
    m = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - stride) // 2)
    return weight_norm(m)

def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation)))



class KaiserDecimator(nn.Module):
    """
    Single-stage decimation:
        Kaiser-windowed sinc lowpass via FFT convolution, followed by integer striding.
        No learnable parameters - purely a deterministic signal processing module.
    """
    _AA_ROLLOFF = 0.95
    _AA_STOPBAND_DB = 120.0

    def __init__(self, downsample_factor: int):
        super().__init__()
        self.factor = downsample_factor

        cutoff = self._AA_ROLLOFF / (2.0 * downsample_factor)
        delta_f = (1.0 - self._AA_ROLLOFF) / (2.0 * downsample_factor)
        beta = 0.1102 * (self._AA_STOPBAND_DB - 8.7)
        n_taps = int(math.ceil((self._AA_STOPBAND_DB - 8.0) / (2.285 * 2.0 * math.pi * delta_f)))

        if n_taps % 2 == 0:
            n_taps += 1

        half_k = (n_taps - 1) // 2
        n_arr = torch.arange(-half_k, half_k + 1, dtype=torch.float64)
        h = torch.sinc(2.0 * cutoff * n_arr)
        win = torch.kaiser_window(n_taps, periodic=False, beta=beta, dtype=torch.float64)
        h = (h * win).float()
        h = h / h.sum()

        self.register_buffer('aa_filter', h.view(1, 1, n_taps))
        self._half_k = half_k
        self._n_taps = n_taps
        self._fft_cache = {}

    def _apply(self, fn):
        self._fft_cache.clear()
        return super()._apply(fn)

    def _get_cached_filter(self, n_fft, device, dtype):
        """Retrieves or computes the FFT of the filter for a given size."""
        key = (n_fft, device, dtype)
        if key not in self._fft_cache:
            w_f32 = self.aa_filter.to(device, dtype=dtype)
            self._fft_cache[key] = torch.fft.rfft(w_f32, n=n_fft)
        return self._fft_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]  -->  [B, C, T // factor]
        with torch.no_grad():
            b, c, t = x.shape
            taps = self._n_taps
            half_k = self._half_k

            N_fft_single = 1 << math.ceil(math.log2(t + taps - 1))

            if N_fft_single <= (1 << 21):
                W = self._get_cached_filter(N_fft_single, x.device, torch.float32)
                y = torch.fft.irfft(torch.fft.rfft(x.float(), n=N_fft_single) * W, n=N_fft_single)
                x = y[..., half_k : half_k + t].to(x.dtype)
            else:
                # Overlap-Add
                N_fft_ola = 1 << 19
                step = N_fft_ola - (taps - 1)
                W = self._get_cached_filter(N_fft_ola, x.device, torch.float32)
                x_fp32 = x.float()
                conv_len = t + taps - 1
                out_full = torch.zeros(b, c, conv_len, device=x.device, dtype=torch.float32)
                pos = 0
                while pos < t:
                    chunk = x_fp32[..., pos : pos + step]
                    if chunk.shape[-1] < N_fft_ola:
                        chunk = F.pad(chunk, (0, N_fft_ola - chunk.shape[-1]))
                    y_block = torch.fft.irfft(torch.fft.rfft(chunk, n=N_fft_ola) * W, n=N_fft_ola)
                    write_end = min(pos + N_fft_ola, conv_len)
                    out_full[..., pos : write_end] += y_block[..., : write_end - pos]
                    pos += step
                x = out_full[..., half_k : half_k + t].to(x.dtype)

            # Stride: integer decimation
            return x[:, :, ::self.factor].contiguous()


class ExcitationPyramid(nn.Module):
    """
    Decomposes the full-rate excitation waveform into a lowpass pyramid.

    Each stage receives the true excitation signal decimated to that stage's
    sample rate — a lowpass representation of the full excitation at the
    resolution that stage can actually process. This gives every generator
    stage a coherent, representative pitch reference rather than a residual band.

    lowpass levels (coarse to fine):
        Stage 0 gets G[0]: excitation decimated to stage-0 rate  (0..stage-0-Nyquist)
        Stage 1 gets G[1]: excitation decimated to stage-1 rate  (0..stage-1-Nyquist)
        Stage 2 gets G[2]: excitation decimated to stage-2 rate  (0..stage-2-Nyquist)
        Stage 3 gets G[3]: full-rate excitation                  (0..full Nyquist)

    Cascade decimation (fine-to-coarse, using upsample_rates[1:][::-1]):
        48k  [12,10,2,2]  -->  cascade [2,2,10]
        40k  [10,10,2,2]  -->  cascade [2,2,10]
        32k  [10, 8,2,2]  -->  cascade [2,2,8]
        24k  [10, 6,2,2]  -->  cascade [2,2,6]

    Right-boundary solution:
        KaiserDecimator FFT-convolves via zero-padding, causing the last
        half_k output samples of each decimated level to taper toward zero.
        We zero-pad the RIGHT side only by _pad_samples before the cascade
        so the taper always falls in the padded zeros. Each level is then
        trimmed back to T_orig-proportional length, discarding the tail.
        No left padding = no temporal offset introduced.

        _pad_samples = sum_i( half_k_i * prod(cascade_factors[0:i]) )
    """
    def __init__(self, upsample_rates: List[int]):
        super().__init__()

        # Cascade decimation factors: reversed(upsample_rates[1:])
        cascade_factors = list(reversed(upsample_rates[1:]))
        self.decimators = nn.ModuleList([KaiserDecimator(f) for f in cascade_factors])

        # Cumulative decimation factors for length trimming (fine-to-coarse):
        #   [1, F_0, F_0*F_1, F_0*F_1*F_2]
        cum = 1
        cum_factors: List[int] = [cum]
        for f in cascade_factors:
            cum *= f
            cum_factors.append(cum)
        self._cum_factors: List[int] = cum_factors

        # Right-boundary padding at full exc rate to absorb Kaiser taper.
        # Each decimator i has half_k = K_i at rate sr/prod(factors[0:i]),
        # which maps to K_i * prod(factors[0:i]) samples at the full exc rate.
        cum_factor = 1
        pad_samples = 0
        for dec, f in zip(self.decimators, cascade_factors):
            pad_samples += dec._half_k * cum_factor
            cum_factor *= f
        self._pad_samples: int = pad_samples

    def forward(self, exc: torch.Tensor) -> List[torch.Tensor]:
        """
        exc: [B, 1, T_audio]
        Returns: list of [B, 1, T_stage_i], length = num_upsamples
            index 0 = coarsest (stage 0), index -1 = finest (full rate)
        """
        T_orig = exc.shape[-1]

        # Zero-pad right only so the Kaiser taper falls in the padded region.
        exc_padded = F.pad(exc, (0, self._pad_samples))

        # Build lowpass pyramid via cascade decimation (fine --> coarse)
        exc_levels: List[torch.Tensor] = [exc_padded]
        for dec in self.decimators:
            exc_levels.append(dec(exc_levels[-1]))

        # Reverse to coarse-first for consistent indexing with generator stages
        exc_levels = list(reversed(exc_levels))

        # Trim each level back to T_orig-proportional length, discarding the padded tail.
        # ceil ensures we never under-deliver; the generator's length guard handles any +1.
        n = len(exc_levels)
        for i in range(n):
            cf = self._cum_factors[n - 1 - i]
            exc_levels[i] = exc_levels[i][..., : math.ceil(T_orig / cf)]

        return exc_levels


class ResBlock(nn.Module):
    """
    A residual block module that applies a series of 1D convolutional layers
    with residual connections.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

        self.snake1 = Snake(channels, init='periodic', correction=None)
        self.snake2 = Snake(channels, init='periodic', correction=None)

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        layers = nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        return layers

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for conv1, conv2 in zip(self.convs1, self.convs2):

            x_residual = x

            xt = self.snake1(x)
            xt = apply_mask(xt, x_mask)
            xt = conv1(xt)

            xt = self.snake2(xt)
            xt = apply_mask(xt, x_mask)
            xt = conv2(xt)

            x = xt + x_residual
            x = apply_mask(x, x_mask)

        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm_legacy_safe(conv)

def fgss_generator(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    r: float = 0.97,
    random_init_phase: bool = True,
    power_factor: float = 0.1,
    max_frequency: Optional[float] = None,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finite Geometric Sine Series (FGSS) excitation signal generator.
    """
    batch, _, _ = f0.size()
    device = f0.device

    upsampler = PchipF0UpsamplerTorch(scale_factor=hop_length).to(device)
    f0_upsampled = upsampler(f0)

    if torch.all(f0_upsampled < 1.0):
        _, _, total_length = f0_upsampled.size()
        zeros = torch.zeros((batch, 1, total_length), device=device, dtype=f0_upsampled.dtype)
        return zeros, zeros

    voiced_mask = (f0_upsampled > 1.0).float()

    phase_increment_f64 = f0_upsampled.double() / sample_rate
    if random_init_phase:
        init_phase = torch.rand((1, 1), device=device, dtype=torch.float64)
        #init_phase = torch.rand((batch, 1, 1), device=device, dtype=torch.float64)
        phase_increment_f64[:, :, :1] += init_phase
    phase_cycles_f64 = torch.cumsum(phase_increment_f64, dim=2)

    # Phase for FusedGeoSaw: frac of cycle in [0, 2π), exactly 0 at period start
    frac = phase_cycles_f64 % 1.0
    floor_curr = torch.floor(phase_cycles_f64)
    floor_prev = F.pad(floor_curr[:, :, :-1], (1, 0), value=0.0)
    reset = (floor_curr > floor_prev) & (f0_upsampled > 1.0)
    frac_snapped = torch.where(reset, torch.zeros_like(frac), frac)
    phase = (frac_snapped * (2.0 * torch.pi)).float()

    nyquist = sample_rate / 2.0
    limit_freq = max_frequency if max_frequency is not None else nyquist
    safe_f0 = torch.clamp(f0_upsampled, min=1e-5)
    N = torch.floor(limit_freq / safe_f0)

    # Fused kernel for FGSS
    harmonics = FusedGeoSaw.apply(phase, N, r, epsilon)

    # Normalization
    r2  = r * r
    r2N = torch.pow(torch.tensor(r, device=device, dtype=phase.dtype), 2.0 * N)

    amp_scale = power_factor * torch.sqrt(2.0 * (1.0 - r2) / (r2 * torch.clamp(1.0 - r2N, min=epsilon)))
    signal = harmonics * amp_scale * voiced_mask

    return signal, f0_upsampled


class ExcitationSynthesizer(nn.Module):
    """
    Synthesizes the excitation signal from F0.
    """
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 480,
        random_init_phase: bool = True,
        power_factor: float = 0.1,
        add_noise_std: float = 0.003,
        r: float = 0.97,
    ):
        super(ExcitationSynthesizer, self).__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.random_init_phase = random_init_phase
        self.power_factor = power_factor
        self.noise_std = add_noise_std
        self.r = r
        self.l_linear = torch.nn.Linear(1, 1, bias=False)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, f0, upsample_factor = None):
        hop = upsample_factor if upsample_factor is not None else self.hop_length

        with autocast('cuda', enabled=False):
            f0 = f0.float()

            with torch.no_grad():
                fgss_harmonic_signal, f0_upsampled = fgss_generator(
                    f0,
                    hop_length=hop,
                    sample_rate=self.sample_rate,
                    random_init_phase=self.random_init_phase,
                    power_factor=self.power_factor,
                    r=self.r,
                )

            voiced_mask = (f0_upsampled > 1.0).float()
            noise_amp = voiced_mask * self.noise_std + (1.0 - voiced_mask) * (self.power_factor / 3.0)
            noise = torch.randn_like(fgss_harmonic_signal) * noise_amp
            excitation_signal = fgss_harmonic_signal + noise

        excitation_signal = excitation_signal.to(dtype=self.l_linear.weight.dtype)
        excitation_signal = excitation_signal.transpose(1, 2)
        excitation = self.l_tanh(self.l_linear(excitation_signal))
        excitation = excitation.transpose(1, 2)

        return excitation


class APEX_GAN_Generator(nn.Module):
    """
    Experimental neural vocoder for GAN-based voice synthesis.

    APEX stands for:
        A  — Adaptive harmonics  ( N scales with F0 to stay below Nyquist )
        P  — Pyramid injection   ( Each generator stage receives its own low-passed excitation level )
        EX — Complex excitation  ( Band-limited sawtooth with sufficient pitch/harmonic coverage )
    """

    def __init__(
        self,
        initial_channel, # 192
        resblock_kernel_sizes, # [3, 7, 11]
        resblock_dilation_sizes, # variable
        upsample_rates, # variable
        upsample_initial_channel, # 512
        upsample_kernel_sizes, # variable
        gin_channels, # 256
        sr,
    ):
        super(APEX_GAN_Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.total_ups_factor = math.prod(upsample_rates)

        # Excitation source
        self.excitation_synthesizer = ExcitationSynthesizer(
            sample_rate=sr,
            hop_length=self.total_ups_factor,
            random_init_phase=True,
            power_factor=0.1,
            add_noise_std=0.003,
            r=0.97,
        )

        # lowpass pyramid
        self.excitation_pyramid = ExcitationPyramid(upsample_rates=upsample_rates)

        # Pre convolution
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.exc_proj = nn.ModuleList()
        self.conv_post = nn.ModuleList()

        ch = ch_conv_post = upsample_initial_channel  # 512
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch //= 2  # 256 -> 128 -> 64 -> 32

            # Upsamplers
            self.ups.append(create_ups_convtranspose1d_layer(2 * ch, ch, k, u))

            # Residual blocks with snake activation
            for j, (kk, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, kk, d))

            # Excitation projection: [B,1,T_stage] --> [B,ch,T_stage]
            self.exc_proj.append(nn.Conv1d(1, ch, kernel_size=1, bias=False))

        # Post convolution
        for i in range(self.num_upsamples):
            ch_conv_post //= 2
            if i >= self.num_upsamples - 3:
                self.conv_post.append(weight_norm(Conv1d(ch_conv_post, 1, 7, 1, padding=3, bias=False)))
            else:
                self.conv_post.append(nn.Identity())

        # Speaker embedding conditioning
        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None, return_intermediates: bool = False):
        # x:  [B, 192, T]
        # f0: [B, T] or [B, 1, T]

        # Prepare frame-level f0
        f0 = f0.unsqueeze(1) if f0.dim() == 2 else f0

        # Generate full-rate excitation waveform [B, 1, T]
        excitation = self.excitation_synthesizer(f0)

        # Decompose into lowpass levels - one per stage, coarse to fine
        exc_levels: List[torch.Tensor] = self.excitation_pyramid(excitation)

        # Feature pre-conv ( 192 --> 512 )
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        outs = []
        for i in range(self.num_upsamples):
            x = F.silu(x)
            x = self.ups[i](x)

            # Project lowpass level [B,1,T_stage] --> [B,ch,T_stage] and inject
            exc_i = self.exc_proj[i](exc_levels[i])

            # Additive injuection of excitation to feats
            x.add_(exc_i)
            #x = x + exc_i

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

            if i >= self.num_upsamples - 3:
                _x = F.silu(x)
                _x = self.conv_post[i](_x)
                _x = torch.tanh(_x)
                outs.append(_x)

        return outs if return_intermediates else outs[-1]

    def remove_weight_norm(self):
        # pre convolution
        remove_weight_norm_legacy_safe(self.conv_pre)
        # upsamplers
        for l in self.ups:
            remove_weight_norm_legacy_safe(l)
        # ResBlocks
        for l in self.resblocks:
            l.remove_weight_norm()
        # post convolution
        for l in self.conv_post:
            if not isinstance(l, nn.Identity):
                remove_weight_norm_legacy_safe(l)

    def __prepare_scriptable__(self):
        # pre convolution
        for hook in self.conv_pre._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm_legacy_safe(self.conv_pre)
        # upsamplers
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm_legacy_safe(l)
        # ResBlocks
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm_legacy_safe(l)
        # post convolution
        for l in self.conv_post:
            if not isinstance(l, nn.Identity):
                for hook in l._forward_pre_hooks.values():
                    if (
                        hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                        and hook.__class__.__name__ == "WeightNorm"
                    ):
                        remove_weight_norm_legacy_safe(l)

        return self