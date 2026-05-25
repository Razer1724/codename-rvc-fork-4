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

from rvc.lib.algorithm.generators.apex_gan_modules import PchipF0UpsamplerTorch, FusedDirichlet, SnakeBeta, snake_kaiming_uniform_, snake_kaiming_normal_

import json

def get_conv1d_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]):
    return tensor * mask if mask is not None else tensor

def remove_weight_norm_legacy_safe(module):
    if is_parametrized(module, "weight"):
        remove_parametrizations(module, "weight", leave_parametrized=True)
    else:
        remove_weight_norm(module)

def _remove_wn_if_present(module: nn.Module) -> None:
    if any(
        hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
        and hook.__class__.__name__ == "WeightNorm"
        for hook in module._forward_pre_hooks.values()
    ):
        remove_weight_norm_legacy_safe(module)

def create_ups_convtranspose1d_layer(in_channels, out_channels, kernel_size, stride):
    m = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - stride) // 2)
    return weight_norm(m)

def create_resblock_conv1d_layer(channels, kernel_size, dilation, snake_init_variant='normal', init_value=None):
    conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=get_conv1d_padding(kernel_size, dilation))
    if snake_init_variant == 'uniform':
        snake_kaiming_uniform_(conv.weight, init_value=init_value)
    elif snake_init_variant == 'normal':
        snake_kaiming_normal_(conv.weight, init_value=init_value)
    return weight_norm(conv)




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

        self.snakes1 = nn.ModuleList([
            SnakeBeta(channels, init=1.0, synced_start=True, correction=True) for _ in dilations
        ])
        self.snakes2 = nn.ModuleList([
            SnakeBeta(channels, init=1.0, synced_start=True, correction=True) for _ in dilations
        ])

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        return nn.ModuleList(
            [create_resblock_conv1d_layer(channels, kernel_size, d, snake_init_variant='normal', init_value=1.0) for d in dilations]
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for conv1, conv2, s1, s2 in zip(self.convs1, self.convs2, self.snakes1, self.snakes2):
            x_residual = x

            xt = s1(x)
            xt = apply_mask(xt, x_mask)
            xt = conv1(xt)

            xt = s2(xt)
            xt = apply_mask(xt, x_mask)
            xt = conv2(xt)

            x = xt + x_residual
            x = apply_mask(x, x_mask)

        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm_legacy_safe(conv)




class ExcSTFTProj_v2(nn.Module):
    def __init__(self, ch: int, sr: int, stage_hop: int = 1):
        super().__init__()
        self.n_fft = self._pick_n_fft(sr)
        self.stage_hop = stage_hop
        self.chunk_len = sr * 10
        self._aligned_overlap = math.ceil((self.n_fft // 2) / self.stage_hop) * self.stage_hop

        n_bins = (self.n_fft // 2 + 1) * 2

        self.proj_linear = weight_norm(nn.Linear(n_bins, ch, bias=False))
        self.register_buffer('filterbank', self._make_filterbank(self.n_fft))

        self._fused_w: Optional[torch.Tensor] = None

        # output_scale: normalize filterbank accumulation + channel width
        channel_gain = math.sqrt(256 / ch)
        self.register_buffer('output_scale',
            torch.tensor(math.sqrt(n_bins) * channel_gain))

    @staticmethod
    def _pick_n_fft(sr: int, f0_min: float = 32.0) -> int:
        n = 16
        while n < sr / f0_min:
            n <<= 1
        return n

    # filterbank v3
    @staticmethod
    def _make_filterbank(n_fft: int) -> torch.Tensor:
        t = torch.arange(n_fft, dtype=torch.float) - (n_fft // 2)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float)

        hann = 0.5 * (1 + torch.cos(2 * math.pi * t / n_fft))
        scale = 2.0 / hann.sum()

        angle = 2 * math.pi * k[:, None] * t[None, :] / n_fft
        real = hann * torch.cos(angle)
        imag = hann * -torch.sin(angle)

        fb = torch.cat([real, imag], dim=0) * scale
        return fb

    def _compute_fused_w(self) -> torch.Tensor:
        return (self.proj_linear.weight.float() @ self.filterbank.float()).unsqueeze(1).contiguous()

    def _forward_chunk(self, exc: torch.Tensor) -> torch.Tensor:
        w = self._fused_w if self._fused_w is not None else self._compute_fused_w()
        with torch.amp.autocast('cuda', enabled=False):
            pad = self.n_fft // 2
            x = F.pad(exc.float(), (pad, pad), mode='reflect')
            out = F.conv1d(x, w.float(), stride=self.stage_hop)
            # expected output length: T // stage_hop  ( drop the trailing +1 )
            out = out[:, :, : exc.shape[-1] // self.stage_hop]
            out = out * self.output_scale
        return out.to(exc.dtype)

    def forward(self, exc: torch.Tensor) -> torch.Tensor:
        T = exc.shape[-1]
        if T <= self.chunk_len:
            return self._forward_chunk(exc)
        overlap = self._aligned_overlap  # self.n_fft // 2
        out_chunks = []
        start = 0
        while start < T:
            end = min(start + self.chunk_len, T)
            c_start = max(0, start - overlap)
            c_end = min(T,   end  + overlap)
            chunk_out = self._forward_chunk(exc[:, :, c_start:c_end])
            trim_l = (start - c_start) // self.stage_hop
            trim_r = trim_l + (end - start) // self.stage_hop
            out_chunks.append(chunk_out[:, :, trim_l:trim_r])
            start = end
        return torch.cat(out_chunks, dim=-1)

    def _apply(self, fn):
        self._fused_w = None
        return super()._apply(fn)

    def remove_weight_norm(self):
        remove_weight_norm_legacy_safe(self.proj_linear)
        self._fused_w = self._compute_fused_w() # fuse




def pcph_generator(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    random_init_phase: bool = True,
    power_factor: float = 0.1,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pseudo-Constant-Power Harmonics (PCPH) excitation signal generator.
    """
    batch, _, frames = f0.size()
    device = f0.device

    if torch.all(f0 == 0.0): # (f0 < 1.0)
        zeros = torch.zeros((batch, 1, frames * hop_length), device=device)
        return zeros, zeros

    upsampler = PchipF0UpsamplerTorch(scale_factor=hop_length).to(device)
    f0_upsampled = upsampler(f0)

    voiced_mask = (f0_upsampled > 1.0).float()

    phase_increment_f64 = f0_upsampled.double() / sample_rate
    if random_init_phase:
        init_phase = torch.rand((1, 1), device=device, dtype=torch.float64)
        phase_increment_f64[:, :, :1] += init_phase

    # Phase for FusedDirichlet: cumulative cycles in [0, 1), unwrapped then remainder
    phase_cycles_f64 = torch.cumsum(phase_increment_f64, dim=2)
    phase_cycles_f64 = torch.remainder(phase_cycles_f64, 1.0).float()

    # Dynamic harmonic count: max harmonics below Nyquist at each sample
    safe_f0 = torch.clamp(f0_upsampled, min=1.0)
    N = torch.floor(sample_rate / (2.0 * safe_f0))

    # Fused kernel for PCPH (Dirichlet sines sum)
    harmonics = FusedDirichlet.apply(phase_cycles_f64, N, epsilon)

    # Normalization: pseudo-constant power across varying harmonic count
    amp_scale = power_factor * torch.sqrt(2.0 / torch.clamp(N, min=1.0))
    signal = harmonics * amp_scale * voiced_mask

    return signal, f0_upsampled


class ExcitationSynthesizer(nn.Module):
    """
    Synthesizes the excitation source from F0:

    - Voiced: Pseudo-Constant-Power Harmonics (PCPH) via fused Dirichlet kernel.
        Natively band-limited up to the Nyquist frequency to prevent aliasing.

    - Unvoiced: Adaptive Gaussian noise.
        Provides a stochastic foundation for sibilance and breath synthesis.
    """
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 480,
        random_init_phase: bool = True,
        power_factor: float = 0.1,
        add_noise_std: float = 0.003,
    ):
        super(ExcitationSynthesizer, self).__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.random_init_phase = random_init_phase
        self.power_factor = power_factor
        self.noise_std = add_noise_std

    def forward(self, f0, upsample_factor = None):
        hop = upsample_factor if upsample_factor is not None else self.hop_length

        with autocast('cuda', enabled=False):
            f0 = f0.float()

            with torch.no_grad():
                harmonic_signal, f0_upsampled = pcph_generator(
                    f0,
                    hop_length=hop,
                    sample_rate=self.sample_rate,
                    random_init_phase=self.random_init_phase,
                    power_factor=self.power_factor,
                )

            voiced_mask = (f0_upsampled > 1.0).float()
            noise_amp = voiced_mask * self.noise_std + (1.0 - voiced_mask) * (self.power_factor / 3.0)

            noise = torch.randn_like(harmonic_signal) * noise_amp 

            excitation_signal = harmonic_signal + noise
            excitation_signal = excitation_signal.to(dtype=f0.dtype)

        return excitation_signal


class APEX_GAN_Generator(nn.Module):
    """
    Experimental neural vocoder for GAN-based voice synthesis.

    APEX stands for:
        A  — Adaptive harmonics  ( N scales with F0 to stay below Nyquist )
        P  — Pyramid injection   ( Each generator stage receives its own low-passed excitation level )
        EX — Complex excitation  ( Spectrally-projected )
    """

    def __init__(
        self,
        initial_channel, # 192
        resblock_kernel_sizes, # [3, 7, 11]
        resblock_dilation_sizes, # [1, 3, 5] * 3
        upsample_rates, # variable
        upsample_initial_channel, # 512
        upsample_kernel_sizes, # variable
        gin_channels, # 256
        sr,
        checkpointing: bool = False,  # For now unused.
    ):
        super(APEX_GAN_Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.total_ups_factor = math.prod(upsample_rates)

        # Excitation synthesizer
        self.excitation_synthesizer = ExcitationSynthesizer_v3(
            sample_rate=sr,
            hop_length=self.total_ups_factor,
            random_init_phase=False,
            power_factor=0.1,
            add_noise_std=0.003
        )

        # Pre convolution
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()            # Upsamplers
        self.resblocks = nn.ModuleList()      # Residual Blocks
        self.exc_proj = nn.ModuleList()       # Spectral excitation projections
        self.conv_post = nn.ModuleList()      # Post convolutions

        ch = ch_conv_post = upsample_initial_channel  # 512

        # Main loop:  input -> upsample -> excitation proj -> resblocks
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch //= 2  # 256 -> 128 -> 64 -> 32
            stage_hop = math.prod(upsample_rates[i + 1:]) if i + 1 < len(upsample_rates) else 1 # for v2

            # Upsamplers
            self.ups.append(create_ups_convtranspose1d_layer(2 * ch, ch, k, u))

            # Excitation projection
            self.exc_proj.append(ExcSTFTProj_v2(ch, sr=sr, stage_hop=stage_hop))

            # Residual blocks
            for j, (kk, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, kk, d))

        # Post convolution heads
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
        # x:  [B, 192, T]  Frame count for SRs: 48, 40, 32, 24 ; 36, 38, 40, 42
        # f0: [B, T]

        # Prepare frame-level f0
        f0 = f0.unsqueeze(1) if f0.dim() == 2 else f0

        # Generate excitation
        excitation = self.excitation_synthesizer(f0)

        # Feature pre-conv
        x = self.conv_pre(x)

        # Initial spk conditioning
        if g is not None:
            x = x + self.cond(g)

        outs = []
        for i in range(self.num_upsamples):
            x = F.silu(x) # Activation
            x = self.ups[i](x) # Upsampling

            # Project lowpass level
            exc_i = self.exc_proj[i](excitation)

            # Source-Filter style Modulation
            x = x * (1.0 + torch.tanh(exc_i))

            # Resblocks processing
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
        # Single modules
        for m in [self.conv_pre]:
            remove_weight_norm_legacy_safe(m)
        # Upsamplers
        for m in self.ups:
            remove_weight_norm_legacy_safe(m)
        # Excitation projections
        for m in self.exc_proj:
            m.remove_weight_norm()
        # ResBlocks
        for m in self.resblocks:
            m.remove_weight_norm()
        # Post convolutions
        for m in self.conv_post:
            if not isinstance(m, nn.Identity):
                remove_weight_norm_legacy_safe(m)

    def __prepare_scriptable__(self):
        # Single modules
        for m in [self.conv_pre]:
            self._remove_wn_if_present(m)
        # Upsamplers
        for m in self.ups:
            self._remove_wn_if_present(m)
        # Excitation projections
        for m in self.exc_proj:
            m.remove_weight_norm()
        # ResBlocks
        for rb in self.resblocks:
            for conv in chain(rb.convs1, rb.convs2):
                _remove_wn_if_present(conv)
        # Post convolutions
        for m in self.conv_post:
            if not isinstance(m, nn.Identity):
                self._remove_wn_if_present(m)

        return self