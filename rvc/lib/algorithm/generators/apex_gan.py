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
from torch.amp import autocast  # guard

from rvc.lib.algorithm.generators.apex_gan_modules import Snake, snake_kaiming_normal_, snake_kaiming_uniform_

from rvc.lib.algorithm.generators.apex_gan_modules.stft import TorchSTFT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    m = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                  padding=(kernel_size - stride) // 2)
    return weight_norm(m)


def create_resblock_conv1d_layer(channels, kernel_size, dilation, snake_init_variant='normal', init_value=None):
    conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation,padding=get_conv1d_padding(kernel_size, dilation))
    kind = init_value if init_value is not None else 'approx'

    if snake_init_variant == 'uniform':
        snake_kaiming_uniform_(conv.weight, kind=kind)
    elif snake_init_variant == 'normal':
        snake_kaiming_normal_(conv.weight, kind=kind)

    return weight_norm(conv)


# ---------------------------------------------------------------------------
# Phase-only f0 controller
# ---------------------------------------------------------------------------
#
# Physical motivation
# --------------------
# The excitation carries two structurally different kinds of information:
#
#   1. PERIODIC / voiced content (the fundamental sine derived from f0).
#      This should act purely as a *phase reference* -- it tells the
#      network WHERE in the cycle we are, not WHAT amplitude/shape the
#      harmonics should have. Amplitude/shape must come from z.
#
#   2. APERIODIC / unvoiced content (breath, sibilants, UV noise).
#      This is genuine broadband ENERGY that z realistically cannot
#      cheaply reconstruct on its own, so it is legitimate to inject it
#      additively, the same way the original implementation did.
#
# The bug being fixed here is that the old code injected BOTH as a single
# combined, amplitude-carrying additive signal. Snake's own nonlinearity
# (sin(alpha*x)^2) is exactly the kind of nonlinearity that can turn a
# clean, high-amplitude sine into a full harmonic stack on its own,
# without needing any information from z/x. That let f0 alone "explain"
# voiced harmonic content, starving z of gradient.
#
# The fix: split the excitation into (a) a phase-only control signal for
# the periodic part, injected directly into Snake's argument via the
# angle-sum identity sin(a*x + beta) = sin(a*x)cos(beta) + cos(a*x)sin(beta),
# and (b) a magnitude-only energy signal for the noise part, injected
# additively as before. (cos_beta, sin_beta) is explicitly re-normalized
# to unit norm every time it's produced, which is what structurally
# prevents this pathway from smuggling amplitude back in -- it can only
# ever encode a rotation.

class PhaseProjector(nn.Module):
    """
    Projects a phase-only spectral reference (STFT phase of the PURE
    voiced sine excitation -- magnitude discarded before this ever sees it)
    down to a per-stage (cos_beta, sin_beta) pair, strided to match that
    stage's temporal resolution.

    Output is unit-normalized, so this module can only ever encode a
    rotation. It has no channel through which to smuggle amplitude/energy
    into the Snake activations it conditions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        stride = max(int(stride), 1)
        kernel_size = stride * 2 if stride > 1 else 1
        padding = (stride + 1) // 2 if stride > 1 else 0
        self.proj = nn.Conv1d(
            in_channels, out_channels * 2,
            kernel_size=kernel_size, stride=stride, padding=padding,
        )

    def forward(self, sine_phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.proj(sine_phase)
        cos_raw, sin_raw = out.chunk(2, dim=1)
        norm = torch.sqrt(cos_raw * cos_raw + sin_raw * sin_raw + 1e-6)
        return cos_raw / norm, sin_raw / norm


# ---------------------------------------------------------------------------
# ResBlock  (extended: optional phase-only beta into the first Snake per layer)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    A residual block module that applies a series of 1D convolutional layers
    with residual connections.

    `beta`, if provided, is a (cos_beta, sin_beta) unit-norm pair that
    phase-shifts the FIRST Snake activation in each layer (s1). The second
    Snake (s2) is left phase-free deliberately, to keep the f0-control
    surface area minimal and easy to isolate/ablate.
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
            Snake(channels, init=1.0, correction=None) for _ in dilations
        ])
        self.snakes2 = nn.ModuleList([
            Snake(channels, init=1.0, correction=None) for _ in dilations
        ])

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        return nn.ModuleList([
            create_resblock_conv1d_layer(channels, kernel_size, d, snake_init_variant='normal', init_value=1.0)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None, beta=None):
        for conv1, conv2, s1, s2 in zip(self.convs1, self.convs2, self.snakes1, self.snakes2):
            x_residual = x

            xt = s1(x, beta=beta)
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


# ---------------------------------------------------------------------------
# Excitation synthesizer
# ---------------------------------------------------------------------------

class SineGenerator(torch.nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        sine_amplitude: float = 0.1,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.sine_amplitude = sine_amplitude

    def _compute_voiced_unvoiced(self, f0: torch.Tensor):
        return (f0 > 0.0).float()

    def _generate_sine_wave(
        self,
        f0: torch.Tensor,
        upsampling_factor: int,
    ):
        batch_size, length, _ = f0.shape

        upsampling_grid = torch.arange(
            1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device,
        )

        phase = (f0 / self.sampling_rate) * upsampling_grid

        # accumulate in fp32 for numerical stability
        phase_remainder = (torch.fmod(phase[:, :-1, -1:].float() + 0.5, 1.0) - 0.5)
        cumulative_phase = (phase_remainder.cumsum(dim=1).fmod(1.0).to(f0.dtype))

        phase += torch.nn.functional.pad(cumulative_phase, (0, 0, 1, 0))
        phase = phase.reshape(batch_size, -1, 1)
        phase.remainder_(1.0)
        phase.mul_(2 * math.pi)

        torch.sin(phase, out=phase)

        return phase

    def forward(
        self,
        f0: torch.Tensor,
        upsampling_factor: int,
    ):
        with torch.no_grad():

            f0 = f0.unsqueeze(-1)

            sine = self._generate_sine_wave(f0, upsampling_factor)
            sine.mul_(self.sine_amplitude)

            # Voiced mask: nearest-neighbor upsample from (B, T, 1) to (B, T*factor, 1)
            # f0 is already either >0 (voiced) or 0 (unvoiced) per frame, so no
            # interpolation of values is needed — just repeat each frame hop_length times.
            voiced_mask_lo = (f0 > 0.0).float()                   # (B, T, 1)
            voiced = voiced_mask_lo.transpose(2, 1)               # (B, 1, T)
            voiced = F.interpolate(voiced, scale_factor=upsampling_factor, mode='nearest')
            voiced = voiced.transpose(2, 1)                        # (B, T*factor, 1)

            sine.mul_(voiced)

        return sine, voiced


class ExcitationSynthesizer(nn.Module):
    """
    Synthesizes the excitation source from F0.

    Returns TWO tensors:

      sine_only: (B, 1, T_audio)  pure voiced sine, zero noise. This is
                 STFT'd upstream and only its PHASE is used -- magnitude is
                 discarded, so it cannot carry an exploitable amplitude cheat.
      noise_only:(B, 1, T_audio)  pure noise (voiced: tiny residual noise
                 floor; unvoiced: full aperiodic energy for breath/UV/
                 sibilants). STFT'd upstream and decomposed into real+imag
                 parts which scale with signal amplitude AND decorrelate
                 across frames to prevent vertical-line artifacts.
    """
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 480,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.sine_amp = sine_amp
        self.add_noise_std = add_noise_std

        self.sine_gen = SineGenerator(sample_rate, sine_amp)

        self.l_linear = nn.Linear(1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0, upsample_factor=None):
        hop = upsample_factor if upsample_factor is not None else self.hop_length

        with autocast('cuda', enabled=False):
            f0 = f0.float()
            # fundamental harmonic, 1 sine -- PURE, no noise mixed in yet
            harmonic_signal, voiced_mask = self.sine_gen(f0, hop)

            # White noise. Voiced: tiny residual (breath-under-voice).
            # Unvoiced: full-strength aperiodic energy (UV / breath / sibilants).
            noise_amp = voiced_mask * self.add_noise_std + (1.0 - voiced_mask) * (self.sine_amp / 3.0)
            noise = noise_amp * torch.randn_like(harmonic_signal)

            # Combined signal kept only for optional external diagnostics.
            #combined = harmonic_signal + noise
            #combined = self.l_tanh(self.l_linear(combined))
            #combined = combined.to(dtype=f0.dtype)

            sine_only = harmonic_signal.to(dtype=f0.dtype)
            noise_only = noise.to(dtype=f0.dtype)

        return (
            #combined.transpose(1, 2),     # (B, 1, T_audio) -- diagnostics only
            sine_only.transpose(1, 2),    # (B, 1, T_audio) -- phase source
            noise_only.transpose(1, 2),   # (B, 1, T_audio) -- energy source
        )


class APEX_GAN_Generator(nn.Module):
    """
    Spectral-output neural vocoder (APEX-GAN, HiFTNet-paradigm).

    APEX stands for:
        A  — Adaptive harmonics  ( N scales with F0 to stay below Nyquist )
        P  — Pyramid injection   ( Each generator stage receives a strided
                                   version of the harmonic spectrum )
        EX — Complex excitation  ( iSTFT-ready spectral output )

    F0 control is injected at EVERY upsampling stage as a PHASE-ONLY signal
    (via PhaseProjector -> phase-conditioned Snake), so it can steer WHERE
    harmonics land but cannot by itself supply the amplitude/shape that
    becomes the reconstructed harmonic content -- that has to come from x
    (i.e. from z, through conv_pre / cond / the resblock convs).

    Aperiodic energy (breath / UV / sibilants) is still injected additively
    at every stage via noise_convs, same as before -- that part of the
    excitation is genuine energy z realistically can't invent, so there is
    no reason to bottleneck it.

    Returns
    -------
    audio : (B, 1, T_audio)  reconstructed waveform via iSTFT
    """

    def __init__(
        self,
        initial_channel,           # 192
        resblock_kernel_sizes,     # [3, 7, 11]
        resblock_dilation_sizes,   # [[1,3,5], [1,3,5], [1,3,5]]
        upsample_rates,            # e.g. [12, 10]  for 48 kHz
        upsample_initial_channel,  # 512
        upsample_kernel_sizes,     # e.g. [24, 20]
        gin_channels,              # 256
        sr,
        gen_istft_n_fft: int = 32,
        gen_istft_hop_size: int = 4,
        checkpointing: bool = False,   # reserved / unused
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Total upsampling factor across all conv-transpose stages.
        # The full hop (frame → sample) also includes the iSTFT hop.
        self.total_ups_factor = math.prod(upsample_rates)

        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        # How many samples does one frame expand to (conv stages + iSTFT hop)?
        self.full_hop = self.total_ups_factor * gen_istft_hop_size

        # ------------------------------------------------------------------
        # Excitation synthesizer
        # ------------------------------------------------------------------
        self.excitation_synthesizer = ExcitationSynthesizer(
            sample_rate=sr,
            hop_length=self.full_hop,
            sine_amp=0.1,
            add_noise_std=0.003,
        )

        # ------------------------------------------------------------------
        # Pre-conv
        # ------------------------------------------------------------------
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))

        # ------------------------------------------------------------------
        # Upsampler stages, per-stage injection (energy + phase), main resblocks
        # ------------------------------------------------------------------
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()   # aperiodic ENERGY injection (real amplitude, legitimate)
        self.phase_convs = nn.ModuleList()   # periodic PHASE-ONLY injection (no amplitude, f0 control)
        self.resblocks = nn.ModuleList()

        # STFT of the excitation signals produces (n_fft//2 + 1) magnitude/phase bins.
        stft_bins = gen_istft_n_fft // 2 + 1
        # noise_convs take real+imaginary concatenated (= n_fft+2 channels)
        noise_channels = gen_istft_n_fft + 2

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch  = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))

            # Upsampler
            self.ups.append(weight_norm(
                ConvTranspose1d(in_ch, out_ch, k, u, padding=(k - u) // 2)
            ))

            # Stride needed to bring the excitation's STFT-frame resolution
            # down to this stage's resolution.
            if i + 1 < self.num_upsamples:
                stride_f0 = math.prod(upsample_rates[i + 1:])
            else:
                # Last stage: excitation and x are already at the same resolution.
                stride_f0 = 1

            # Aperiodic energy injector (breath / UV / sibilants) -- additive, real amplitude.
            # Takes [noise_mag, noise_phase * envelope] (n_fft+2 channels).
            # Phase channels carry per-frame decorrelation AND amplitude envelope,
            # preventing coherent comb patterns while preserving V/UV ratio.
            if stride_f0 > 1:
                self.noise_convs.append(Conv1d(
                    noise_channels, out_ch,
                    kernel_size=stride_f0 * 2,
                    stride=stride_f0,
                    padding=(stride_f0 + 1) // 2,
                    bias=False,
                ))
            else:
                self.noise_convs.append(Conv1d(noise_channels, out_ch, kernel_size=1, bias=False))

            # Periodic phase-only f0 controller -- feeds Snake's argument, not x.
            self.phase_convs.append(PhaseProjector(stft_bins, out_ch, stride_f0))

            # Main resblocks
            for kk, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(out_ch, kk, d))

        # ------------------------------------------------------------------
        # Post-conv: outputs (gen_istft_n_fft + 2) channels
        #   [:n_fft//2+1]  → magnitude (will be exp'd by caller or here)
        #   [n_fft//2+1:]  → phase     (will be sin'd by caller or here)
        # ------------------------------------------------------------------
        final_ch = upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_post = weight_norm(
            Conv1d(final_ch, gen_istft_n_fft + 2, 7, 1, padding=3)
        )

        # Reflection pad before final conv (matches HiFTNet)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))

        # ------------------------------------------------------------------
        # Speaker conditioning
        # ------------------------------------------------------------------
        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

        # STFT
        self.stft = TorchSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                    # (B, initial_channel, T_frames)
        f0: torch.Tensor,                   # (B, T_frames) or (B, 1, T_frames)
        g: Optional[torch.Tensor] = None,   # (B, gin_channels, 1)
    ) -> torch.Tensor:
        """
        Returns
        -------
        audio : (B, 1, T_audio)
        """

        # ---- Generate excitation, split into phase-only / energy-only parts ----
        sine_only, noise_only = self.excitation_synthesizer(f0)

        # Pure voiced sine -> STFT -> keep PHASE, discard magnitude
        # normalize the sine to unit peak amplitude before STFT
        peak = sine_only.abs().amax(dim=-1, keepdim=True)  # (B, 1, 1)
        sine_unit = sine_only / (peak + 1e-8)
        _, sine_phase = self.stft.transform(sine_unit.squeeze(1).float())
        sine_phase = sine_phase.to(x.dtype)

        # Pure noise -> STFT -> mag + phase.
        # phase channels carry per-frame incoherence (prevents comb artifacts)
        # but raw phase has constant RMS ~1.81 regardless of signal amplitude,
        # which destroys the V/UV amplitude ratio and lets the conv learn a
        # coherent comb filter.
        # Fix: pre-scale phase by the noise magnitude envelope so the conv
        # sees both decorrelation AND amplitude structure from the start.
        # This prevents the comb (input amplitude varies frame-to-frame)
        # while preserving the V/UV ratio (envelope carries it).
        noise_mag, noise_phase = self.stft.transform(noise_only.squeeze(1).float())
        noise_mag = noise_mag.to(x.dtype)
        noise_phase = noise_phase.to(x.dtype)

        # Per-frame amplitude envelope, normalized to unit mean.
        # Computed once (not per-stage) since it's used by all noise_convs.
        envelope = noise_mag.mean(dim=1, keepdim=True)              # (B, 1, T_stft)
        envelope = envelope / (envelope.mean() + 1e-8)

        # Bake the envelope into phase channels before the conv sees them.
        # Phase still varies randomly frame-to-frame (decorrelation preserved),
        # but its scale now tracks the actual noise amplitude (V/UV preserved).
        har_noise = torch.cat([noise_mag, noise_phase * envelope], dim=1)  # (B, n_fft+2, T_stft)

        # ---- Pre-conv + speaker conditioning -----------------------------
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        # ---- Upsample stages with dual injection --------------------------
        for i in range(self.num_upsamples):
            #x = F.leaky_relu(x, negative_slope=0.1)
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)

            # Reflection-pad before the last upsampler
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # Real aperiodic energy injection (breath / UV / sibilants).
            # The conv sees [mag, phase] for artifact prevention (phase channels
            # break up coherent temporal patterns).
            energy_source = self.noise_convs[i](har_noise)
            x = x + energy_source

            # Phase-only f0 control for this stage's Snake activations.
            # cos_beta/sin_beta are unit-norm by construction (see
            # PhaseProjector) -- they can only rotate Snake's argument,
            # never add amplitude.
            beta = self.phase_convs[i](sine_phase)

            # Main multi-kernel resblocks, phase-conditioned
            xs = None
            for j in range(self.num_kernels):
                rb_out = self.resblocks[i * self.num_kernels + j](x, beta=beta)
                if xs is None:
                    xs = rb_out
                else:
                    xs += rb_out
            x = xs / self.num_kernels

        # ---- Post-conv -------------------------
        #x = F.leaky_relu(x, negative_slope=0.01)
        x = F.silu(x, inplace=True)

        x = self.conv_post(x)                           # (B, n_fft+2, T_stft)

        # ---- split into spec + phase → inverse -------------------------
        with autocast('cuda', enabled=False):

            n_fft = self.gen_istft_n_fft
            x = x.float()
            spec  = torch.exp(x[:, :n_fft // 2 + 1, :])    # magnitude (always > 0)
            phase = torch.sin(x[:, n_fft // 2 + 1:, :])    # wrapped phase ∈ [-1, 1]

            audio = self.stft.inverse(spec, phase)            # (B, 1, T_audio)

        return audio #.squeeze(1)                           # (B, 1, T_audio)

    # ------------------------------------------------------------------
    # Weight-norm removal
    # ------------------------------------------------------------------

    def remove_weight_norm(self):
        remove_weight_norm_legacy_safe(self.conv_pre)
        remove_weight_norm_legacy_safe(self.conv_post)
        for m in self.ups:
            remove_weight_norm_legacy_safe(m)
        for m in self.resblocks:
            m.remove_weight_norm()

    def __prepare_scriptable__(self):
        _remove_wn_if_present(self.conv_pre)
        _remove_wn_if_present(self.conv_post)
        for m in self.ups:
            _remove_wn_if_present(m)
        for rb in self.resblocks:
            for conv in chain(rb.convs1, rb.convs2):
                _remove_wn_if_present(conv)
        return self
