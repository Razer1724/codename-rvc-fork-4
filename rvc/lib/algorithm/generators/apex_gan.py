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

from rvc.lib.algorithm.generators.apex_gan_modules import PchipF0UpsamplerTorch, FusedDirichlet, FusedGeoSaw, Snake, snake_kaiming_normal_, snake_kaiming_uniform_

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
# ResBlock  (unchanged from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Excitation synthesizer
# ---------------------------------------------------------------------------

def pcph_generator(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    random_init_phase: bool = False,
    power_factor: float = 0.1,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pseudo-Constant-Power Harmonics (PCPH) excitation signal generator.
    Returns (signal [B,1,T_audio], f0_upsampled [B,1,T_audio]).
    """
    batch, _, frames = f0.size()
    device = f0.device

    if torch.all(f0 == 0.0):
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


def fgss_generator(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    random_init_phase: bool = False,
    power_factor: float = 0.1,
    geosaw_r: float = 0.90,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch, _, frames = f0.size()
    device = f0.device

    if torch.all(f0 == 0.0):
        zeros = torch.zeros((batch, 1, frames * hop_length), device=device)
        return zeros, zeros

    upsampler = PchipF0UpsamplerTorch(scale_factor=hop_length).to(device)
    f0_upsampled = upsampler(f0)

    voiced_mask = (f0_upsampled > 1.0).float()

    phase_increment_f64 = f0_upsampled.double() / sample_rate
    if random_init_phase:
        init_phase = torch.rand((1, 1), device=device, dtype=torch.float64)
        phase_increment_f64[:, :, :1] += init_phase

    # Cumulative phase in cycles [0,1)
    phase_cycles_f64 = torch.cumsum(phase_increment_f64, dim=2)
    phase_cycles_f64 = torch.remainder(phase_cycles_f64, 1.0).float()

    # Radians calculation
    phase_rad = phase_cycles_f64 * (2.0 * math.pi)

    # Dynamic harmonic count
    safe_f0 = torch.clamp(f0_upsampled, min=1.0)
    N = torch.floor(sample_rate / (2.0 * safe_f0))

    # GeoSaw kernel
    harmonics = FusedGeoSaw.apply(phase_rad, N, geosaw_r, epsilon)

    # Fixed normalization
    signal = harmonics * (power_factor / 1.46) * voiced_mask

    return signal, f0_upsampled


class ExcitationSynthesizer_OLD(nn.Module):
    """
    Synthesizes the excitation source from F0:

    - Voiced:   Pseudo-Constant-Power Harmonics (PCPH) via fused Dirichlet kernel.
                Natively band-limited up to Nyquist to prevent aliasing.

    - Unvoiced: Adaptive Gaussian noise.
                Provides a stochastic foundation for sibilance and breath synthesis.
    """
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 480,
        random_init_phase: bool = False,
        power_factor: float = 0.1,
        add_noise_std: float = 0.003,
        geosaw_r: float = 0.90,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.random_init_phase = random_init_phase
        self.power_factor = power_factor
        self.noise_std = add_noise_std
        self.geosaw_r = geosaw_r

    def forward(self, f0, upsample_factor=None):
        hop = upsample_factor if upsample_factor is not None else self.hop_length

        with autocast('cuda', enabled=False):
            f0 = f0.float()

            with torch.no_grad():
                harmonic_signal, f0_upsampled = fgss_generator(
                    f0,
                    hop_length=hop,
                    sample_rate=self.sample_rate,
                    random_init_phase=self.random_init_phase,
                    power_factor=self.power_factor,
                    geosaw_r=self.geosaw_r,
                )

            voiced_mask = (f0_upsampled > 1.0).float()
            noise_amp = voiced_mask * self.noise_std + (1.0 - voiced_mask) * (self.power_factor / 3.0)
            noise = torch.randn_like(harmonic_signal) * noise_amp
            excitation_signal = harmonic_signal + noise
            excitation_signal = excitation_signal.to(dtype=f0.dtype)

        return excitation_signal  # (B, 1, T_audio)


class ExcitationSynthesizer(nn.Module):
    """
    Synthesizes the excitation source from F0.

    - Voiced:   GeoSaw harmonic signal (band-limited sawtooth).
    - Unvoiced: Learnable-shaped stochastic noise.
                A small FIR filter learns the spectral color of fricatives
                and breath, giving the network a richer prior than white noise.
    """
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 480,
        random_init_phase: bool = False,
        power_factor: float = 0.1,
        add_noise_std: float = 0.003,
        geosaw_r: float = 0.90,
        noise_shaper_duration_ms: float = 0.65,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.random_init_phase = random_init_phase
        self.power_factor = power_factor
        self.noise_std = add_noise_std
        self.geosaw_r = geosaw_r

        # Learnable noise shaper: 1-channel FIR filter that shapes white noise
        # into something closer to speech-like fricative/breath spectra.
        kernel_size = int(sample_rate * noise_shaper_duration_ms / 1000.0)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.noise_shaper = nn.Conv1d(
            1, 1, kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        # near-identity init (delta-like) so training starts from white noise
        with torch.no_grad():
            self.noise_shaper.weight.zero_()
            self.noise_shaper.weight[:, :, kernel_size // 2] = 1.0

    def forward(self, f0, upsample_factor=None):
        hop = upsample_factor if upsample_factor is not None else self.hop_length

        with autocast('cuda', enabled=False):
            f0 = f0.float()

            with torch.no_grad():
                harmonic_signal, f0_upsampled = fgss_generator(
                    f0,
                    hop_length=hop,
                    sample_rate=self.sample_rate,
                    random_init_phase=self.random_init_phase,
                    power_factor=self.power_factor,
                    geosaw_r=self.geosaw_r,
                )

            voiced_mask = (f0_upsampled > 1.0).float()

            # Base noise: white Gaussian
            raw_noise = torch.randn_like(harmonic_signal)

            # Shape the noise: the filter learns spectral color for unvoiced sounds
            # (e.g., high-shelf for sibilants, pink-ish for breath).
            # Voiced regions get near-unshaped noise (very low amplitude anyway).
            shaped_noise = self.noise_shaper(raw_noise)

            # Amplitude envelope: voiced gets subtle breath, unvoiced gets louder hiss
            noise_amp = voiced_mask * self.noise_std + (1.0 - voiced_mask) * (self.power_factor / 3.0)
            noise = shaped_noise * noise_amp

            excitation_signal = harmonic_signal + noise
            excitation_signal = excitation_signal.to(dtype=f0.dtype)

        return excitation_signal  # (B, 1, T_audio)


class APEX_GAN_Generator(nn.Module):
    """
    Spectral-output neural vocoder (APEX-GAN, HiFTNet-paradigm).

    APEX stands for:
        A  — Adaptive harmonics  ( N scales with F0 to stay below Nyquist )
        P  — Pyramid injection   ( Each generator stage receives a strided
                                   version of the PCPH harmonic spectrum )
        EX — Complex excitation  ( iSTFT-ready spectral output )

    Returns
    -------
    spec  : (B, gen_istft_n_fft // 2 + 1, T_stft)   log-magnitude via exp()
    phase : (B, gen_istft_n_fft // 2 + 1, T_stft)   wrapped phase via sin()
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
        # Excitation synthesizer  (PCPH, replaces hn-nsf / SineGen)
        # ------------------------------------------------------------------
        self.excitation_synthesizer = ExcitationSynthesizer(
            sample_rate=sr,
            hop_length=self.full_hop,       # upsample f0 to full audio length
            random_init_phase=False,
            power_factor=0.1,
            add_noise_std=0.003,
            geosaw_r=0.90,
            noise_shaper_duration_ms=0.65,
        )


        # ------------------------------------------------------------------
        # Pre-conv
        # ------------------------------------------------------------------
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))

        # ------------------------------------------------------------------
        # Upsampler stages, per-stage spectral injection, main resblocks
        # ------------------------------------------------------------------
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()   # spectral excitation injectors
        self.noise_res = nn.ModuleList()     # refinement resblocks for injected exc
        self.resblocks = nn.ModuleList()

        # har has gen_istft_n_fft+2 channels (spec + phase concatenated)
        har_channels = gen_istft_n_fft + 2

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch  = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))

            # Upsampler
            self.ups.append(weight_norm(
                ConvTranspose1d(in_ch, out_ch, k, u, padding=(k - u) // 2)
            ))

            # Spectral excitation injection:
            if i + 1 < self.num_upsamples:
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(
                    har_channels, out_ch,
                    kernel_size=stride_f0 * 2,
                    stride=stride_f0,
                    padding=(stride_f0 + 1) // 2,
                ))
                # Mid-stage injection: use a 7-kernel resblock (same as HiFTNet)
                self.noise_res.append(ResBlock(out_ch, kernel_size=7, dilations=(1, 3, 5)))
            else:
                # Last stage: har and x are already at the same resolution
                self.noise_convs.append(Conv1d(har_channels, out_ch, kernel_size=1))
                # Final injection: use an 11-kernel resblock (same as HiFTNet)
                self.noise_res.append(ResBlock(out_ch, kernel_size=11, dilations=(1, 3, 5)))

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        spec  : (B, gen_istft_n_fft // 2 + 1, T_stft)
        phase : (B, gen_istft_n_fft // 2 + 1, T_stft)

        The caller is responsible for iSTFT reconstruction.
        """
        # ---- Prepare frame-level f0 → (B, 1, T_frames) ------------------
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)    # (B, 1, T_frames)


        # ---- Generate PCPH excitation waveform ---------------------------
        excitation = self.excitation_synthesizer(f0)
        # excitation: (B, 1, T_audio)

        # ---- STFT of excitation → spectral representation ----------------
        har_spec, har_phase = self.stft.transform(excitation.squeeze(1).float())

        har = torch.cat([har_spec, har_phase], dim=1)   # (B, n_fft+2, T_stft)
        har = har.to(x.dtype)

        # ---- Pre-conv + speaker conditioning -----------------------------
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        # ---- Upsample stages with spectral injection ---------------------
        for i in range(self.num_upsamples):
            #x = F.silu(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.ups[i](x)

            # Reflection-pad before the last upsampler
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # Inject spectral excitation
            x_source = self.noise_convs[i](har)         # strided to match x
            x_source = self.noise_res[i](x_source)      # refine
            x = x + x_source

            # Main multi-kernel resblocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # ---- Post-conv -------------------------
        #x = F.silu(x)
        x = F.leaky_relu(x, negative_slope=0.01)

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
        for m in self.noise_res:
            m.remove_weight_norm()
        # noise_convs are plain Conv1d (no weight_norm), nothing to strip

    def __prepare_scriptable__(self):
        _remove_wn_if_present(self.conv_pre)
        _remove_wn_if_present(self.conv_post)
        for m in self.ups:
            _remove_wn_if_present(m)
        for rb in chain(self.resblocks, self.noise_res):
            for conv in chain(rb.convs1, rb.convs2):
                _remove_wn_if_present(conv)
        return self
