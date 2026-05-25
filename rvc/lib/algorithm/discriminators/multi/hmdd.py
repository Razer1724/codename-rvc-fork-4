import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

from rvc.lib.algorithm.discriminators.multi.pqmf import PQMF

# =============================================================================
# Hardcoded discriminator config
# =============================================================================

_PQMF_LV2  = (4,  192, 0.13, 10.0)   # CoMBD hierarchy: 4-band lowpass
_PQMF_LV1  = (2,  256, 0.25, 10.0)   # CoMBD hierarchy: 2-band lowpass
_PQMF_SBD  = (16, 256, 0.03, 10.0)   # SBD standard bands
_PQMF_FSBD = (64, 256, 0.10,  9.0)   # SBD transposed (frequency) bands

_COMBD_H_U = [
    [16, 64, 256, 1024, 1024, 1024],
    [16, 64, 256, 1024, 1024, 1024],
    [16, 64, 256, 1024, 1024, 1024],
]
_COMBD_D_K = [
    [ 7, 11, 11, 11, 11, 5],
    [11, 21, 21, 21, 21, 5],
    [15, 41, 41, 41, 41, 5],
]
_COMBD_D_S = [
    [1, 1, 4, 4, 4, 1],
    [1, 1, 4, 4, 4, 1],
    [1, 1, 4, 4, 4, 1],
]
_COMBD_D_D = [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
]
_COMBD_D_G = [
    [  1,   4,  16,  64, 256, 1],
    [  1,   4,  16,  64, 256, 1],
    [  1,   4,  16,  64, 256, 1],
]
_COMBD_D_P = [
    [ 3,  5,  5,  5,  5, 2],
    [ 5, 10, 10, 10, 10, 2],
    [ 7, 20, 20, 20, 20, 2],
]
_COMBD_OP_F = [1, 1, 1]
_COMBD_OP_K = [3, 3, 3]
_COMBD_OP_G = [1, 1, 1]

_SBD_FILTERS = [
    [ 64, 128, 256, 256, 256],
    [ 64, 128, 256, 256, 256],
    [ 64, 128, 256, 256, 256],
    [ 32,  64, 128, 128, 128],
]
_SBD_STRIDES = [
    [1, 1, 3, 3, 1],
    [1, 1, 3, 3, 1],
    [1, 1, 3, 3, 1],
    [1, 1, 3, 3, 1],
]
_SBD_KERNEL_SIZES = [
    [[7,7,7],[7,7,7],[7,7,7],[7,7,7],[7,7,7]],
    [[5,5,5],[5,5,5],[5,5,5],[5,5,5],[5,5,5]],
    [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
    [[5,5,5],[5,5,5],[5,5,5],[5,5,5],[5,5,5]],
]
_SBD_DILATIONS = [
    [[5,7,11],[5,7,11],[5,7,11],[5,7,11],[5,7,11]],
    [[3,5, 7],[3,5, 7],[3,5, 7],[3,5, 7],[3,5, 7]],
    [[1,2, 3],[1,2, 3],[1,2, 3],[1,2, 3],[1,2, 3]],
    [[1,2, 3],[1,2, 3],[1,2, 3],[2,3, 5],[2,3, 5]],
]
_SBD_BAND_RANGES = [[0, 6], [0, 11], [0, 16], [0, 64]]
_SBD_TRANSPOSE   = [False, False, False, True]



class HolisticMultiDomainDiscriminator(nn.Module):
    """
    Holistic discriminator framework

    CoMBD + SBD + UnivHD + GLD

      CoMBD:    collaborative multi-band: evaluates generator intermediate outputs at each resolution against real audio at the matching rate.
      SBD:      sub-band: evaluates the final full-res output in PQMF sub-bands.
      UnivHD:   harmonic-aware dynamic spectral resolution on full-res output.
      GLD:      perceptual gammatone filterbank discriminator; specific loudness, AM envelopes, and spectral tilt on the full-res output.

    forward(y, y_hat_list) -> (y_d_rs, y_d_gs, fmap_rs, fmap_gs)

    y : [B, 1, T] full-res real waveform
    y_hat_list : List[Tensor] of 3 fakes (return_intermediates=True) [y_hat_lv2, y_hat_lv1, y_hat_full]
    """

    def __init__(
        self,
        sample_rate: int,
        segment_size_samples: int,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        # PQMF banks for preparing real audio hierarchy for CoMBD.
        # pqmf_lv2: 4-band -> subband 0 = 1/4-rate lowpass (matches gen stage lv2)
        # pqmf_lv1: 2-band -> subband 0 = 1/2-rate lowpass (matches gen stage lv1)
        self._pqmf_lv2 = PQMF(*_PQMF_LV2)
        self._pqmf_lv1 = PQMF(*_PQMF_LV1)

        self.combd = CoMBD(use_spectral_norm=use_spectral_norm)
        self.sbd = SBD(segment_size_samples=segment_size_samples, use_spectral_norm=use_spectral_norm)
        self.univhd = UnivHD(sample_rate=sample_rate)
        self.gld = GammatoneLoudnessDiscriminator(sample_rate=sample_rate)

        self.discriminators = nn.ModuleList([self.combd, self.sbd, self.univhd, self.gld])

    def forward(
        self,
        y: torch.Tensor,
        y_hat_list: List[torch.Tensor],
    ) -> Tuple[List, List, List, List]:
        y_hat_full = y_hat_list[-1]

        # Prepare real audio hierarchy for CoMBD (matches ALPEX-GAN intermediate resolutions)
        y_lv2 = self._pqmf_lv2.analysis(y)[:, :1, :]
        y_lv1 = self._pqmf_lv1.analysis(y)[:, :1, :]
        ys = [y_lv2, y_lv1, y]

        # CoMBD: hierarchical multi-band (all 3 resolutions)
        combd_rs, combd_gs, combd_frs, combd_fgs = self.combd(ys, y_hat_list)

        # SBD: sub-band on full-res only
        sbd_rs, sbd_gs, sbd_frs, sbd_fgs = self.sbd(y, y_hat_full)

        # UnivHD: harmonic discriminator on full-res only
        univhd_r, fmap_univhd_r = self.univhd(y)
        univhd_g, fmap_univhd_g = self.univhd(y_hat_full)

        # GLD: perceptual gammatone loudness discriminator on full-res only
        gld_rs, gld_gs, gld_frs, gld_fgs = self.gld(y, y_hat_list)

        return (
            combd_rs + sbd_rs + [univhd_r] + gld_rs,
            combd_gs + sbd_gs + [univhd_g] + gld_gs,
            combd_frs + sbd_frs + [fmap_univhd_r] + gld_frs,
            combd_fgs + sbd_fgs + [fmap_univhd_g] + gld_fgs,
        )


# =============================================================================
# CoMBD - Collaborative Multi-Band Discriminator
# =============================================================================

class CoMBDBlock(nn.Module):
    def __init__(self, h_u, d_k, d_s, d_d, d_g, d_p, op_f, op_k, op_g, use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        filters = [[1, h_u[0]]]
        for i in range(len(h_u) - 1):
            filters.append([h_u[i], h_u[i + 1]])

        self.convs = nn.ModuleList()
        for _f, _k, _s, _d, _g, _p in zip(filters, d_k, d_s, d_d, d_g, d_p):
            self.convs.append(norm_f(Conv1d(
                in_channels=_f[0], out_channels=_f[1],
                kernel_size=_k, stride=_s, dilation=_d, groups=_g, padding=_p,
            )))

        self.projection_conv = norm_f(Conv1d(
            in_channels=filters[-1][1], out_channels=op_f,
            kernel_size=op_k, groups=op_g,
        ))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            fmap.append(x)
        x = self.projection_conv(x)
        return x, fmap


class CoMBD(nn.Module):
    """
    Expects:
        ys     : [y_lv2, y_lv1, y_full]  - real audio at each resolution
        ys_hat : [y_hat_lv2, y_hat_lv1, y_hat_full]  - fake at each resolution

    Internally also runs blocks[0] and blocks[1] against PQMF sub-bands of
    the full-resolution signal (the Avocodo "multi-scale" path).
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()

        # PQMF banks: same params used for real-audio multi-scale path
        self.pqmf = nn.ModuleList([
            PQMF(*_PQMF_LV2),
            PQMF(*_PQMF_LV1),
        ])

        self.blocks = nn.ModuleList()
        for h_u, d_k, d_s, d_d, d_g, d_p, op_f, op_k, op_g in zip(
            _COMBD_H_U, _COMBD_D_K, _COMBD_D_S,
            _COMBD_D_D, _COMBD_D_G, _COMBD_D_P,
            _COMBD_OP_F, _COMBD_OP_K, _COMBD_OP_G,
        ):
            self.blocks.append(CoMBDBlock(
                h_u, d_k, d_s, d_d, d_g, d_p, op_f, op_k, op_g,
                use_spectral_norm=use_spectral_norm,
            ))

    def _block_forward(self, inputs, blocks, outs, fmaps):
        for x, block in zip(inputs, blocks):
            out, fmap = block(x)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps

    def forward(self, ys, ys_hat):
        y_full     = ys[-1]
        y_hat_full = ys_hat[-1]

        multi_real = [pqmf.analysis(y_full)[:, :1, :]     for pqmf in self.pqmf]
        multi_fake = [pqmf.analysis(y_hat_full)[:, :1, :] for pqmf in self.pqmf]

        outs_real, fmaps_real = [], []
        outs_fake, fmaps_fake = [], []

        # Hierarchical path
        outs_real, fmaps_real = self._block_forward(ys,     self.blocks,       outs_real, fmaps_real)
        outs_fake, fmaps_fake = self._block_forward(ys_hat, self.blocks,       outs_fake, fmaps_fake)
        # Multi-scale path
        outs_real, fmaps_real = self._block_forward(multi_real, self.blocks[:-1], outs_real, fmaps_real)
        outs_fake, fmaps_fake = self._block_forward(multi_fake, self.blocks[:-1], outs_fake, fmaps_fake)

        return outs_real, outs_fake, fmaps_real, fmaps_fake


# =============================================================================
# SBD - Sub-Band Discriminator
# =============================================================================

def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class MDC(nn.Module):
    """Multi-Dilated Conv: parallel dilated branches -> sum -> stride conv."""

    def __init__(self, in_channels, out_channels, strides, kernel_size, dilations, use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.d_convs = nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(norm_f(Conv1d(in_channels, out_channels, _k, dilation=_d, padding=_get_padding(_k, _d))))
        # NOTE: padding uses last _k, _d — preserved from original avocodo (must be a loop, not a comprehension)
        self.post_conv = norm_f(Conv1d(out_channels, out_channels, 3, stride=strides, padding=_get_padding(_k, _d)))

    def forward(self, x):
        _out = None
        for conv in self.d_convs:
            _x = F.leaky_relu(conv(x).unsqueeze(-1), 0.2)
            _out = _x if _out is None else torch.cat([_out, _x], dim=-1)
        return F.leaky_relu(self.post_conv(torch.sum(_out, dim=-1)), 0.2)


class SBDBlock(nn.Module):
    def __init__(self, segment_dim, strides, filters, kernel_size, dilations, use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append((filters[i], filters[i + 1]))

        self.convs = nn.ModuleList([
            MDC(in_channels=_f[0], out_channels=_f[1], strides=_s,
                kernel_size=_k, dilations=_d, use_spectral_norm=use_spectral_norm)
            for _s, _f, _k, _d in zip(strides, filters_in_out, kernel_size, dilations)
        ])
        self.post_conv = norm_f(Conv1d(filters[-1], 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            fmap.append(x)
        return self.post_conv(x), fmap


class SBD(nn.Module):
    """
    Sub-Band Discriminator.
    Splits input into PQMF sub-bands and runs SBDBlock on each frequency band.
    The transposed discriminator's segment_dim scales with audio segment length.
    """

    def __init__(self, segment_size_samples: int, use_spectral_norm=False):
        super().__init__()

        self.pqmf   = PQMF(*_PQMF_SBD)
        self.f_pqmf = PQMF(*_PQMF_FSBD)

        self.band_ranges = _SBD_BAND_RANGES
        self.transpose   = _SBD_TRANSPOSE

        self.discriminators = nn.ModuleList()
        for _f, _k, _d, _s, br, tr in zip(
            _SBD_FILTERS, _SBD_KERNEL_SIZES,
            _SBD_DILATIONS, _SBD_STRIDES,
            _SBD_BAND_RANGES, _SBD_TRANSPOSE,
        ):
            segment_dim = (segment_size_samples // br[1]) - br[0] if tr else (br[1] - br[0])
            self.discriminators.append(SBDBlock(
                segment_dim=segment_dim, filters=_f, kernel_size=_k,
                dilations=_d, strides=_s, use_spectral_norm=use_spectral_norm,
            ))

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        y_sub       = self.pqmf.analysis(y)
        y_hat_sub   = self.pqmf.analysis(y_hat)
        y_sub_f     = self.f_pqmf.analysis(y)
        y_hat_sub_f = self.f_pqmf.analysis(y_hat)

        for d, br, tr in zip(self.discriminators, self.band_ranges, self.transpose):
            if tr:
                _y     = torch.transpose(y_sub_f[:, br[0]:br[1], :],     1, 2)
                _y_hat = torch.transpose(y_hat_sub_f[:, br[0]:br[1], :], 1, 2)
            else:
                _y     = y_sub[:, br[0]:br[1], :]
                _y_hat = y_hat_sub[:, br[0]:br[1], :]
            y_d_r, fmap_r = d(_y)
            y_d_g, fmap_g = d(_y_hat)
            y_d_rs.append(y_d_r);  fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g);  fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# =============================================================================
# UnivHD - Universal Harmonic Discriminator
# Identical to the version in mpd_msd_mrd_univhd_combined.py
# =============================================================================

class UnivHD(nn.Module):
    _N_MDC:   int = 3
    _HCB_OUT: int = 32
    _MDC_OUT: int = 32

    def __init__(self, sample_rate, n_fft=None, hop_length=None, win_length=None,
                 n_harmonics=10, bins_per_octave=24, fmin=32.7,
                 add_half_harmonic=True, lrelu_slope=0.1):
        super().__init__()

        n_fft      = n_fft      if n_fft      is not None else _derive_n_fft(sample_rate)
        hop_length = hop_length if hop_length is not None else _derive_hop(sample_rate)
        win_length = win_length if win_length is not None else n_fft

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

        self.harmonic_filter = HarmonicFilter(
            sample_rate=sample_rate, n_fft=n_fft, n_harmonics=n_harmonics,
            bins_per_octave=bins_per_octave, fmin=fmin, add_half_harmonic=add_half_harmonic,
        )
        n_total = self.harmonic_filter.n_total

        self.hcb = HybridConvBlock(in_channels=n_total, out_channels=self._HCB_OUT)
        self.mdc_blocks = nn.ModuleList([
            _MDC_UnivHD(
                in_channels  = self._HCB_OUT if i == 0 else self._MDC_OUT,
                out_channels = self._MDC_OUT,
                lrelu_slope  = lrelu_slope,
            )
            for i in range(self._N_MDC)
        ])
        freq_kernel = _freq_after_mdc(self.harmonic_filter.n_bins)
        self.final_conv = weight_norm(nn.Conv2d(self._MDC_OUT, 1, kernel_size=(freq_kernel, 1)))

    def _stft_magnitude(self, x):
        return torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True
        ).abs()

    def forward(self, waveform):
        x = self.harmonic_filter(self._stft_magnitude(waveform))
        feat_maps = []
        x = self.hcb(x)
        for mdc in self.mdc_blocks:
            x = mdc(x)
            feat_maps.append(x)
        return self.final_conv(x).squeeze(1).squeeze(1), feat_maps


class _MDC_UnivHD(nn.Module):
    """MDC used internally by UnivHD (renamed to avoid collision with SBD's MDC)."""

    def __init__(self, in_channels, out_channels=32, kernel_size=5,
                 dilation_rates=(1, 2, 4), lrelu_slope=0.1):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        k = kernel_size
        layers, ch = [], in_channels
        for d in dilation_rates:
            layers.append(weight_norm(nn.Conv2d(ch, out_channels, (k, k), stride=(1, 1),
                                                dilation=(d, 1), padding=(d*(k-1)//2, (k-1)//2))))
            ch = out_channels
        self.dilated_convs = nn.ModuleList(layers)
        self.final_conv = weight_norm(nn.Conv2d(out_channels, out_channels, (k, k),
                                                stride=(2, 1), padding=((k-1)//2, (k-1)//2)))

    def forward(self, x):
        for conv in self.dilated_convs:
            x = conv(x)
        return self.final_conv(F.leaky_relu(x, self.lrelu_slope))


class HybridConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_size=(7, 7)):
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.ds_conv     = weight_norm(nn.Conv2d(in_channels, in_channels, kernel_size, padding=pad, groups=in_channels))
        self.p_conv      = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.normal_conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad))

    def forward(self, x):
        return self.p_conv(self.ds_conv(x)) + self.normal_conv(x)


class HarmonicFilter(nn.Module):
    def __init__(self, sample_rate, n_fft, n_harmonics=10, bins_per_octave=24,
                 fmin=32.7, add_half_harmonic=True):
        super().__init__()
        self.n_fft  = n_fft
        self.n_bins = _compute_n_bins(sample_rate, n_harmonics, bins_per_octave, fmin)

        k  = torch.arange(self.n_bins, dtype=torch.float32)
        fc = fmin * torch.pow(2.0, k / bins_per_octave)
        self.register_buffer("fc", fc)

        stft_freqs = torch.arange(n_fft // 2 + 1, dtype=torch.float32) * (sample_rate / n_fft)
        self.register_buffer("stft_freqs", stft_freqs)

        orders = ([0.5] if add_half_harmonic else []) + [float(h) for h in range(1, n_harmonics + 1)]
        self.n_total = len(orders)
        self.register_buffer("harmonic_orders", torch.tensor(orders, dtype=torch.float32))

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, stft_mag):
        gamma = self.gamma
        h_fc  = self.harmonic_orders.unsqueeze(1) * self.fc.unsqueeze(0)
        h_bw  = (0.1079 * h_fc + 24.7) / gamma.unsqueeze(1)
        diff  = (self.stft_freqs.unsqueeze(0).unsqueeze(0) - h_fc.unsqueeze(2)).abs()
        filter_bank = F.relu(1.0 - 2.0 * diff / h_bw.unsqueeze(2))
        return torch.einsum("hfn,bnt->bhft", filter_bank, stft_mag)


# Helper functions for UnivHD

def _next_pow2(x: float) -> int:
    return 2 ** math.ceil(math.log2(x))

def _derive_n_fft(sample_rate: int) -> int:
    return _next_pow2(1024 * sample_rate / 24_000)

def _derive_hop(sample_rate: int) -> int:
    return round(256 * sample_rate / 24_000)

def _compute_n_bins(sample_rate: int, n_harmonics: int, bins_per_octave: int, fmin: float) -> int:
    fmax_first = sample_rate / (2.0 * n_harmonics)
    return int(math.floor(bins_per_octave * math.log2(fmax_first / fmin)))

def _freq_after_mdc(f: int, n_mdc: int = 3, k: int = 5, stride: int = 2, pad: int = 2) -> int:
    for _ in range(n_mdc):
        f = math.floor((f + 2 * pad - k) / stride) + 1
    return f


# =============================================================================
# GLD - Gammatone Loudness Discriminator v3
# =============================================================================

def _gld_erb_space(fmin: float, fmax: float, n: int) -> np.ndarray:
    """
    Return n centre frequencies spaced uniformly on the ERB-rate scale.
    Moore & Glasberg (1990):
      ERB(f) = 24.7 · (4.37 · f/1000 + 1)       [Hz]
      E(f)   = 21.4 · log10(4.37 · f/1000 + 1)  [Cams]
    Inverse: f = (10^(E/21.4) − 1) / 4.37 · 1000
    """
    e_min = 21.4 * math.log10(4.37 * fmin / 1000.0 + 1.0)
    e_max = 21.4 * math.log10(4.37 * fmax / 1000.0 + 1.0)
    erb_rate = np.linspace(e_min, e_max, n)
    return (10.0 ** (erb_rate / 21.4) - 1.0) / 4.37 * 1000.0


def _gld_build_gammatone_kernels(
    cfs: np.ndarray,
    sample_rate: int,
    order: int = 4,
    filter_len: int = 1024,
    n_fft: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build analytic (complex) gammatone FIR kernels (Patterson et al. 1988)."""
    n_filt = len(cfs)
    t      = np.arange(filter_len, dtype=np.float64) / sample_rate

    erb = 24.7 * (4.37 * cfs / 1000.0 + 1.0)
    b   = (2.0 * math.pi * 1.019 * erb)[:, np.newaxis]

    env  = (t ** (order - 1)) * np.exp(-b * t)
    phi  = 2.0 * math.pi * cfs[:, np.newaxis] * t

    h_re = (env * np.cos(phi)).astype(np.float32)
    h_im = (env * np.sin(phi)).astype(np.float32)

    fft_sz  = int(2 ** math.ceil(math.log2(max(8192, 4 * filter_len))))
    H_cplx  = np.fft.fft(h_re.astype(np.float64) + 1j * h_im.astype(np.float64),
                         n=fft_sz, axis=-1)
    peaks   = np.max(np.abs(H_cplx), axis=-1, keepdims=True) + 1e-12

    k_re = (h_re / peaks).astype(np.float32)
    k_im = (h_im / peaks).astype(np.float32)

    k_re_t = torch.from_numpy(k_re).unsqueeze(1)
    k_im_t = torch.from_numpy(k_im).unsqueeze(1)

    if n_fft is None:
        n_fft = 2 ** math.ceil(math.log2(sample_rate + filter_len))

    k_re_fft = (torch.fft.rfft(k_re_t, n=n_fft, dim=-1)
                .squeeze(1).unsqueeze(0).to(torch.complex64))
    k_im_fft = (torch.fft.rfft(k_im_t, n=n_fft, dim=-1)
                .squeeze(1).unsqueeze(0).to(torch.complex64))

    return k_re_t, k_im_t, k_re_fft, k_im_fft


class _GLD_DSConv2d(nn.Module):
    """Depthwise-Separable Conv2d (Howard et al. 2017 – MobileNet)."""
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        st = stride      if isinstance(stride,       tuple) else (stride, stride)
        pa = padding     if isinstance(padding,      tuple) else (padding, padding)
        di = dilation    if isinstance(dilation,     tuple) else (dilation, dilation)
        self.dw = weight_norm(nn.Conv2d(in_c, in_c, ks, stride=st, padding=pa,
                                        dilation=di, groups=in_c))
        self.pw = weight_norm(nn.Conv2d(in_c, out_c, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class _GLD_SEBlock2d(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al. 2018 – SE-Net)."""
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        mid = max(channels // r, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(-2, -1))
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


class _GLD_ResBlock2d(nn.Module):
    """Residual block: DS-conv → LReLU → DS-conv → SE → add skip."""
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0,
                 dilation=1, lrelu_slope: float = 0.1, se_r: int = 8):
        super().__init__()
        self.slope = lrelu_slope
        self.conv1 = _GLD_DSConv2d(in_c,  out_c, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
        self.conv2 = _GLD_DSConv2d(out_c, out_c, (3, 3), padding=(1, 1))
        self.se    = _GLD_SEBlock2d(out_c, r=se_r)

        need_proj = (in_c != out_c) or (
            (stride if isinstance(stride, int) else stride[0]) != 1 or
            (stride if isinstance(stride, int) else stride[1]) != 1
        )
        self.proj = (weight_norm(nn.Conv2d(in_c, out_c, 1, stride=stride))
                     if need_proj else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = F.leaky_relu(self.conv1(x), self.slope)
        h    = self.conv2(h)
        h    = self.se(h)
        skip = self.proj(x) if self.proj is not None else x
        return F.leaky_relu(h + skip, self.slope)


class _GLD_AMHead(nn.Module):
    """
    Amplitude-Modulation temporal head.
    Projects 64 ERB bands → 8 coarse groups via a learned Conv1d (instead of
    hard reshape+mean, which destroys within-group amplitude relationships),
    then runs a dilated 1D Conv stack on the fine-hop temporal power envelope.
    Dilation schedule [1,2,4,8] → 61-frame receptive field at 5 ms hop.
    """
    def __init__(self, n_filters: int = 64, n_band_groups: int = 8, channels: int = 32, lrelu_slope: float = 0.1):
        super().__init__()
        self.slope    = lrelu_slope
        TC            = channels
        kernel_size   = 5
        dilations     = [1, 2, 4, 8]

        # Learned band projection: preserves inter-band amplitude structure
        self.band_proj = weight_norm(nn.Conv1d(n_filters, n_band_groups, kernel_size=1))

        self.convs = nn.ModuleList()
        in_c = n_band_groups
        for d in dilations:
            pad = (kernel_size - 1) * d // 2
            self.convs.append(
                weight_norm(nn.Conv1d(in_c, TC, kernel_size, dilation=d, padding=pad))
            )
            in_c = TC

        self.head = weight_norm(nn.Conv1d(TC, 1, 1))

    def forward(
        self,
        power_fine: torch.Tensor,   # [B, 64, n_frames_fine]  raw mean power at fine hop
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Compress to log-loudness
        log_loud = torch.log((power_fine + 1e-8).pow(0.23) + 1e-5)

        # Learned projection: 64 ERB bands → n_band_groups  [B, n_groups, n_f]
        x = F.leaky_relu(self.band_proj(log_loud), self.slope)

        fmaps: List[torch.Tensor] = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), self.slope)
            fmaps.append(x)

        logit = self.head(x).squeeze(1).mean(dim=-1, keepdim=True)  # [B, 1]
        return logit, fmaps


class _GLD_ModSpecHead(nn.Module):
    """
    Modulation spectrum discriminator.

    Takes the fine-hop ERB-band power envelopes [B, 64, n_f], groups them,
    computes a short-time amplitude spectrum along the time axis (modulation
    frequencies ~0-100 Hz at 5 ms hop with mod_fft=64), then runs a small
    Conv2d stack.  This captures roughness, breathiness, and vibrato — all of
    which live in the 2-50 Hz modulation band and are absent everywhere else in
    the discriminator ensemble.
    """
    def __init__(
        self,
        n_filters:    int   = 64,
        n_groups:     int   = 8,
        mod_fft:      int   = 64,
        channels:     int   = 32,
        lrelu_slope:  float = 0.1,
    ):
        super().__init__()
        self.n_groups   = n_groups
        self.mod_fft    = mod_fft
        self.mod_hop    = mod_fft // 2
        self.slope      = lrelu_slope
        n_mod_bins      = mod_fft // 2 + 1  # 33 bins

        # Learned ERB grouping (same philosophy as AM head fix)
        self.band_proj = weight_norm(nn.Conv1d(n_filters, n_groups, kernel_size=1))

        self.convs = nn.Sequential(
            weight_norm(nn.Conv2d(n_groups, channels, kernel_size=(3, 3), padding=(1, 1))),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            weight_norm(nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))),
            nn.LeakyReLU(lrelu_slope, inplace=True),
        )
        self.head = weight_norm(nn.Conv2d(channels, 1, kernel_size=(1, 1)))

    def forward(
        self,
        power_fine: torch.Tensor,   # [B, 64, n_f]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        log_loud = torch.log((power_fine + 1e-8).pow(0.23) + 1e-5)

        # Learned band grouping  [B, n_groups, n_f]
        x = F.leaky_relu(self.band_proj(log_loud), self.slope)

        B, G, n_f = x.shape
        # Need at least one full window; silently skip if clip too short
        if n_f < self.mod_fft:
            dummy = x.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, G, 1, 1]
            feat = self.convs(dummy.expand(B, G, 1, 1))
            logit = self.head(feat).flatten(1).mean(-1, keepdim=True)
            return logit, [feat]

        # Short-time spectrum along time axis

        # frames  = x.unfold(-1, self.mod_fft, self.mod_hop)   # [B, G, n_frames, mod_fft]
        # window  = torch.hann_window(self.mod_fft, device=x.device, dtype=x.dtype)
        # frames  = frames * window
        # mod_spec = torch.fft.rfft(frames, dim=-1).abs()       # [B, G, n_frames, 33]
        # mod_spec = torch.log(mod_spec + 1e-5)
        # mod_spec = mod_spec.permute(0, 1, 3, 2)               # [B, G, 33, n_frames]

        frames  = x.unfold(-1, self.mod_fft, self.mod_hop).to(torch.float32)
        window  = torch.hann_window(self.mod_fft, device=x.device, dtype=torch.float32)
        frames  = frames * window
        mod_spec = torch.fft.rfft(frames, dim=-1).abs()       # [B, G, n_frames, 33]
        mod_spec = torch.log(mod_spec + 1e-5).to(x.dtype)
        mod_spec = mod_spec.permute(0, 1, 3, 2)               # [B, G, 33, n_frames]

        feat  = self.convs(mod_spec)
        logit = self.head(feat).flatten(1).mean(-1, keepdim=True)  # [B, 1]
        return logit, [feat]


class _GLD_FluxHead(nn.Module):
    """
    Spectral flux discriminator.

    Computes frame-to-frame magnitude change in the ERB-loudness domain,
    averaged across bands → a 1D temporal signal that directly reflects
    transient sharpness and onset/offset clarity.  A small dilated Conv1d
    stack then scores the flux envelope.

    Penalises smeared transients and over-smoothed attacks — common vocoder
    artifacts that the backbone's static spectral view cannot detect.
    """
    def __init__(self, channels: int = 32, lrelu_slope: float = 0.1):
        super().__init__()
        self.slope = lrelu_slope
        TC = channels
        self.convs = nn.Sequential(
            weight_norm(nn.Conv1d(1,  TC, kernel_size=7, padding=3)),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            weight_norm(nn.Conv1d(TC, TC, kernel_size=5, dilation=2, padding=4)),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            weight_norm(nn.Conv1d(TC, TC, kernel_size=3, dilation=4, padding=4)),
            nn.LeakyReLU(lrelu_slope, inplace=True),
        )
        self.head = weight_norm(nn.Conv1d(TC, 1, kernel_size=1))

    def forward(
        self,
        log_loud_c: torch.Tensor,   # [B, 64, n_c]  ERB log-loudness at coarse hop
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Mean spectral flux over ERB bands per frame  [B, 1, n_c-1]
        flux = (log_loud_c[:, :, 1:] - log_loud_c[:, :, :-1]).abs().mean(dim=1, keepdim=True)
        feat = self.convs(flux)
        logit = self.head(feat).squeeze(1).mean(dim=-1, keepdim=True)  # [B, 1]
        return logit, [feat]


class GammatoneLoudnessDiscriminator(nn.Module):
    """
    Perceptual discriminator v4: ERB filterbank → loudness + AM + modulation spectrum
    + spectral flux + dynamic spectral tilt.

    Heads
    -----
    backbone   : 2D ResBlock stack on log-ERB-loudness [freq × time]
    am_head    : learned-grouped dilated Conv1d on fine-hop AM envelopes
    mod_spec   : short-time modulation spectrum (roughness/breathiness/vibrato)
    flux_head  : frame-to-frame spectral flux (transient sharpness)
    tilt_head  : dynamic Conv2d spectral tilt (temporal + spectral context)

    Parameters
    ----------
    sample_rate      : int
    n_filters        : int    — ERB channel count (default 64)
    fmin             : float  — lowest ERB centre frequency in Hz (default 80)
    order            : int    — gammatone filter order; 4 is standard
    filter_ms        : float  — FIR kernel duration in ms (default 30)
    hop_ms           : float  — main loudness frame hop in ms (default 10)
    frame_ms         : float  — main loudness frame size in ms (default 20)
    fine_hop_ms      : float  — fine hop for AM / mod-spec heads in ms (default 5)
    loudness_exp     : float  — Stevens power-law exponent ≈ 0.23 for sones
    channels         : int    — backbone channel width (default 128)
    am_channels      : int    — AM head internal channels (default 32)
    am_groups        : int    — learned ERB band groups for AM / mod-spec (default 8)
    mod_fft          : int    — modulation FFT window length in frames (default 64)
    flux_channels    : int    — spectral flux head channels (default 32)
    tilt_channels    : int    — dynamic tilt head channels (default 32)
    se_r             : int    — SE reduction ratio (default 8)
    lrelu_slope      : float
    power_floor      : float
    log_floor        : float
    max_clip_ms      : float  — hard upper bound on input clip duration in ms (default 500)
    """

    def __init__(
        self,
        sample_rate:     int,
        n_filters:       int   = 64,
        fmin:            float = 80.0,
        order:           int   = 4,
        filter_ms:       float = 30.0,
        hop_ms:          float = 10.0,
        frame_ms:        float = 20.0,
        fine_hop_ms:     float = 5.0,
        loudness_exp:    float = 0.23,
        channels:        int   = 128,
        am_channels:     int   = 32,
        am_groups:       int   = 8,
        mod_fft:         int   = 64,
        flux_channels:   int   = 32,
        tilt_channels:   int   = 32,
        se_r:            int   = 8,
        lrelu_slope:     float = 0.1,
        power_floor:     float = 1e-8,
        log_floor:       float = 1e-5,
        max_clip_ms:     float = 500.0,
    ):
        super().__init__()

        self.n_filters    = n_filters
        self.loudness_exp = loudness_exp
        self.lrelu_slope  = lrelu_slope
        self.power_floor  = power_floor
        self.log_floor    = log_floor

        self.hop_samples      = round(sample_rate * hop_ms      / 1000.0)
        self.frame_samples    = round(sample_rate * frame_ms    / 1000.0)
        self.fine_hop_samples = round(sample_rate * fine_hop_ms / 1000.0)

        # ── frozen gammatone filterbank ────────────────────────────────────
        fmax       = sample_rate * 0.45
        cfs        = _gld_erb_space(fmin, fmax, n_filters)
        filter_len = max(256, round(sample_rate * filter_ms / 1000.0))
        self.filter_len = filter_len

        max_clip_samples  = round(sample_rate * max_clip_ms / 1000.0)
        N_min_for_max     = max_clip_samples + 2 * filter_len - 2
        N_canonical       = 2 ** math.ceil(math.log2(N_min_for_max))
        self._N_canonical = N_canonical

        k_re, k_im, W_re_fft, W_im_fft = _gld_build_gammatone_kernels(
            cfs, sample_rate, order, filter_len, n_fft=N_canonical)

        self.register_buffer("kernels_re",   k_re)
        self.register_buffer("kernels_im",   k_im)
        self.register_buffer("W_re_fft",     W_re_fft)
        self.register_buffer("W_im_fft",     W_im_fft)
        self.register_buffer("centre_freqs", torch.from_numpy(cfs.astype(np.float32)))

        # ── Conv2d backbone ────────────────────────────────────────────────
        C = channels
        self.backbone = nn.ModuleList([
            _GLD_ResBlock2d(1,    C,    (5, 3), stride=(2, 1), padding=(2, 1),
                            lrelu_slope=lrelu_slope, se_r=se_r),
            _GLD_ResBlock2d(C,    C*2,  (5, 3), stride=(2, 1), padding=(2, 1),
                            lrelu_slope=lrelu_slope, se_r=se_r),
            _GLD_ResBlock2d(C*2,  C*2,  (3, 5), stride=(1, 1),
                            padding=(1, 4), dilation=(1, 2),
                            lrelu_slope=lrelu_slope, se_r=se_r),
            _GLD_ResBlock2d(C*2,  C*2,  (3, 5), stride=(1, 1),
                            padding=(1, 8), dilation=(1, 4),
                            lrelu_slope=lrelu_slope, se_r=se_r),
        ])
        self.head_main = weight_norm(nn.Conv2d(C * 2, 1, kernel_size=(1, 1)))

        # ── AM temporal head (learned band grouping) ───────────────────────
        self.am_head = _GLD_AMHead(
            n_filters=n_filters,
            n_band_groups=am_groups,
            channels=am_channels,
            lrelu_slope=lrelu_slope,
        )

        # ── Modulation spectrum head ───────────────────────────────────────
        self.mod_spec_head = _GLD_ModSpecHead(
            n_filters=n_filters,
            n_groups=am_groups,
            mod_fft=mod_fft,
            channels=am_channels,
            lrelu_slope=lrelu_slope,
        )

        # ── Spectral flux head ────────────────────────────────────────────
        self.flux_head = _GLD_FluxHead(
            channels=flux_channels,
            lrelu_slope=lrelu_slope,
        )

        # ── Dynamic spectral tilt head (Conv2d: freq × time) ──────────────
        # Operates on the full 2D log-loudness map [B, 1, 64, n_c] so it sees
        # how spectral tilt *evolves* — F0-dependent roll-off changes, vowel
        # onsets, etc. — rather than a single time-collapsed average.
        TC = tilt_channels
        self.tilt_backbone = nn.Sequential(
            weight_norm(nn.Conv2d(1,  TC, kernel_size=(9, 3), padding=(4, 1))),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            weight_norm(nn.Conv2d(TC, TC, kernel_size=(5, 3), dilation=(1, 2), padding=(2, 2))),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            weight_norm(nn.Conv2d(TC, TC, kernel_size=(3, 3), dilation=(1, 4), padding=(1, 4))),
            nn.LeakyReLU(lrelu_slope, inplace=True),
        )
        self.tilt_head = weight_norm(nn.Conv2d(TC, 1, kernel_size=(1, 1)))

    # ── internal: FFT convolution ──────────────────────────────────────────
    def _fft_gammatone(self, x_pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, T_pad = x_pad.shape
        L           = self.filter_len
        T           = T_pad - L + 1
        N_min       = T_pad + L - 1

        if N_min > self._N_canonical:
            raise ValueError(
                f"Input clip ({T} samples) exceeds the declared max_clip_ms budget. "
                f"N_min={N_min} > _N_canonical={self._N_canonical}. "
                f"Increase max_clip_ms at construction time."
            )
        N, W_re, W_im = self._N_canonical, self.W_re_fft, self.W_im_fft

        orig_dtype = x_pad.dtype
        X  = torch.fft.rfft(x_pad.to(torch.float32), n=N, dim=-1)
        re = torch.fft.irfft(X * W_re, n=N, dim=-1)[..., L-1: L-1+T].to(orig_dtype)
        im = torch.fft.irfft(X * W_im, n=N, dim=-1)[..., L-1: L-1+T].to(orig_dtype)
        return re, im

    # ── internal: instantaneous power ─────────────────────────────────────
    def _instantaneous_power(self, x: torch.Tensor) -> torch.Tensor:
        x_pad  = F.pad(x, (self.filter_len - 1, 0))
        re, im = self._fft_gammatone(x_pad)
        return re.pow(2) + im.pow(2)

    # ── internal: pool power → log-loudness ───────────────────────────────
    def _pool_to_loudness(self, power: torch.Tensor, hop_samples: int) -> torch.Tensor:
        B, C, T  = power.shape
        pad      = self.frame_samples // 2
        p_flat   = F.pad(power.reshape(B * C, 1, T), (pad, pad), mode='reflect')
        mean_pow = F.avg_pool1d(p_flat, kernel_size=self.frame_samples,
                                stride=hop_samples, padding=0)
        mean_pow = mean_pow.reshape(B, C, -1)
        loud = (mean_pow + self.power_floor).pow(self.loudness_exp)
        return torch.log(loud + self.log_floor)

    # ── internal: pool power → raw mean power (no compression) ───────────
    def _pool_power(self, power: torch.Tensor, hop_samples: int) -> torch.Tensor:
        B, C, T  = power.shape
        pad      = self.frame_samples // 2
        p_flat   = F.pad(power.reshape(B * C, 1, T), (pad, pad), mode='reflect')
        mean_pow = F.avg_pool1d(p_flat, kernel_size=self.frame_samples,
                                stride=hop_samples, padding=0)
        return mean_pow.reshape(B, C, -1)

    # ── forward step ──────────────────────────────────────────────────────
    def _forward_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = x.shape[0]

        power      = self._instantaneous_power(x)                              # [B, 64, T]
        log_loud_c = self._pool_to_loudness(power, self.hop_samples)           # [B, 64, n_c]
        mean_pow_f = self._pool_power(power, self.fine_hop_samples)            # [B, 64, n_f]

        # ── 2D backbone ───────────────────────────────────────────────────
        feat      = log_loud_c.unsqueeze(1)                                    # [B, 1, 64, n_c]
        feat_maps: List[torch.Tensor] = []
        for block in self.backbone:
            feat = block(feat)
            feat_maps.append(feat)

        main_logit = self.head_main(feat).squeeze(1).flatten(1)                # [B, ...]

        # ── AM head (learned band grouping) ───────────────────────────────
        am_logit, am_fmaps = self.am_head(mean_pow_f)                          # [B, 1]
        feat_maps.extend(am_fmaps)

        # ── Modulation spectrum head ───────────────────────────────────────
        mod_logit, mod_fmaps = self.mod_spec_head(mean_pow_f)                  # [B, 1]
        feat_maps.extend(mod_fmaps)

        # ── Spectral flux head ────────────────────────────────────────────
        flux_logit, flux_fmaps = self.flux_head(log_loud_c)                    # [B, 1]
        feat_maps.extend(flux_fmaps)

        # ── Dynamic spectral tilt head (2D: freq × time) ──────────────────
        # Reuses log_loud_c.unsqueeze(1) — [B, 1, 64, n_c] — no extra compute
        tilt_feat  = self.tilt_backbone(log_loud_c.unsqueeze(1))
        feat_maps.append(tilt_feat)
        tilt_logit = self.tilt_head(tilt_feat).flatten(1).mean(-1, keepdim=True)  # [B, 1]

        logits = torch.cat([main_logit, am_logit, mod_logit, flux_logit, tilt_logit], dim=-1)
        return logits, feat_maps

    # ── public API ────────────────────────────────────────────────────────
    def forward(
        self,
        y:          torch.Tensor,
        y_hat_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_hat_full           = y_hat_list[-1]
        logits_r, fmap_r     = self._forward_step(y)
        logits_g, fmap_g     = self._forward_step(y_hat_full)
        return [logits_r], [logits_g], [fmap_r], [fmap_g]
