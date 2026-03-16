import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.utils.checkpoint import checkpoint

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


class CoMBD_SBD_UnivHD_Combined(nn.Module):
    """
    CoMBD + SBD + UnivHD

      CoMBD:    collaborative multi-band: evaluates generator intermediate outputs at each resolution against real audio at the matching rate.
      SBD:      sub-band: evaluates the final full-res output in PQMF sub-bands.
      UnivHD:   harmonic-aware dynamic spectral resolution on full-res output.

    forward(y, y_hat_list) -> (y_d_rs, y_d_gs, fmap_rs, fmap_gs)

    y           : [B, 1, T] full-res real waveform
    y_hat_list  : List[Tensor] of 3 fakes (return_intermediates=True) [y_hat_lv2, y_hat_lv1, y_hat_full]
    """

    def __init__(
        self,
        sample_rate: int,
        segment_size_samples: int,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        # PQMF banks for preparing real audio hierarchy for CoMBD.
        # pqmf_lv2: 4-band -> subband 0 = 1/4-rate lowpass (matches gen stage lv2)
        # pqmf_lv1: 2-band -> subband 0 = 1/2-rate lowpass (matches gen stage lv1)
        self._pqmf_lv2 = PQMF(*_PQMF_LV2)
        self._pqmf_lv1 = PQMF(*_PQMF_LV1)

        self.combd = CoMBD(use_spectral_norm=use_spectral_norm)
        self.sbd = SBD(segment_size_samples=segment_size_samples, use_spectral_norm=use_spectral_norm)
        self.univhd = UnivHD(sample_rate=sample_rate)

        self.discriminators = nn.ModuleList([self.combd, self.sbd, self.univhd])

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
        if self.training and self.use_checkpointing:
            univhd_r, fmap_univhd_r = checkpoint(self.univhd, y, use_reentrant=False)
            univhd_g, fmap_univhd_g = checkpoint(self.univhd, y_hat_full, use_reentrant=False)
        else:
            univhd_r, fmap_univhd_r = self.univhd(y)
            univhd_g, fmap_univhd_g = self.univhd(y_hat_full)

        return (
            combd_rs + sbd_rs + [univhd_r],
            combd_gs + sbd_gs + [univhd_g],
            combd_frs + sbd_frs + [fmap_univhd_r],
            combd_fgs + sbd_fgs + [fmap_univhd_g],
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
        return torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window,
                          center=True, return_complex=True).abs()

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
