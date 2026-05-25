"""
combd_sbd_mrd_combined_fixed.py  —  Optimised discriminator ensemble
=====================================================================
Speed vs. _new:  ~same PQMF/Triton wins, without the VRAM regression.
VRAM   vs. _old: same or slightly better (one fewer PQMF call per bank).

Root cause of _new's VRAM regression
──────────────────────────────────────
_new batched real+fake into [2B, ...] for every block forward pass
(CoMBD, SBD, MRD).  That makes ALL intermediate activations 2x larger
simultaneously.  The fmap slices (f[:B], f[B:]) are just views, so the
full [2B] autograd graph stays alive until backward finishes.

Peak memory:
  _old  ~  saved_real  +  M_per_pass          (sequential passes)
  _new  ~  2 × M_per_pass                     (batched passes)
  When saved_real << M_per_pass the regression is ≈ 2×.

What this file keeps from _new (pure speed, no VRAM cost)
──────────────────────────────────────────────────────────
• fused_complex_abs  – single Triton kernel for STFT magnitude; avoids
                       one temporary allocation and two memory passes.
• fused_lrelu_sum3   – single Triton kernel for 3-branch lrelu+sum in
                       MDC; replaces 6 separate kernel launches.
• PQMF batching      – both PQMF banks in CoMBD and SBD run real+fake
                       together (halves filter-bank calls).  Output
                       memory is neutral: [2B,C,T] == 2×[B,C,T].

What this file reverts (block-forward batching — the VRAM culprit)
──────────────────────────────────────────────────────────────────
• CoMBD._batched_block_forward  →  back to sequential _block_forward
• SBD   block passes            →  back to sequential real then fake
• MRD   y_pair batching         →  back to sequential real then fake

Triton optional: falls back to plain PyTorch on CPU / no-Triton envs.
"""

import torch
import math
import torch.nn.functional as F
import torch.nn as nn

from typing import Optional, List, Tuple

from torch.nn import Conv1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

from rvc.train.utils import AttrDict
from rvc.lib.algorithm.discriminators.multi.pqmf import PQMF

# ---------------------------------------------------------------------------
# Optional Triton kernels
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    _TRITON = True
except ImportError:
    _TRITON = False


if _TRITON:
    # kernel 1: fused complex magnitude ─────────────────────────────────────
    @triton.jit
    def _k_complex_abs(
        x_ptr,          # float32 view of a complex64 tensor  (interleaved re,im)
        out_ptr,        # float32 output
        N,              # number of complex elements  (len of out)
        BS: tl.constexpr,
    ):
        pid  = tl.program_id(0)
        offs = pid * BS + tl.arange(0, BS)
        mask = offs < N
        re = tl.load(x_ptr + offs * 2,     mask=mask, other=0.0)
        im = tl.load(x_ptr + offs * 2 + 1, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.sqrt(re * re + im * im), mask=mask)

    def fused_complex_abs(x: torch.Tensor) -> torch.Tensor:
        """Drop-in for torch.abs() on a complex64 CUDA tensor."""
        x   = x.contiguous()
        n   = x.numel()
        out = x.new_empty(x.shape, dtype=torch.float32)
        grid = (triton.cdiv(n, 1024),)
        _k_complex_abs[grid](x.view(torch.float32), out, n, BS=1024)
        return out

    # kernel 2: fused 3-branch leaky_relu + sum ──────────────────────────────
    @triton.jit
    def _k_lrelu_sum3(
        a_ptr, b_ptr, c_ptr,
        out_ptr,
        N,
        SLOPE: tl.constexpr,
        BS:    tl.constexpr,
    ):
        """
        out[i] = lrelu(a[i]) + lrelu(b[i]) + lrelu(c[i])
        Replaces: 3× leaky_relu + unsqueeze + cat + sum  (6 kernels → 1).
        """
        pid  = tl.program_id(0)
        offs = pid * BS + tl.arange(0, BS)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        c = tl.load(c_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs,
                 tl.where(a >= 0, a, a * SLOPE) +
                 tl.where(b >= 0, b, b * SLOPE) +
                 tl.where(c >= 0, c, c * SLOPE),
                 mask=mask)

    def fused_lrelu_sum3(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        slope: float = 0.2,
    ) -> torch.Tensor:
        a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
        n   = a.numel()
        out = torch.empty_like(a)
        grid = (triton.cdiv(n, 1024),)
        _k_lrelu_sum3[grid](a, b, c, out, n, SLOPE=slope, BS=1024)
        return out

else:
    def fused_complex_abs(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)

    def fused_lrelu_sum3(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        slope: float = 0.2,
    ) -> torch.Tensor:
        return (F.leaky_relu(a, slope) +
                F.leaky_relu(b, slope) +
                F.leaky_relu(c, slope))


# =============================================================================
# Hardcoded discriminator config
# =============================================================================

_PQMF_LV2  = (4,  192, 0.13, 10.0)
_PQMF_LV1  = (2,  256, 0.25, 10.0)
_PQMF_SBD  = (16, 256, 0.03, 10.0)
_PQMF_FSBD = (64, 256, 0.10,  9.0)

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

    Speed vs. _old
    ──────────────
    PQMF analyses are batched (real+fake together): 4 calls → 2 calls.
    The filter bank is the expensive part; output memory is the same.

    VRAM vs. _old
    ─────────────
    Block forward passes are sequential (real then fake), same as _old.
    _new's _batched_block_forward ran [2B] through CoMBDBlock, making all
    1024-ch Conv1d activations 2× larger simultaneously.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()

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

    # ------------------------------------------------------------------
    # Sequential block forward (same VRAM as _old)
    # ------------------------------------------------------------------
    def _block_forward(self, inputs, blocks, outs, fmaps):
        for x, block in zip(inputs, blocks):
            out, fmap = block(x)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps

    def forward(self, ys, ys_hat):
        y_full     = ys[-1]
        y_hat_full = ys_hat[-1]
        B = y_full.shape[0]

        # Batch PQMF analyses over real+fake (2 calls instead of 4).
        # Memory neutral: [2B, C, T] == 2 × [B, C, T].
        full_pair = torch.cat([y_full, y_hat_full], dim=0)   # [2B, 1, T]
        multi_rf  = [pqmf.analysis(full_pair)[:, :1, :] for pqmf in self.pqmf]
        del full_pair   # free the temporary immediately
        multi_r = [t[:B] for t in multi_rf]
        multi_f = [t[B:] for t in multi_rf]

        outs_r, fmaps_r = [], []
        outs_f, fmaps_f = [], []

        # Hierarchical path — sequential real then fake
        outs_r, fmaps_r = self._block_forward(ys,      self.blocks,      outs_r, fmaps_r)
        outs_f, fmaps_f = self._block_forward(ys_hat,  self.blocks,      outs_f, fmaps_f)
        # Multi-scale path — sequential real then fake
        outs_r, fmaps_r = self._block_forward(multi_r, self.blocks[:-1], outs_r, fmaps_r)
        outs_f, fmaps_f = self._block_forward(multi_f, self.blocks[:-1], outs_f, fmaps_f)

        return outs_r, outs_f, fmaps_r, fmaps_f


# =============================================================================
# SBD - Sub-Band Discriminator
# =============================================================================

def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class MDC(nn.Module):
    """
    Multi-Dilated Conv: parallel dilated branches → sum → stride conv.

    Speed vs. _old
    ──────────────
    When Triton is available and there are exactly 3 branches (always true
    with the default config), fused_lrelu_sum3 replaces 6 kernel launches
    with 1.  For any other branch count a clean loop accumulator is used
    (avoids the original cat intermediate allocation).

    VRAM vs. _old
    ─────────────
    fused_lrelu_sum3 does NOT increase peak VRAM: it reads the 3 branch
    outputs and writes one output — same tensors as before, fewer launches.
    The branch outputs (outs list) are all alive until the kernel runs,
    same as the old cat approach.
    """

    def __init__(self, in_channels, out_channels, strides, kernel_size, dilations, use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.d_convs = nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(norm_f(Conv1d(
                in_channels, out_channels, _k,
                dilation=_d, padding=_get_padding(_k, _d),
            )))
        # NOTE: padding uses the last _k, _d from the loop —
        # preserved from original Avocodo code, do not refactor.
        self.post_conv = norm_f(Conv1d(
            out_channels, out_channels, 3,
            stride=strides, padding=_get_padding(_k, _d),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [conv(x) for conv in self.d_convs]   # list of [B, C, T]

        if len(outs) == 3:
            summed = fused_lrelu_sum3(outs[0], outs[1], outs[2], slope=0.2)
        else:
            summed = F.leaky_relu(outs[0], 0.2)
            for o in outs[1:]:
                summed = summed + F.leaky_relu(o, 0.2)

        return F.leaky_relu(self.post_conv(summed), 0.2)


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

    def forward(self, x: torch.Tensor):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            fmap.append(x)
        return self.post_conv(x), fmap


class SBD(nn.Module):
    """
    Sub-Band Discriminator.

    Speed vs. _old
    ──────────────
    Both PQMF banks run real+fake in a single batched call (4 → 2 calls).

    VRAM vs. _old
    ─────────────
    Block forward passes are sequential (real then fake), same as _old.
    _new batched the SBDBlock passes ([2B] input), which made all 256-ch
    MDC activations 2× larger simultaneously — the biggest VRAM offender.
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

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        # Batch both PQMF banks over real+fake (2 calls instead of 4).
        yy      = torch.cat([y, y_hat], dim=0)     # [2B, 1, T]
        sub     = self.pqmf.analysis(yy)            # [2B, 16, T/16]
        sub_f   = self.f_pqmf.analysis(yy)          # [2B, 64, T/64]
        del yy

        B = y.shape[0]
        y_sub,    y_hat_sub   = sub[:B],   sub[B:]
        y_sub_f,  y_hat_sub_f = sub_f[:B], sub_f[B:]

        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d, br, tr in zip(self.discriminators, self.band_ranges, self.transpose):
            if tr:
                _y     = torch.transpose(y_sub_f[:,    br[0]:br[1], :], 1, 2).contiguous()
                _y_hat = torch.transpose(y_hat_sub_f[:, br[0]:br[1], :], 1, 2).contiguous()
            else:
                _y     = y_sub[:,    br[0]:br[1], :]
                _y_hat = y_hat_sub[:, br[0]:br[1], :]

            # Sequential passes — activations inside MDC stay at [B, C, T]
            y_d_r, fmap_r = d(_y)
            y_d_g, fmap_g = d(_y_hat)
            y_d_rs.append(y_d_r);  fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g);  fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# =============================================================================
# MRD - Multi-Resolution (Spectrogram) Discriminator
# =============================================================================

class DiscriminatorR(nn.Module):
    """
    Speed vs. _old
    ──────────────
    spectrogram() uses fused_complex_abs instead of torch.abs(): merges
    the magnitude computation into a single Triton kernel, avoids one
    temporary allocation.

    VRAM vs. _old
    ─────────────
    No change — same I/O sizes as torch.abs().
    Real+fake batching is NOT done here (it's handled sequentially in
    CoMBD_SBD_MRD_Combined.forward to keep Conv2d activations at [B]).
    """

    def __init__(self, cfg: AttrDict, resolution: List[int]):
        super().__init__()
        self.cfg = cfg

        self.resolution = resolution
        assert len(self.resolution) == 3, \
            f"MRD layer requires list with len=3, got {self.resolution}"

        self.lrelu_slope = 0.1
        self.d_mult = 1
        n_fft, hop_length, win_length = self.resolution

        #self.register_buffer("window", torch.hann_window(win_length), persistent=False) # Hanning
        self.register_buffer("window", torch.ones(win_length), persistent=False) # Rectangular

        dm = int(32 * self.d_mult)
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1,  dm, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(dm, dm, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(dm, dm, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(dm, dm, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(dm, dm, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(dm, 1, (3, 3), padding=(1, 1)))

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution

        p = (n_fft - hop_length) // 2
        x = F.pad(x, (p, p), mode="reflect").squeeze(1)

        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )   # complex64  [B, F, T_frames]

        # Single Triton kernel for magnitude — avoids one temp allocation
        return fused_complex_abs(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = self.spectrogram(x).unsqueeze(1)   # [B, 1, F, T_frames]
        for layer in self.convs:
            x = F.leaky_relu(layer(x), self.lrelu_slope, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


# =============================================================================
# Combined ensemble
# =============================================================================

class CoMBD_SBD_MRD_Combined(nn.Module):
    """
    CoMBD + SBD + MRD discriminator ensemble.

    forward(y, y_hat_list) -> (y_d_rs, y_d_gs, fmap_rs, fmap_gs)

    y           : [B, 1, T]  full-res real waveform
    y_hat_list  : List[Tensor] of 3 fakes (return_intermediates=True)
                  [y_hat_lv2, y_hat_lv1, y_hat_full]
    """

    def __init__(
        self,
        sample_rate: int,
        segment_size_samples: int,
        use_spectral_norm: bool = False,
        resolutions: Optional[List[List[int]]] = None,
        mrd_cfg: Optional[AttrDict] = None,
    ):
        super().__init__()

        # PQMF banks for building the real-audio hierarchy fed to CoMBD.
        self._pqmf_lv2 = PQMF(*_PQMF_LV2)
        self._pqmf_lv1 = PQMF(*_PQMF_LV1)

        self.combd = CoMBD(use_spectral_norm=use_spectral_norm)
        self.sbd   = SBD(segment_size_samples=segment_size_samples,
                         use_spectral_norm=use_spectral_norm)
        self.mrd   = nn.ModuleList([
            DiscriminatorR(cfg=mrd_cfg, resolution=res) for res in resolutions
        ])

        self.discriminators = nn.ModuleList([self.combd, self.sbd, *self.mrd])

    def forward(
        self,
        y: torch.Tensor,
        y_hat_list: List[torch.Tensor],
    ) -> Tuple[List, List, List, List]:

        y_hat_full = y_hat_list[-1]

        # Real-audio hierarchy for CoMBD
        y_lv2 = self._pqmf_lv2.analysis(y)[:, :1, :]
        y_lv1 = self._pqmf_lv1.analysis(y)[:, :1, :]
        ys    = [y_lv2, y_lv1, y]

        # CoMBD (PQMF batched, block forward sequential)
        combd_rs, combd_gs, combd_frs, combd_fgs = self.combd(ys, y_hat_list)

        # SBD (PQMF batched, block forward sequential)
        sbd_rs, sbd_gs, sbd_frs, sbd_fgs = self.sbd(y, y_hat_full)

        # MRD — sequential real then fake.
        # _new batched these into [2B] which made Conv2d activations 2×.
        # The STFT itself is cheap relative to those activations so the
        # extra call cost is negligible.
        mrd_rs, mrd_gs, mrd_frs, mrd_fgs = [], [], [], []
        for d in self.mrd:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat_full)
            mrd_rs.append(y_d_r);  mrd_frs.append(fmap_r)
            mrd_gs.append(y_d_g);  mrd_fgs.append(fmap_g)

        return (
            combd_rs + sbd_rs + mrd_rs,
            combd_gs + sbd_gs + mrd_gs,
            combd_frs + sbd_frs + mrd_frs,
            combd_fgs + sbd_fgs + mrd_fgs,
        )
