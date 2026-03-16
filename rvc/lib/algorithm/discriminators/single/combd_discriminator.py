# Collaborative Multi-Band Discriminator (CoMBD)
# From: "Avocodo: Generative Adversarial Network for Artifact-free Vocoder"
# https://arxiv.org/abs/2206.13404

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm
from torch.nn.utils import spectral_norm

from rvc.lib.algorithm.discriminators.pqmf import PQMF


class CoMBDBlock(nn.Module):
    """Single sub-discriminator block used inside CoMBD."""

    def __init__(
        self,
        h_u: List[int],
        d_k: List[int],
        d_s: List[int],
        d_d: List[int],
        d_g: List[int],
        d_p: List[int],
        op_f: int,
        op_k: int,
        op_g: int,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        filters = [[1, h_u[0]]]
        for i in range(len(h_u) - 1):
            filters.append([h_u[i], h_u[i + 1]])

        self.convs = nn.ModuleList()
        for _f, _k, _s, _d, _g, _p in zip(filters, d_k, d_s, d_d, d_g, d_p):
            self.convs.append(norm_f(Conv1d(
                in_channels=_f[0],
                out_channels=_f[1],
                kernel_size=_k,
                stride=_s,
                dilation=_d,
                groups=_g,
                padding=_p,
            )))

        self.projection_conv = norm_f(Conv1d(
            in_channels=filters[-1][1],
            out_channels=op_f,
            kernel_size=op_k,
            groups=op_g,
        ))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap: List[torch.Tensor] = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            fmap.append(x)
        x = self.projection_conv(x)
        return x, fmap


# ---------------------------------------------------------------------------
# Default hyper-parameters taken directly from the Avocodo paper / config.
# These are the values used in the original implementation and work well
# across different sample rates (32 / 40 / 48 kHz) without modification.
# ---------------------------------------------------------------------------
_COMBD_H_U = [
    [16, 64, 256, 1024, 1024, 1024],
    [16, 64, 256, 1024, 1024, 1024],
    [16, 64, 256, 1024, 1024, 1024],
]
_COMBD_D_K = [
    [7,  11, 11, 11, 11, 5],
    [11, 21, 21, 21, 21, 5],
    [15, 41, 41, 41, 41, 5],
]
_COMBD_D_S  = [[1, 1, 4, 4, 4, 1]] * 3
_COMBD_D_D  = [[1, 1, 1, 1, 1, 1]] * 3
_COMBD_D_G  = [[1, 4, 16, 64, 256, 1]] * 3
_COMBD_D_P  = [
    [3,  5,  5,  5,  5,  2],
    [5,  10, 10, 10, 10, 2],
    [7,  20, 20, 20, 20, 2],
]
_COMBD_OP_F = [1, 1, 1]
_COMBD_OP_K = [3, 3, 3]
_COMBD_OP_G = [1, 1, 1]

# PQMF configs:  [subbands, taps, cutoff_ratio, beta]
# lv2 = 4-band → used to down-sample real audio to 1/4 resolution (matching gen stage 1)
# lv1 = 2-band → used to down-sample real audio to 1/2 resolution (matching gen stage 2)
_PQMF_LV2 = (4,  192, 0.13, 10.0)
_PQMF_LV1 = (2,  256, 0.25, 10.0)


class CoMBD(nn.Module):
    """
    Collaborative Multi-Band Discriminator.

    Expects:
        ys     : [y_lv2, y_lv1, y_full]  — real audio at each resolution
        ys_hat : [y_hat_lv2, y_hat_lv1, y_hat_full]  — fake audio at each resolution

    The three resolutions must match the three blocks:
        block[0] ← 1/4 rate   (lv2)
        block[1] ← 1/2 rate   (lv1)
        block[2] ← full rate  (full)

    Internally the discriminator also runs block[0] and block[1] against PQMF
    sub-bands of the full-resolution signal (the "multi-scale" path), making
    each block collaborate on both the generator intermediate output AND the
    frequency-matched real sub-band.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        # PQMF banks stored as nn.ModuleList so they move to the correct device
        self.pqmf = nn.ModuleList([
            PQMF(*_PQMF_LV2),  # 4-band → subband 0 = 1/4-rate lowpass
            PQMF(*_PQMF_LV1),  # 2-band → subband 0 = 1/2-rate lowpass
        ])

        self.blocks = nn.ModuleList()
        for h_u, d_k, d_s, d_d, d_g, d_p, op_f, op_k, op_g in zip(
            _COMBD_H_U, _COMBD_D_K, _COMBD_D_S,
            _COMBD_D_D, _COMBD_D_G, _COMBD_D_P,
            _COMBD_OP_F, _COMBD_OP_K, _COMBD_OP_G,
        ):
            self.blocks.append(CoMBDBlock(
                h_u, d_k, d_s, d_d, d_g, d_p,
                op_f, op_k, op_g,
                use_spectral_norm=use_spectral_norm,
            ))

    def _block_forward(
        self,
        inputs: List[torch.Tensor],
        blocks: nn.ModuleList,
        outs: List[torch.Tensor],
        fmaps: List[List[torch.Tensor]],
    ):
        for x, block in zip(inputs, blocks):
            out, fmap = block(x)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps

    def forward(
        self,
        ys: List[torch.Tensor],
        ys_hat: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        ys / ys_hat: lists of length 3 — [lv2_wave, lv1_wave, full_wave]
        Returns: (outs_real, outs_fake, fmaps_real, fmaps_fake)
        """
        y_full     = ys[-1]      # full-resolution real audio
        y_hat_full = ys_hat[-1]

        # Build PQMF multi-scale inputs from the full-res signal
        multi_real = [pqmf.analysis(y_full)[:, :1, :]     for pqmf in self.pqmf]
        multi_fake = [pqmf.analysis(y_hat_full)[:, :1, :] for pqmf in self.pqmf]

        outs_real, fmaps_real = [], []
        outs_fake, fmaps_fake = [], []

        # Hierarchical path: each block sees the matching intermediate output
        outs_real, fmaps_real = self._block_forward(ys,     self.blocks,       outs_real, fmaps_real)
        outs_fake, fmaps_fake = self._block_forward(ys_hat, self.blocks,       outs_fake, fmaps_fake)

        # Multi-scale path: blocks[0] and blocks[1] also see the PQMF sub-bands
        outs_real, fmaps_real = self._block_forward(multi_real, self.blocks[:-1], outs_real, fmaps_real)
        outs_fake, fmaps_fake = self._block_forward(multi_fake, self.blocks[:-1], outs_fake, fmaps_fake)

        return outs_real, outs_fake, fmaps_real, fmaps_fake
