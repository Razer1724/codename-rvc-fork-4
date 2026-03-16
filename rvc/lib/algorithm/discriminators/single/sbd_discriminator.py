# Sub-Band Discriminator (SBD)
# From: "Avocodo: Generative Adversarial Network for Artifact-free Vocoder"
# https://arxiv.org/abs/2206.13404

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, spectral_norm

from rvc.lib.algorithm.discriminators.pqmf import PQMF


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class MDC(nn.Module):
    """Multi-Dilated Conv: parallel dilated branches → sum → stride conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strides: int,
        kernel_size: List[int],
        dilations: List[int],
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.d_convs = nn.ModuleList([
            norm_f(Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=_k,
                dilation=_d,
                padding=_get_padding(_k, _d),
            ))
            for _k, _d in zip(kernel_size, dilations)
        ])

        # NOTE: padding uses last _k, _d from the zip — preserved from original
        self.post_conv = norm_f(Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=strides,
            padding=_get_padding(_k, _d),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _out = None
        for conv in self.d_convs:
            _x = F.leaky_relu(conv(x).unsqueeze(-1), 0.2)
            _out = _x if _out is None else torch.cat([_out, _x], dim=-1)
        x = torch.sum(_out, dim=-1)
        return F.leaky_relu(self.post_conv(x), 0.2)


class SBDBlock(nn.Module):
    """One SBD discriminator operating on a particular frequency band."""

    def __init__(
        self,
        segment_dim: int,
        strides: List[int],
        filters: List[int],
        kernel_size: List[List[int]],
        dilations: List[List[int]],
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append((filters[i], filters[i + 1]))

        self.convs = nn.ModuleList([
            MDC(
                in_channels=_f[0],
                out_channels=_f[1],
                strides=_s,
                kernel_size=_k,
                dilations=_d,
                use_spectral_norm=use_spectral_norm,
            )
            for _s, _f, _k, _d in zip(strides, filters_in_out, kernel_size, dilations)
        ])

        last_ch = filters[-1]
        self.post_conv = norm_f(Conv1d(last_ch, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap: List[torch.Tensor] = []
        for conv in self.convs:
            x = conv(x)
            fmap.append(x)
        x = self.post_conv(x)
        return x, fmap


# ---------------------------------------------------------------------------
# Avocodo default SBD configuration — sample-rate agnostic except for the
# transposed discriminator whose segment_dim scales with audio segment length.
# ---------------------------------------------------------------------------
_SBD_FILTERS = [
    [64, 128, 256, 256, 256],
    [64, 128, 256, 256, 256],
    [64, 128, 256, 256, 256],
    [32,  64, 128, 128, 128],
]
_SBD_STRIDES = [[1, 1, 3, 3, 1]] * 4
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

# PQMF params for SBD band decomposition
# sbd  : 16-band regular PQMF
# fsbd : 64-band PQMF used for the transposed discriminator
_PQMF_SBD  = (16, 256, 0.03, 10.0)
_PQMF_FSBD = (64, 256, 0.10,  9.0)


class SBD(nn.Module):
    """
    Sub-Band Discriminator.

    Splits the input waveform into PQMF sub-bands and runs a separate
    SBDBlock on each frequency band.

    The last discriminator uses a 64-band PQMF with transposed input
    (frequency axis ↔ time axis), making its in_channels depend on the
    audio segment length:
        segment_dim = segment_size_samples // 64

    Args:
        segment_size_samples (int): Audio segment length **in samples**
            (= config.train.segment_size * config.data.hop_length).
        use_spectral_norm (bool): Use spectral norm instead of weight norm.
    """

    def __init__(self, segment_size_samples: int, use_spectral_norm: bool = False):
        super().__init__()

        self.pqmf   = PQMF(*_PQMF_SBD)
        self.f_pqmf = PQMF(*_PQMF_FSBD)

        self.discriminators = nn.ModuleList()
        for _f, _k, _d, _s, br, tr in zip(
            _SBD_FILTERS, _SBD_KERNEL_SIZES, _SBD_DILATIONS,
            _SBD_STRIDES, _SBD_BAND_RANGES, _SBD_TRANSPOSE,
        ):
            if tr:
                # Transposed: time-axis becomes channels; dim depends on segment length
                segment_dim = segment_size_samples // br[1] - br[0]
            else:
                segment_dim = br[1] - br[0]

            self.discriminators.append(SBDBlock(
                segment_dim=segment_dim,
                filters=_f,
                kernel_size=_k,
                dilations=_d,
                strides=_s,
                use_spectral_norm=use_spectral_norm,
            ))

        self.band_ranges = _SBD_BAND_RANGES
        self.transpose   = _SBD_TRANSPOSE

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        y_sub     = self.pqmf.analysis(y)
        y_hat_sub = self.pqmf.analysis(y_hat)
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
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
