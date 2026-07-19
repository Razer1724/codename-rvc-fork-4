import torch
import torch.nn.functional as F

import torch.nn as nn

from torch.nn import Conv2d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

import typing
from typing import Optional, List, Union, Dict, Tuple

from torch.utils.checkpoint import checkpoint
from rvc.train.utils import AttrDict

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE
from rvc.lib.algorithm.discriminators.multi.pqmf import PQMF
from rvc.lib.algorithm.discriminators.multi.hmdd import (
    SBDBlock,
    _PQMF_SBD, _PQMF_FSBD,
    _SBD_FILTERS, _SBD_STRIDES, _SBD_KERNEL_SIZES,
    _SBD_DILATIONS, _SBD_BAND_RANGES, _SBD_TRANSPOSE,
)

LRELU_INPLACE = False

class MPD_MSD_MRD_Combined(torch.nn.Module):
    """
    Class combining:
    Multi-Period, Sub-Band and Multi-Resolution Discriminators.
    """

    def __init__(self, segment_size_samples: int, use_spectral_norm: bool = False, use_checkpointing: bool = False, **multi_resolution_cfg):
        super().__init__()
        self.mrd_cfg = multi_resolution_cfg
        self.use_checkpointing = use_checkpointing

        periods = [2, 3, 5, 7, 11]
        #periods = [2, 3, 5, 7, 11, 17, 23, 37] # MPD carry style

        self.resolutions = self.mrd_cfg["resolutions"]

        assert len(self.resolutions) == 3, \
            f"MRD requires list of list with len=3, each element having a list with len=3. Got {self.resolutions}"


        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
            + [DiscriminatorR(self.mrd_cfg, resolution) for resolution in self.resolutions]
        )
        self.sbd = SBD(segment_size_samples, use_spectral_norm=use_spectral_norm)

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for (y_d_r, fmap_r), (y_d_g, fmap_g) in zip(self.sbd(y), self.sbd(y_hat)):
            y_d_rs.append(y_d_r);  fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g);  fmap_gs.append(fmap_g)

        for d in self.discriminators:
            if self.training and self.use_checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SBD(torch.nn.Module):
    """
    Sub-Band Discriminator — 1:1 with HMDD SBD.
    Shared PQMF analysis, per-band SBDBlock, returns list of (logit, fmap) tuples.
    """

    def __init__(self, segment_size_samples: int, use_spectral_norm: bool = False):
        super().__init__()

        self.pqmf   = PQMF(*_PQMF_SBD)
        self.f_pqmf = PQMF(*_PQMF_FSBD)

        self.band_ranges = _SBD_BAND_RANGES
        self.transpose   = _SBD_TRANSPOSE

        self.blocks = torch.nn.ModuleList()
        for _f, _k, _d, _s, br, tr in zip(
            _SBD_FILTERS, _SBD_KERNEL_SIZES,
            _SBD_DILATIONS, _SBD_STRIDES,
            _SBD_BAND_RANGES, _SBD_TRANSPOSE,
        ):
            segment_dim = (segment_size_samples // br[1]) - br[0] if tr else (br[1] - br[0])
            self.blocks.append(SBDBlock(
                segment_dim=segment_dim, filters=_f, kernel_size=_k,
                dilations=_d, strides=_s, use_spectral_norm=use_spectral_norm,
            ))

    def forward(self, x):
        y_sub     = self.pqmf.analysis(x)
        y_sub_f   = self.f_pqmf.analysis(x)

        band_outputs = []
        for d, br, tr in zip(self.blocks, self.band_ranges, self.transpose):
            if tr:
                _x = torch.transpose(y_sub_f[:, br[0]:br[1], :], 1, 2)
            else:
                _x = y_sub[:, br[0]:br[1], :]
            out, fmap = d(_x)
            band_outputs.append((torch.flatten(out, 1, -1), fmap))
        return band_outputs


class DiscriminatorP(torch.nn.Module):
    """
    Discriminator for the long-term component.

    This class implements a discriminator for the long-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal at a given
    period.

    Args:
        period (int): Period of the discriminator.
        kernel_size (int): Kernel size of the convolutional layers. Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE, inplace=LRELU_INPLACE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorR(nn.Module):
    def __init__(self, cfg: AttrDict, resolution: List[List[int]]):
        super().__init__()
        self.cfg = cfg

        self.resolution = resolution
        assert len(self.resolution) == 3, f"MRD layer requires list with len=3, got {self.resolution}"

        self.lrelu_slope = 0.1
        self.d_mult = 1
        n_fft, hop_length, win_length = self.resolution
        self.register_buffer("window", torch.ones(win_length), persistent=False)

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = weight_norm(
            nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1))
        )

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
        )

        return torch.abs(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = self.spectrogram(x).unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope, inplace=LRELU_INPLACE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap