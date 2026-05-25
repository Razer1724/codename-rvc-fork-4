"""
Gammatone Loudness Discriminator  v3
─────────────────────────────────────────────────────────────────────────────
Perceptual discriminator based on the auditory gammatone filterbank.

Pipeline
────────
  waveform [B,1,T]
    │
    ├─ frozen complex gammatone filterbank (64 ERB channels, FFT conv)
    │    re [B,64,T]  +  im [B,64,T]
    │
    ├─ instantaneous power  re² + im²                  [B, 64, T]
    │
    ├─┬─ mean power @ 10 ms hop  (coarse)              [B, 64, n_c]
    │ │     → specific loudness  (^0.23 + log)
    │ │     → Conv2d residual backbone (DS + SE)
    │ │     → main head logits
    │ │
    │ └─ mean power @ 5 ms hop  (fine)                 [B, 64, n_f]
    │       → 8-group coarse band AM envelopes
    │       → dilated 1D Conv AM head logits
    │
    └─ spectral tilt head  (mean-spectrum Conv1d)       logits [B, 1]
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


# ── ERB / gammatone helpers (unchanged from v1) ───────────────────────────────

def _erb_space(fmin: float, fmax: float, n: int) -> np.ndarray:
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


def _build_gammatone_kernels(
    cfs: np.ndarray,
    sample_rate: int,
    order: int = 4,
    filter_len: int = 1024,
    n_fft: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build analytic (complex) gammatone FIR kernels (Patterson et al. 1988).

    n_fft : canonical FFT size for the cached kernel spectra.
            Must satisfy  n_fft >= max_clip_samples + 2*filter_len - 2.
            Defaults to 1-second sizing (sample_rate + filter_len) if not
            supplied, but callers that know their max clip length should pass
            a tighter value to halve FFT memory and compute.
    """
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

    k_re_fft  = (torch.fft.rfft(k_re_t, n=n_fft, dim=-1)
                 .squeeze(1).unsqueeze(0).to(torch.complex64))
    k_im_fft  = (torch.fft.rfft(k_im_t, n=n_fft, dim=-1)
                 .squeeze(1).unsqueeze(0).to(torch.complex64))

    return k_re_t, k_im_t, k_re_fft, k_im_fft


# ── lightweight building blocks ───────────────────────────────────────────────

class _DSConv2d(nn.Module):
    """
    Depthwise-Separable Conv2d  (Howard et al. 2017 – MobileNet).

    Factorises a standard (C_in, C_out, kH, kW) convolution into:
      • Depthwise  – one (kH×kW) filter per input channel  (groups=C_in)
      • Pointwise  – 1×1 conv to mix channels

    Param ratio  =  (kH·kW + C_out) / (kH·kW·C_out)
    At C=128, kernel 5×3:  standard=245 760 vs DS=22 144  ≈ 11× fewer.
    """
    def __init__(
        self,
        in_c:        int,
        out_c:       int,
        kernel_size,
        stride  = 1,
        padding = 0,
        dilation= 1,
    ):
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


class _SEBlock2d(nn.Module):
    """
    Squeeze-and-Excitation channel attention  (Hu et al. 2018 – SE-Net).

    Global-average-pools the spatial (band × frame) dimensions → two FC
    layers → sigmoid gate per channel → recalibrate feature map.

    Param cost:  2 · C · (C // r)  e.g. C=128, r=8 → 4 096 params per block.
    """
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
        w = x.mean(dim=(-2, -1))                     # [B, C]
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        return x * w


class _ResBlock2d(nn.Module):
    """
    Residual block: DS-conv → LReLU → DS-conv → SE → add skip.

    A 1×1 conv aligns the skip whenever C_in ≠ C_out or stride ≠ 1.
    The second DS-conv always uses dilation=1 so gradients don't vanish
    on the skip path.
    """
    def __init__(
        self,
        in_c:        int,
        out_c:       int,
        kernel_size,
        stride      = 1,
        padding     = 0,
        dilation    = 1,
        lrelu_slope: float = 0.1,
        se_r:        int   = 8,
    ):
        super().__init__()
        self.slope = lrelu_slope
        self.conv1 = _DSConv2d(in_c,  out_c, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = _DSConv2d(out_c, out_c, (3, 3), padding=(1, 1))
        self.se    = _SEBlock2d(out_c, r=se_r)

        need_proj = (in_c != out_c) or (
            (stride if isinstance(stride, int) else stride[0]) != 1 or
            (stride if isinstance(stride, int) else stride[1]) != 1
        )
        self.proj  = (weight_norm(nn.Conv2d(in_c, out_c, 1, stride=stride))
                      if need_proj else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = F.leaky_relu(self.conv1(x), self.slope)
        h    = self.conv2(h)
        h    = self.se(h)
        skip = self.proj(x) if self.proj is not None else x
        return F.leaky_relu(h + skip, self.slope)


class _AMHead(nn.Module):
    """
    Amplitude-Modulation temporal head.

    The gammatone filterbank decomposes audio into per-ERB-band analytic
    signals — this is exactly the representation used in classical AM analysis.
    v1 discarded the temporal structure after pooling.  This head keeps it.

    Architecture
    ────────────
    log_loud [B, 64, n_frames_fine]
      → reshape to 8 coarse frequency groups  [B, 8, n_frames_fine]
      → dilated 1D Conv stack  (dilations: 1, 2, 4, 8)
         receptive field = 1 + (ksize−1)·(1+2+4+8) = 1 + 4·15 = 61 frames
         At 5 ms hop → ~300 ms context (covers syllable-rate modulation)
      → head logit

    Why 8 groups?  Fine enough to separate low/mid/high vocal bands; coarse
    enough that the 1D conv stack stays cheap.

    Fine-hop (5 ms) vs coarse (10 ms): the extra pool is a second call to
    avg_pool1d on an already-computed [B·64, 1, T] power tensor — essentially
    free compared to the FFT filterbank.
    """
    def __init__(
        self,
        n_band_groups: int   = 8,
        channels:      int   = 32,
        lrelu_slope:   float = 0.1,
    ):
        super().__init__()
        self.n_groups   = n_band_groups
        self.slope      = lrelu_slope
        TC              = channels
        kernel_size     = 5
        dilations       = [1, 2, 4, 8]

        # Build as a list of (conv, activation) pairs so we can harvest fmaps
        self.convs = nn.ModuleList()
        in_c = n_band_groups
        for i, d in enumerate(dilations):
            pad = (kernel_size - 1) * d // 2
            self.convs.append(
                weight_norm(nn.Conv1d(in_c, TC, kernel_size, dilation=d, padding=pad))
            )
            in_c = TC

        self.head = weight_norm(nn.Conv1d(TC, 1, 1))

    def forward(
        self,
        power_fine: torch.Tensor,   # [B, 64, n_frames_fine]  mean power at fine hop
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, C, n_f = power_fine.shape
        # Compress to log-loudness (same formula as main path)
        log_loud = torch.log((power_fine + 1e-8).pow(0.23) + 1e-5)

        # Coarse-group the 64 ERB bands → 8 groups
        x = log_loud.reshape(B, self.n_groups, C // self.n_groups, n_f).mean(dim=2)

        fmaps: List[torch.Tensor] = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), self.slope)
            fmaps.append(x)

        logit = self.head(x).squeeze(1).mean(dim=-1, keepdim=True)  # [B, 1]
        return logit, fmaps


# ── main discriminator ────────────────────────────────────────────────────────

class GammatoneLoudnessDiscriminator(nn.Module):
    """
    Perceptual discriminator v3: ERB filterbank → loudness + AM → timbral features.

    Parameters
    ----------
    sample_rate   : int    — 24000 / 32000 / 44100 / 48000 …
    n_filters     : int    — ERB channel count (default 64)
    fmin          : float  — lowest ERB centre frequency in Hz (default 80)
    order         : int    — gammatone filter order; 4 is standard
    filter_ms     : float  — FIR kernel duration in ms (default 30)
    hop_ms        : float  — main loudness frame hop in ms (default 10)
    frame_ms      : float  — main loudness frame size in ms (default 20)
    fine_hop_ms   : float  — fine hop for AM head in ms (default 5)
    loudness_exp  : float  — Stevens power-law exponent ≈ 0.23 for sones
    channels      : int    — backbone channel width (default 128)
                            v1 used 32; DS convs let us run 128 at ~same cost
    am_channels   : int    — AM head internal channels (default 32)
    am_groups     : int    — number of coarse ERB band groups for AM (default 8)
    se_r          : int    — SE reduction ratio (default 8)
    lrelu_slope   : float
    power_floor   : float
    log_floor     : float
    max_clip_ms   : float  — hard upper bound on input clip duration in ms (default 500).
                            _N_canonical is sized exactly for this, halving FFT cost
                            vs the old 1-second default.  Raise if your training
                            segments are ever longer than this value.
    """

    def __init__(
        self,
        sample_rate:   int,
        n_filters:     int   = 64,
        fmin:          float = 80.0,
        order:         int   = 4,
        filter_ms:     float = 30.0,
        hop_ms:        float = 10.0,
        frame_ms:      float = 20.0,
        fine_hop_ms:   float = 5.0,
        loudness_exp:  float = 0.23,
        channels:      int   = 128,
        am_channels:   int   = 32,
        am_groups:     int   = 8,
        se_r:          int   = 8,
        lrelu_slope:   float = 0.1,
        power_floor:   float = 1e-8,
        log_floor:     float = 1e-5,
        max_clip_ms:   float = 500.0,
    ):
        super().__init__()

        self.n_filters    = n_filters
        self.loudness_exp = loudness_exp
        self.lrelu_slope  = lrelu_slope
        self.power_floor  = power_floor
        self.log_floor    = log_floor

        self.hop_samples       = round(sample_rate * hop_ms      / 1000.0)
        self.frame_samples     = round(sample_rate * frame_ms    / 1000.0)
        self.fine_hop_samples  = round(sample_rate * fine_hop_ms / 1000.0)

        # ── frozen gammatone filterbank ────────────────────────────────────
        fmax       = sample_rate * 0.45
        cfs        = _erb_space(fmin, fmax, n_filters)
        filter_len = max(256, round(sample_rate * filter_ms / 1000.0))
        self.filter_len = filter_len

        # ── FFT sizing: fit exactly to the declared clip budget ────────────
        # N_canonical is the FFT size used for every forward pass.
        # Old sizing: sample_rate + filter_len  (covers up to 1 s).
        # New sizing: covers only up to max_clip_ms, which is all we need.
        #
        # Why this saves:
        #   Convolution of a T-sample signal with an L-sample kernel needs
        #   N >= T + L - 1.  After the (L-1) left-pad in _instantaneous_power,
        #   T_pad = T + L - 1, so N_min = T_pad + L - 1 = T + 2L - 2.
        #   T_max = round(sample_rate * max_clip_ms / 1000)
        #
        # Example at 24 kHz, filter_ms=30, max_clip_ms=500:
        #   T_max = 12 000,  L = 720
        #   N_min = 12 000 + 1 440 − 2 = 13 438  →  2^14 = 16 384
        #   Old N = 2^ceil(log2(24 000 + 720))   = 2^15 = 32 768
        #   Saving: FFT size halved → ~50 % less FFT compute and buffer VRAM.
        #
        # Example at 48 kHz, filter_ms=30, max_clip_ms=500:
        #   T_max = 24 000,  L = 1 440
        #   N_min = 24 000 + 2 880 − 2 = 26 878  →  2^15 = 32 768
        #   Old N = 2^ceil(log2(48 000 + 1 440)) = 2^16 = 65 536
        #   Saving: again halved.
        max_clip_samples        = round(sample_rate * max_clip_ms / 1000.0)
        N_min_for_max           = max_clip_samples + 2 * filter_len - 2
        N_canonical             = 2 ** math.ceil(math.log2(N_min_for_max))
        self._N_canonical       = N_canonical
        self._max_clip_samples  = max_clip_samples  # used in assertion below

        k_re, k_im, W_re_fft, W_im_fft = _build_gammatone_kernels(
            cfs, sample_rate, order, filter_len, n_fft=N_canonical)

        self.register_buffer("kernels_re",   k_re)
        self.register_buffer("kernels_im",   k_im)
        self.register_buffer("W_re_fft",     W_re_fft)
        self.register_buffer("W_im_fft",     W_im_fft)
        self.register_buffer("centre_freqs", torch.from_numpy(cfs.astype(np.float32)))

        # ── Conv2d backbone  ───────────────────────────────────────────────
        # Spatial layout:  [B, 1, 64 ERB bands, n_frames]
        #
        # Stride (2,1) halves the band axis twice: 64 → 32 → 16.
        # Temporal axis never strided — dilation grows the receptive field
        # across time instead, keeping the full temporal resolution for the
        # head.
        #
        # Layer  In    Out   Kernel  Stride   Dilation(time)  RF bands  RF frames
        # ─────  ───   ───   ──────  ──────   ──────────────  ────────  ─────────
        #  0     1     C     5×3     (2,1)    1               5          3
        #  1     C     2C    5×3     (2,1)    1               9          5
        #  2     2C    2C    3×5     (1,1)    dil=(1,2)       11         13
        #  3     2C    2C    3×5     (1,1)    dil=(1,4)       11         29
        #
        C = channels
        self.backbone = nn.ModuleList([
            # [B,  1, 64, F] → [B,  C, 32, F]
            _ResBlock2d(1,    C,    (5, 3), stride=(2, 1), padding=(2, 1),
                        lrelu_slope=lrelu_slope, se_r=se_r),
            # [B,  C, 32, F] → [B, 2C, 16, F]
            _ResBlock2d(C,    C*2,  (5, 3), stride=(2, 1), padding=(2, 1),
                        lrelu_slope=lrelu_slope, se_r=se_r),
            # [B, 2C, 16, F] → [B, 2C, 16, F]   dilation in time = 2
            _ResBlock2d(C*2,  C*2,  (3, 5), stride=(1, 1),
                        padding=(1, 4), dilation=(1, 2),
                        lrelu_slope=lrelu_slope, se_r=se_r),
            # [B, 2C, 16, F] → [B, 2C, 16, F]   dilation in time = 4
            _ResBlock2d(C*2,  C*2,  (3, 5), stride=(1, 1),
                        padding=(1, 8), dilation=(1, 4),
                        lrelu_slope=lrelu_slope, se_r=se_r),
        ])
        self.head_main = weight_norm(nn.Conv2d(C * 2, 1, kernel_size=(1, 1)))

        # ── AM temporal head  ──────────────────────────────────────────────
        self.am_head = _AMHead(
            n_band_groups=am_groups,
            channels=am_channels,
            lrelu_slope=lrelu_slope,
        )

        # ── spectral tilt head  ────────────────────────────────────────────
        # v3: TC doubled (16 → 32), dilated convs, stronger capacity.
        TC = 32
        self.tilt_backbone = nn.Sequential(
            weight_norm(nn.Conv1d(1,  TC, kernel_size=9, padding=4)),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            weight_norm(nn.Conv1d(TC, TC, kernel_size=5, dilation=2, padding=4)),
            nn.LeakyReLU(lrelu_slope, inplace=True),
            weight_norm(nn.Conv1d(TC, TC, kernel_size=3, dilation=4, padding=4)),
            nn.LeakyReLU(lrelu_slope, inplace=True),
        )
        self.tilt_head = weight_norm(nn.Conv1d(TC, 1, kernel_size=1))

    # ── internal: FFT convolution ──────────────────────────────────────────
    def _fft_gammatone(
        self,
        x_pad: torch.Tensor,       # [B, 1, T+L-1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Linear convolution via FFT using cached kernel spectra.
        Returns re, im each [B, n_filters, T].
        """
        B, _, T_pad = x_pad.shape
        L           = self.filter_len
        T           = T_pad - L + 1
        N_min       = T_pad + L - 1

        # Hard guard: if this fires, a clip longer than max_clip_ms was passed.
        # Increase max_clip_ms at init rather than silently paying the cost of
        # an unplanned FFT recompute.
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
    def _instantaneous_power(
        self,
        x: torch.Tensor,            # [B, 1, T]
    ) -> torch.Tensor:              # [B, n_filters, T]
        """Returns re² + im² per ERB channel per sample.
        Separated from pooling so we can pool at multiple hop sizes cheaply.
        """
        x_pad    = F.pad(x, (self.filter_len - 1, 0))
        re, im   = self._fft_gammatone(x_pad)
        return re.pow(2) + im.pow(2)   # ≥ 0

    # ── internal: pool power → loudness ───────────────────────────────────
    def _pool_to_loudness(
        self,
        power:       torch.Tensor,  # [B, n_filters, T]
        hop_samples: int,
    ) -> torch.Tensor:              # [B, n_filters, n_frames]
        """
        Mean power per frame → specific loudness (power-law + log).

        Mean power pooled *before* compression is the correct Stevens' law
        basis: loudness ∝ intensity^0.23, and intensity ≡ mean power.
        """
        B, C, T   = power.shape
        pad       = self.frame_samples // 2
        p_flat    = power.reshape(B * C, 1, T)
        p_flat    = F.pad(p_flat, (pad, pad), mode='reflect')
        mean_pow  = F.avg_pool1d(p_flat,
                                 kernel_size=self.frame_samples,
                                 stride=hop_samples,
                                 padding=0)
        mean_pow  = mean_pow.reshape(B, C, -1)

        loud = (mean_pow + self.power_floor).pow(self.loudness_exp)
        return torch.log(loud + self.log_floor)

    def _pool_power(
        self,
        power:       torch.Tensor,  # [B, n_filters, T]
        hop_samples: int,
    ) -> torch.Tensor:              # [B, n_filters, n_frames]
        """Mean power per frame — no compression. For heads that compress themselves."""
        B, C, T = power.shape
        pad     = self.frame_samples // 2
        p_flat  = power.reshape(B * C, 1, T)
        p_flat  = F.pad(p_flat, (pad, pad), mode='reflect')
        mean_pow = F.avg_pool1d(p_flat,
                                kernel_size=self.frame_samples,
                                stride=hop_samples,
                                padding=0)
        return mean_pow.reshape(B, C, -1)

    # ── forward step ──────────────────────────────────────────────────────
    def _forward_step(
        self,
        x: torch.Tensor,            # [B, 1, T]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns
        -------
        logits    : [B, N]   — concatenated main + AM + tilt logits
        feat_maps : List     — feature maps for FM loss
        """
        B = x.shape[0]

        # shared power (computed once, pooled twice)
        power = self._instantaneous_power(x)  # [B, 64, T]

        # coarse loudness for the 2D backbone
        log_loud_c = self._pool_to_loudness(power, self.hop_samples)      # [B, 64, n_c]
        # fine power for the AM head (no loudness compression here; AMHead does its own)
        mean_pow_f = self._pool_power(power, self.fine_hop_samples)  # [B, 64, n_f]

        # ── 2D backbone ───────────────────────────────────────────────────
        feat      = log_loud_c.unsqueeze(1)  # [B, 1, 64, n_c]
        feat_maps: List[torch.Tensor] = []

        for block in self.backbone:
            feat = block(feat)
            feat_maps.append(feat)

        main_logit = self.head_main(feat)                    # [B, 1, 16, n_c]
        main_logit = main_logit.squeeze(1).flatten(1)        # [B, 16·n_c]

        # ── AM head ───────────────────────────────────────────────────────
        am_logit, am_fmaps = self.am_head(mean_pow_f)        # [B, 1], List
        feat_maps.extend(am_fmaps)

        # ── spectral tilt head ────────────────────────────────────────────
        mean_spec  = log_loud_c.mean(dim=-1).unsqueeze(1)    # [B, 1, 64]
        tilt_feat  = self.tilt_backbone(mean_spec)
        feat_maps.append(tilt_feat)
        tilt_logit = self.tilt_head(tilt_feat)               # [B, 1, 64]
        tilt_logit = tilt_logit.squeeze(1).mean(dim=-1, keepdim=True)  # [B, 1]

        # ── combine ───────────────────────────────────────────────────────
        logits = torch.cat([main_logit, am_logit, tilt_logit], dim=-1)
        return logits, feat_maps

    # ── public API (drop-in compatible with v1) ───────────────────────────
    def forward(
        self,
        y:          torch.Tensor,        # [B, 1, T] real waveform
        y_hat_list: List[torch.Tensor],  # fakes; last entry is full-res
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        """
        API-compatible with standard multi-discriminator loops.
        Returns (y_d_rs, y_d_gs, fmap_rs, fmap_gs).
        """
        y_hat_full           = y_hat_list[-1]
        logits_r, fmap_r     = self._forward_step(y)
        logits_g, fmap_g     = self._forward_step(y_hat_full)
        return [logits_r], [logits_g], [fmap_r], [fmap_g]
