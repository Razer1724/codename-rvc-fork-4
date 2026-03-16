# Pseudo Quadrature Mirror Filter Bank
# Adapted from: https://github.com/kan-bayashi/ParallelWaveGAN
# Paper: "Near-perfect-reconstruction pseudo-QMF banks" (IEEE, 1994)
# MIT License

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def design_prototype_filter(taps: int = 62, cutoff_ratio: float = 0.142, beta: float = 9.0) -> np.ndarray:
    """Design the Kaiser-windowed prototype lowpass filter for PQMF."""
    assert taps % 2 == 0, "taps must be even."
    assert 0.0 < cutoff_ratio < 1.0, "cutoff_ratio must be in (0, 1)."
    omega_c = np.pi * cutoff_ratio
    n = np.arange(taps + 1)
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (n - 0.5 * taps)) / (np.pi * (n - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix NaN at centre tap
    return h_i * np.kaiser(taps + 1, beta)


class PQMF(nn.Module):
    """
    Pseudo Quadrature Mirror Filter Bank.

    The cutoff_ratio and beta are optimised per subbands count:
        subbands=2  → cutoff_ratio=0.25,  beta=10.0,  taps=256
        subbands=4  → cutoff_ratio=0.13,  beta=10.0,  taps=192
        subbands=16 → cutoff_ratio=0.03,  beta=10.0,  taps=256
        subbands=64 → cutoff_ratio=0.10,  beta=9.0,   taps=256

    Args:
        subbands     (int):   Number of subbands.
        taps         (int):   Filter length (even).
        cutoff_ratio (float): Prototype filter cut-off ratio in (0, 1).
        beta         (float): Kaiser window beta.
    """

    def __init__(self, subbands: int = 4, taps: int = 62, cutoff_ratio: float = 0.142, beta: float = 9.0):
        super().__init__()
        self.subbands = subbands

        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis  = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            phase = (2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - taps / 2)
            sign  = (-1) ** k * np.pi / 4
            h_analysis[k]  = 2 * h_proto * np.cos(phase + sign)
            h_synthesis[k] = 2 * h_proto * np.cos(phase - sign)

        self.register_buffer("analysis_filter",  torch.from_numpy(h_analysis).float().unsqueeze(1))
        self.register_buffer("synthesis_filter", torch.from_numpy(h_synthesis).float().unsqueeze(0))

        updown = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown)

        self.pad_fn = nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, T] → [B, subbands, T // subbands]"""
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        """[B, subbands, T // subbands] → [B, 1, T]"""
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)
