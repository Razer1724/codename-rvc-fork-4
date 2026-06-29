"""
pcph_geosaw_fused.py
====================
Band-limited sawtooth pitch signal via Geometric Sine-Sum.

Fixed-r variant. No adaptive mode.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N_elements'],
)
@triton.jit
def _pcph_geosaw_kernel(
    PHASE,
    N_HARMS,
    OUT,
    stride_b,
    stride_c,
    stride_n,
    N_elements,
    C,
    R,
    EPS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx  = pid // C
    channel_idx = pid % C
    bc_offset  = batch_idx * stride_b + channel_idx * stride_c

    PHASE   = PHASE   + bc_offset
    N_HARMS = N_HARMS + bc_offset
    OUT     = OUT     + bc_offset

    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elements

    phi = tl.load(PHASE   + offsets * stride_n, mask=mask)
    n   = tl.load(N_HARMS + offsets * stride_n, mask=mask)

    r = tl.clamp(R, 1e-6, 1.0 - 1e-6)

    log_r = tl.log(r)
    rNp1  = tl.exp((n + 1.0) * log_r)
    rNp2  = rNp1 * r

    sp  = tl.sin(phi)
    cp  = tl.cos(phi)
    sN1 = tl.sin((n + 1.0) * phi)
    sN  = tl.sin(n * phi)

    numer = r * sp - rNp1 * sN1 + rNp2 * sN
    denom = 1.0 - 2.0 * r * cp + r * r

    result = tl.where(tl.abs(denom) < EPS, 0.0, numer / denom)
    tl.store(OUT + offsets * stride_n, result, mask=mask)


def pcph_geosaw_fwd(phase, n_harms, r=0.90, eps=1e-6):
    """
    Fixed-r GeoSaw kernel.

    Args:
        phase:   [B, C, T] float32, instantaneous phase in radians
        n_harms: [B, C, T] float32, number of harmonics to Nyquist
        r:       float in (0, 1). Fixed rolloff factor.
                   0.90 → crest ~3.2x, smooth sawtooth (recommended)
                   0.95 → crest ~4.6x, brighter
        eps:     singularity threshold

    Returns:
        [B, C, T] float32 — band-limited sawtooth pitch signal
    """
    phase = phase.contiguous()
    n_harms = n_harms.contiguous()
    batch, channels, length = phase.shape
    out = torch.empty_like(phase)

    r_val = float(r)
    grid = lambda meta: (batch * channels, triton.cdiv(length, meta['BLOCK_SIZE']))

    _pcph_geosaw_kernel[grid](
        phase, n_harms, out,
        phase.stride(0), phase.stride(1), phase.stride(2),
        length, channels, r_val, eps
    )
    return out


class FusedGeoSaw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phase, n_harms, r=0.90, eps=1e-6):
        return pcph_geosaw_fwd(phase, n_harms, r=r, eps=eps)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None