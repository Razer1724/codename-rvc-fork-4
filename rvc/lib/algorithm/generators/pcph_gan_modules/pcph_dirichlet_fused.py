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
def _pcph_dirichlet_kernel(
    PHASE,
    N_HARMS,
    OUT,
    stride_b,
    stride_c,
    stride_n,
    N_elements,
    EPS,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elements

    # Calculate pointers for the specific batch/channel
    # This assumes input shape is (Batch, 1, Length)
    PHASE = PHASE + pid * stride_b
    N_HARMS = N_HARMS + pid * stride_b
    OUT = OUT + pid * stride_b

    # Load data once from VRAM into registers
    phase = tl.load(PHASE + offsets, mask=mask)
    n = tl.load(N_HARMS + offsets, mask=mask)

    # Calculate components in registers ( fused ) 
    half_phase = phase * 0.5
    num = tl.cos(half_phase) - tl.cos((n + 0.5) * phase)
    den = 2.0 * tl.sin(half_phase)

    # Safe division
    is_singular = tl.abs(den) < EPS
    result = tl.where(is_singular, 0.0, num / den)

    # Store final result once back to VRAM
    tl.store(OUT + offsets, result, mask=mask)

def pcph_dirichlet_fwd(phase, n_harms, eps=1e-6):
    batch, channels, length = phase.shape
    out = torch.empty_like(phase)

    # Launch grid: (Batch * Channels, Blocks)
    grid = lambda meta: (batch * channels, triton.cdiv(length, meta['BLOCK_SIZE']))

    _pcph_dirichlet_kernel[grid](
        phase, n_harms, out,
        phase.stride(0), phase.stride(1), phase.stride(2),
        length, eps
    )
    return out

class FusedDirichlet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phase, n_harms, eps=1e-6):
        return pcph_dirichlet_fwd(phase, n_harms, eps)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None