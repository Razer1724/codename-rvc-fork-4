import torch
import torch.nn as nn
import torch.nn.init as init

import math

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def tensor_like(x, y):
    return torch.as_tensor(x, dtype=y.dtype, device=y.device)

def exp(x):
    return torch.exp(x) if torch.is_tensor(x) else math.exp(x)

def sqrt(x):
    return torch.sqrt(x) if torch.is_tensor(x) else math.sqrt(x)

def snake_variance(alpha):
    num = 1 + exp(-8 * alpha ** 2) - 2 * exp(-4 * alpha ** 2)
    return 1 + num / (8 * alpha ** 2)

def snake_second_moment(alpha):
    num = 3 + exp(-8 * alpha ** 2) - 4 * exp(-2 * alpha ** 2)
    return 1 + num / (8 * alpha ** 2)

alpha_max_var = 0.5604532115
max_std = sqrt(snake_variance(alpha_max_var))  # 1.0971017221...

alpha_max_second_moment = 0.65797
max_second_moment_sqrt = sqrt(snake_second_moment(alpha_max_second_moment))  # 1.1787158655

def snake_correction(alpha, kind=None):
    if kind == 'std':
        return sqrt(snake_variance(alpha))
    elif kind == 'max':
        return max_std
    else:
        return kind

def snake_gain(x):
    if x == 'approx':
        return 1
    elif x == 'max':
        return 1 / max_second_moment_sqrt
    else:
        return 1 / sqrt(snake_second_moment(x))

# initialization functions for network parameters preceding a Snake non-linearity
# pass alpha as 'kind' to use the exact second moment
# optionally pass the correction
def snake_kaiming_uniform_(tensor, kind='approx', correction=None, mode='fan_in'):
    fan = init._calculate_correct_fan(tensor, mode)
    correction = snake_correction(kind, correction)
    gain = snake_gain(kind)
    gain = correction ** 2 * gain if correction is not None else gain
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def snake_kaiming_normal_(tensor, kind='approx', correction=None, mode='fan_in'):
    fan = init._calculate_correct_fan(tensor, mode)
    correction = snake_correction(kind, correction)
    gain = snake_gain(kind)
    gain = correction ** 2 * gain if correction is not None else gain
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)

# ---------------------------------------------------------------------------
# Phase-conditioned Snake
# ---------------------------------------------------------------------------
#
# Standard Snake:      out = x + sin(alpha*x)^2 / alpha
# Phase-conditioned:    out = x + sin(alpha*x + beta)^2 / alpha
#
# where beta is supplied not as a raw angle but as a unit-norm (cos_beta,
# sin_beta) pair (see PhaseProjector upstream), expanded via the angle-sum
# identity so the kernel never needs an atan2/trig call on beta itself:
#
#   sin(alpha*x + beta) = sin(alpha*x)*cos_beta + cos(alpha*x)*sin_beta
#
# cos_beta/sin_beta are full (B, C, N)-broadcastable tensors (they vary
# per-timestep, unlike alpha which is per-channel), so they are handled as
# a genuine elementwise input to the kernel, not a per-channel parameter.
#
# The no-beta path is kept completely separate (COND=False) so the common
# case (s2 in ResBlock, or any plain Snake) pays no extra cost and needs
# no phase tensors at all.

try:
    import triton
    import triton.language as tl

    @triton.autotune(
            configs=[
                triton.Config({}, num_warps=4),
                triton.Config({}, num_warps=8),
                triton.Config({}, num_warps=16),
            ],
            key=['N'],
    )
    @triton.jit
    def _snake_fwd_triton(X, OUT, ALPHA, CR, CB, SB,
                          X_stride1, X_stride2, X_stride3,
                          OUT_stride1, OUT_stride2, OUT_stride3,
                          CB_stride1, CB_stride2, CB_stride3,
                          A_stride, C_stride, C, N,
                          CORR: tl.constexpr,
                          COND: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        batch_idx = pid // C
        channel_idx = pid % C
        block_start = tl.program_id(1) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        X = X + batch_idx * X_stride1 + channel_idx * X_stride2
        x = tl.load(X + offsets * X_stride3, mask=offsets < N)
        alpha = tl.load(ALPHA + channel_idx * A_stride)
        ax = alpha * x

        if COND:
            s = tl.sin(ax)
            c = tl.cos(ax)
            CBp = CB + batch_idx * CB_stride1 + channel_idx * CB_stride2
            SBp = SB + batch_idx * CB_stride1 + channel_idx * CB_stride2
            cb = tl.load(CBp + offsets * CB_stride3, mask=offsets < N)
            sb = tl.load(SBp + offsets * CB_stride3, mask=offsets < N)
            q = s * cb + c * sb
        else:
            q = tl.sin(ax)

        out = x + q * q / alpha

        if CORR:
            cr = tl.load(CR + channel_idx * C_stride)
            out = out / cr

        OUT = OUT + batch_idx * OUT_stride1 + channel_idx * OUT_stride2
        tl.store(OUT + offsets * OUT_stride3, out, mask=offsets < N)

    def snake_fwd(x, alpha, cr=None, cos_beta=None, sin_beta=None, out=None):
        if out is None:
            out = torch.empty_like(x)
        B, C, N = x.shape
        cr_ = default(cr, x)
        cond = exists(cos_beta)
        cb_ = default(cos_beta, x)
        sb_ = default(sin_beta, x)
        BLOCK_SIZE = min(triton.next_power_of_2(N), 2 ** 14)
        grid = lambda meta: (B * C, triton.cdiv(N, meta['BLOCK_SIZE']))
        _snake_fwd_triton[grid](x, out, alpha, cr_, cb_, sb_,
                                x.stride(0), x.stride(1), x.stride(2),
                                out.stride(0), out.stride(1), out.stride(2),
                                cb_.stride(0), cb_.stride(1), cb_.stride(2),
                                alpha.stride(0), cr_.stride(0),
                                C, N, exists(cr), cond, BLOCK_SIZE)
        return out

    @triton.autotune(
            configs=[
                triton.Config({}, num_warps=4),
                triton.Config({}, num_warps=8),
                triton.Config({}, num_warps=16),
            ],
            reset_to_zero=['DYDA', 'DYDC'],
            key=['N'],
    )
    @triton.jit
    def _snake_bwd_triton(X, ALPHA, CR, CB, SB, GRAD,
                          DYDX, DYDA, DYDC, DYDCB, DYDSB,
                          X_stride1, X_stride2, X_stride3,
                          GRAD_stride1, GRAD_stride2, GRAD_stride3,
                          DYDX_stride1, DYDX_stride2, DYDX_stride3,
                          CB_stride1, CB_stride2, CB_stride3,
                          DYDCB_stride1, DYDCB_stride2, DYDCB_stride3,
                          DYDA_stride, DYDC_stride,
                          ALPHA_stride, CR_stride, C, N,
                          CORR: tl.constexpr,
                          COND: tl.constexpr,
                          X_NEEDS_GRAD: tl.constexpr,
                          ALPHA_NEEDS_GRAD: tl.constexpr,
                          CR_NEEDS_GRAD: tl.constexpr,
                          BETA_NEEDS_GRAD: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        batch_idx = pid // C
        channel_idx = pid % C
        block_start = tl.program_id(1) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        GRAD = GRAD + batch_idx * GRAD_stride1 + channel_idx * GRAD_stride2
        grad = tl.load(GRAD + offsets * GRAD_stride3, mask=offsets < N, other=0)

        if CORR:
            cr = tl.load(CR + channel_idx * CR_stride)

        # Pre-declare everything computed inside the guarded block below with
        # a dummy value of the correct type. Triton's IR builder requires a
        # name to be defined on *every* path through the function, not just
        # the path where the runtime flag happens to be true -- unlike plain
        # Python, an assignment made only inside `if need_trig:` is not
        # guaranteed to be visible to the later `if X_NEEDS_GRAD:` /
        # `if ALPHA_NEEDS_GRAD:` / `if BETA_NEEDS_GRAD:` blocks. Triton
        # compiles each `if` as a structurally separate block and does NOT
        # prove that e.g. `ALPHA_NEEDS_GRAD` being true implies `need_trig`
        # was also true -- even though that's logically guaranteed by how
        # need_trig is defined, Triton doesn't reason about the relationship
        # between two different constexpr conditions. Every name used later
        # (x, alpha, dydx, q, s, c) must be bound on literally every path,
        # so all of them are pre-declared here, not just the top-level ones.
        #
        # Dummies must also match the dtype the real branch produces, or
        # Triton raises "initial value for `dydx` is of type fp16[...], but
        # the then block redefines it as fp32[...]". tl.sin/tl.cos/tl.load
        # of ALPHA always end up fp32-promoted (loads of `alpha` follow the
        # parameter tensor's own dtype, which is fp32 in this codebase; x
        # follows grad/input dtype), so we match those explicitly with casts
        # rather than inferring from grad's dtype.
        dydx = (grad * 0).to(tl.float32)
        q = (grad * 0).to(tl.float32)
        s = (grad * 0).to(tl.float32)
        c = (grad * 0).to(tl.float32)
        x = (grad * 0).to(grad.dtype)
        alpha = tl.zeros((), dtype=tl.float32)
        out = (grad * 0).to(tl.float32)

        # `out` is never loaded from global memory here -- it is not saved
        # by the forward pass at all anymore. Wherever it's needed
        # (ALPHA_NEEDS_GRAD, CR_NEEDS_GRAD), it's cheaply reconstructed
        # in-register from x/alpha/q, which need_trig below already loads
        # and computes. This costs two extra scalar ops (q*q/alpha, x + ...)
        # against values already resident in registers -- no extra memory
        # traffic, unlike the removed load of a full (B, C, N) tensor.
        need_trig = X_NEEDS_GRAD | ALPHA_NEEDS_GRAD | CR_NEEDS_GRAD | BETA_NEEDS_GRAD
        if need_trig:
            X = X + batch_idx * X_stride1 + channel_idx * X_stride2
            x = tl.load(X + offsets * X_stride3, mask=offsets < N, other=0)
            alpha = tl.load(ALPHA + channel_idx * ALPHA_stride)
            ax = alpha * x
            s = tl.sin(ax)
            c = tl.cos(ax)

            if COND:
                CBp = CB + batch_idx * CB_stride1 + channel_idx * CB_stride2
                SBp = SB + batch_idx * CB_stride1 + channel_idx * CB_stride2
                cb = tl.load(CBp + offsets * CB_stride3, mask=offsets < N, other=0)
                sb = tl.load(SBp + offsets * CB_stride3, mask=offsets < N, other=0)
                q = s * cb + c * sb
                # d(q)/d(alpha*x)
                dqdax = c * cb - s * sb
            else:
                q = s
                dqdax = c

            # d/dx (q^2/alpha) = 2*q*alpha*dqdax/alpha = 2*q*dqdax
            dydx = (1 + 2 * q * dqdax) * grad
            dydx = dydx.to(tl.float32)
            if CORR:
                dydx = dydx / cr

            if ALPHA_NEEDS_GRAD | CR_NEEDS_GRAD:
                # Reconstruct forward's `out` from values already in
                # registers (matches _snake_fwd_triton exactly).
                out = x + q * q / alpha
                if CORR:
                    out = out / cr

        if X_NEEDS_GRAD:
            DYDX = DYDX + batch_idx * DYDX_stride1 + channel_idx * DYDX_stride2
            tl.store(DYDX + offsets * DYDX_stride3, dydx, mask=offsets < N)

        if ALPHA_NEEDS_GRAD:
            # matches original derivation: sum(x*dydx - out*grad)/alpha
            outgrad = tl.sum(out * grad, axis=0)
            dyda = (tl.sum(x * dydx, axis=0) - outgrad) / alpha
            tl.atomic_add(DYDA + channel_idx * DYDA_stride, dyda)

        if CR_NEEDS_GRAD:
            outgrad = tl.sum(out * grad, axis=0)
            dydc = -outgrad / cr
            tl.atomic_add(DYDC + channel_idx * DYDC_stride, dydc)

        if BETA_NEEDS_GRAD:
            # d(q^2/alpha)/d(cos_beta) = 2*q*s/alpha ; d(.)/d(sin_beta) = 2*q*c/alpha
            dydcb = 2 * q * s / alpha * grad
            dydsb = 2 * q * c / alpha * grad
            if CORR:
                dydcb = dydcb / cr
                dydsb = dydsb / cr
            DYDCBp = DYDCB + batch_idx * DYDCB_stride1 + channel_idx * DYDCB_stride2
            DYDSBp = DYDSB + batch_idx * DYDCB_stride1 + channel_idx * DYDCB_stride2
            tl.store(DYDCBp + offsets * DYDCB_stride3, dydcb, mask=offsets < N)
            tl.store(DYDSBp + offsets * DYDCB_stride3, dydsb, mask=offsets < N)

    def snake_bwd(x, alpha, cr, cos_beta, sin_beta, grad,
                  x_needs_grad, alpha_needs_grad, cr_needs_grad, beta_needs_grad):
        # `out` is no longer passed in: it is not saved by the forward pass,
        # and the backward kernel reconstructs it in-register from x/alpha/q
        # (all of which it already loads/computes for dydx) instead of
        # loading a stored (B, C, N) tensor from global memory.
        B, C, N = x.shape
        cond = exists(cos_beta)
        dydx = torch.empty_like(x, dtype=grad.dtype) if x_needs_grad else None
        dyda = torch.zeros_like(alpha, dtype=alpha.dtype) if alpha_needs_grad else None
        dydc = torch.zeros_like(cr, dtype=cr.dtype) if cr_needs_grad else None
        dydcb = torch.empty_like(x, dtype=grad.dtype) if beta_needs_grad else None
        dydsb = torch.empty_like(x, dtype=grad.dtype) if beta_needs_grad else None

        # dydx must always be materialized internally since alpha's backward
        # depends on it even when x itself doesn't need a grad returned.
        if exists(dydx):
            dydx_ = dydx
        elif alpha_needs_grad:
            dydx_ = torch.empty_like(x, dtype=grad.dtype)
        else:
            dydx_ = x.new_empty((1, 1, 1))

        dyda_ = default(dyda, dydc)
        dydc_ = default(dydc, dyda)
        if not exists(dyda_) and not exists(dydc_):
            dyda_ = dydc_ = x.new_empty((1,))

        dydcb_ = default(dydcb, x)
        dydsb_ = default(dydsb, x)

        cr_ = default(cr, x)
        cb_ = default(cos_beta, x)
        sb_ = default(sin_beta, x)

        BLOCK_SIZE = min(triton.next_power_of_2(N), 2 ** 14)
        grid = lambda meta: (B * C, triton.cdiv(N, meta['BLOCK_SIZE']))
        _snake_bwd_triton[grid](x, alpha, cr_, cb_, sb_, grad,
                                dydx_, dyda_, dydc_, dydcb_, dydsb_,
                                x.stride(0), x.stride(1), x.stride(2),
                                grad.stride(0), grad.stride(1), grad.stride(2),
                                dydx_.stride(0), dydx_.stride(1), dydx_.stride(2),
                                cb_.stride(0), cb_.stride(1), cb_.stride(2),
                                dydcb_.stride(0), dydcb_.stride(1), dydcb_.stride(2),
                                dyda_.stride(0), dydc_.stride(0),
                                alpha.stride(0), cr_.stride(0), C, N, exists(cr), cond,
                                x_needs_grad, alpha_needs_grad, cr_needs_grad, beta_needs_grad,
                                BLOCK_SIZE)
        return dydx, dyda, dydc, dydcb, dydsb

except ImportError:
    # fall back to torchscript
    # have to break things up like this for torchscript to fuse properly
    @torch.jit.script
    def snake_fwd_jit(x, alpha):
        return x + torch.sin(alpha[..., None] * x) ** 2 * torch.reciprocal(alpha[..., None])

    @torch.jit.script
    def snake_fwd_c_jit(x, alpha, correction):
        return snake_fwd_jit(x, alpha) * torch.reciprocal(correction[..., None])

    @torch.jit.script
    def snake_fwd_cond_jit(x, alpha, cos_beta, sin_beta):
        ax = alpha[..., None] * x
        q = torch.sin(ax) * cos_beta + torch.cos(ax) * sin_beta
        return x + q * q * torch.reciprocal(alpha[..., None])

    @torch.jit.script
    def snake_fwd_cond_c_jit(x, alpha, cos_beta, sin_beta, correction):
        return snake_fwd_cond_jit(x, alpha, cos_beta, sin_beta) * torch.reciprocal(correction[..., None])

    @torch.jit.script
    def snake_dydx_bwd_jit(x, alpha, grad_output):
        return (torch.sin(2 * alpha[..., None] * x) + 1) * grad_output

    @torch.jit.script
    def snake_dydx_bwd_c_jit(x, alpha, correction, grad_output):
        return torch.reciprocal(correction[..., None]) * snake_dydx_bwd_jit(x, alpha, grad_output)

    @torch.jit.script
    def snake_dydx_bwd_cond_jit(x, alpha, cos_beta, sin_beta, grad_output):
        ax = alpha[..., None] * x
        s = torch.sin(ax)
        c = torch.cos(ax)
        q = s * cos_beta + c * sin_beta
        dqdax = c * cos_beta - s * sin_beta
        return (1 + 2 * q * dqdax) * grad_output

    @torch.jit.script
    def snake_dydx_bwd_cond_c_jit(x, alpha, cos_beta, sin_beta, correction, grad_output):
        return torch.reciprocal(correction[..., None]) * snake_dydx_bwd_cond_jit(
            x, alpha, cos_beta, sin_beta, grad_output)

    @torch.jit.script
    def snake_recompute_out_jit(x, alpha, grad_output):
        # Recomputes forward's `out` (pre-correction) from x/alpha alone,
        # for the plain (unconditioned) case. Cheap relative to storing a
        # full (B, C, N) tensor across the forward/backward boundary.
        q = torch.sin(alpha[..., None] * x)
        return x + q * q * torch.reciprocal(alpha[..., None])

    @torch.jit.script
    def snake_recompute_out_cond_jit(x, alpha, cos_beta, sin_beta, grad_output):
        ax = alpha[..., None] * x
        q = torch.sin(ax) * cos_beta + torch.cos(ax) * sin_beta
        return x + q * q * torch.reciprocal(alpha[..., None])

    @torch.jit.script
    def snake_dyda_bwd_jit(x, dydx, alpha, out, grad_output):
        return torch.reciprocal(alpha) * torch.sum(x * dydx - out * grad_output, dim=(0, 2))

    @torch.jit.script
    def snake_dydc_bwd_jit(out, correction, grad_output):
        return -torch.reciprocal(correction) * torch.sum(out * grad_output, dim=(0, 2))

    @torch.jit.script
    def snake_dydbeta_bwd_jit(x, alpha, cos_beta, sin_beta, grad_output):
        # Returns (dydcb, dydsb). Both are full (B, C, N) elementwise grads
        # (cos_beta/sin_beta vary per-timestep, unlike alpha), not reduced.
        ax = alpha[..., None] * x
        s = torch.sin(ax)
        c = torch.cos(ax)
        q = s * cos_beta + c * sin_beta
        two_q_over_alpha = 2 * q * torch.reciprocal(alpha[..., None])
        dydcb = two_q_over_alpha * s * grad_output
        dydsb = two_q_over_alpha * c * grad_output
        return dydcb, dydsb

    @torch.jit.script
    def snake_dydbeta_bwd_c_jit(x, alpha, cos_beta, sin_beta, correction, grad_output):
        dydcb, dydsb = snake_dydbeta_bwd_jit(x, alpha, cos_beta, sin_beta, grad_output)
        inv_c = torch.reciprocal(correction[..., None])
        return dydcb * inv_c, dydsb * inv_c

    # disable autocast to avoid type promotion
    # to float32 when x is float16
    @torch.cuda.amp.autocast(enabled=False)
    def snake_fwd(x, alpha, cr=None, cos_beta=None, sin_beta=None, out=None):
        cond = exists(cos_beta)
        if cond:
            if cr is None:
                return snake_fwd_cond_jit(x, alpha, cos_beta, sin_beta)
            else:
                return snake_fwd_cond_c_jit(x, alpha, cos_beta, sin_beta, cr)
        else:
            if cr is None:
                return snake_fwd_jit(x, alpha)
            else:
                return snake_fwd_c_jit(x, alpha, cr)

    def snake_bwd(x, alpha, cr, cos_beta, sin_beta, grad_output,
                  x_needs_grad, alpha_needs_grad, cr_needs_grad, beta_needs_grad):
        # `out` is not passed in and is not saved by the forward pass. It's
        # recomputed here, only when actually needed (alpha or correction
        # grad), from x/alpha/(cos_beta, sin_beta), which are already the
        # inputs needed for the trig in dydx above.
        cond = exists(cos_beta)
        dyda, dydc, dydcb, dydsb = None, None, None, None
        dydx = None
        need_dydx_intermediate = x_needs_grad or alpha_needs_grad
        need_out = alpha_needs_grad or cr_needs_grad

        if need_dydx_intermediate:
            if cond:
                if cr is None:
                    dydx = snake_dydx_bwd_cond_jit(x, alpha, cos_beta, sin_beta, grad_output)
                else:
                    dydx = snake_dydx_bwd_cond_c_jit(x, alpha, cos_beta, sin_beta, cr, grad_output)
            else:
                if cr is None:
                    dydx = snake_dydx_bwd_jit(x, alpha, grad_output)
                else:
                    dydx = snake_dydx_bwd_c_jit(x, alpha, cr, grad_output)

        if need_out:
            if cond:
                out = snake_recompute_out_cond_jit(x, alpha, cos_beta, sin_beta, grad_output)
            else:
                out = snake_recompute_out_jit(x, alpha, grad_output)
            if cr is not None:
                out = out * torch.reciprocal(cr[..., None])

        if alpha_needs_grad:
            dyda = snake_dyda_bwd_jit(x, dydx, alpha, out, grad_output)
        if cr_needs_grad:
            dydc = snake_dydc_bwd_jit(out, cr, grad_output)
        if beta_needs_grad:
            if cr is None:
                dydcb, dydsb = snake_dydbeta_bwd_jit(x, alpha, cos_beta, sin_beta, grad_output)
            else:
                dydcb, dydsb = snake_dydbeta_bwd_c_jit(x, alpha, cos_beta, sin_beta, cr, grad_output)

        return (dydx if x_needs_grad else None), dyda, dydc, dydcb, dydsb

class SnakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, correction=None, cos_beta=None, sin_beta=None):
        out = snake_fwd(x, alpha, correction, cos_beta, sin_beta)
        # `out` is intentionally NOT saved here. Backward only ever needs it
        # for dyda (x*dydx - out*grad) and dydc (-out*grad/cr), and in both
        # cases every value needed to reconstruct out -- x, alpha, q -- is
        # already reloaded/recomputed in the backward kernel for dydx. So
        # storing a whole extra (B, C, N) tensor across the autograd
        # boundary just to re-read it once in backward is redundant; it's
        # cheaper to recompute out from values already resident wherever
        # backward needs it, at zero (Triton) or minimal (fallback) cost.
        ctx.save_for_backward(x, alpha, correction, cos_beta, sin_beta)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, cr, cos_beta, sin_beta = ctx.saved_tensors
        needs = ctx.needs_input_grad  # (x, alpha, correction, cos_beta, sin_beta)
        beta_needs_grad = needs[3] or needs[4]
        dydx, dyda, dydc, dydcb, dydsb = snake_bwd(
            x, alpha, cr, cos_beta, sin_beta, grad_output,
            needs[0], needs[1], needs[2], beta_needs_grad,
        )
        return (
            dydx,
            dyda,
            dydc,
            dydcb if needs[3] else None,
            dydsb if needs[4] else None,
        )

class Snake(nn.Module):
    def __init__(self, num_channels, init=0.5, correction=None):
        super().__init__()
        if init == 'periodic':
            # "for tasks with expected periodicity, larger a, 
            # usually from 5 to 50 tend to work well"
            # => use a gamma distribution with median ~5 and a heavy right tail
            gamma = torch.distributions.Gamma(concentration=1.5, rate=0.1)
            self.alpha = nn.Parameter(gamma.sample((num_channels,)))
        elif callable(init):  # e.g. torch.randn
            self.alpha = nn.Parameter(init(num_channels) * torch.ones(num_channels))
        else:  # assume init is a constant
            self.alpha = nn.Parameter(init * torch.ones(num_channels))
        self.correction = correction

    def forward(self, x, beta=None):
        """
        beta, if provided, is a (cos_beta, sin_beta) tuple of unit-norm,
        full (B, C, N)-shaped tensors (see PhaseProjector upstream). When
        provided, Snake's argument is phase-shifted: sin(alpha*x + beta),
        expanded via the angle-sum identity so no trig call on beta is
        ever needed here -- the caller is responsible for keeping
        (cos_beta, sin_beta) unit-norm.

        When beta is None (the common case, e.g. the second Snake in each
        ResBlock layer), this is identical to the original unconditioned
        Snake -- no extra tensors are read, no extra branch cost paid in
        the fused kernel.
        """
        correction = snake_correction(self.alpha, kind=self.correction)
        alpha = self.alpha.expand(x.size(1))
        if correction is not None:
            correction = tensor_like(correction, self.alpha)
            correction = correction.expand(x.size(1))

        cos_beta, sin_beta = (None, None) if beta is None else beta
        return SnakeFunction.apply(x, alpha, correction, cos_beta, sin_beta)
