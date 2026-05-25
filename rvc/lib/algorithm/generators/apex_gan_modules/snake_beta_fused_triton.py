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


def snake_beta_variance(alpha, beta):
    num = 1 + exp(-8 * alpha ** 2) - 2 * exp(-4 * alpha ** 2)
    return 1 + num / (8 * beta ** 2)

def snake_beta_second_moment(alpha, beta):
    num = 3 + exp(-8 * alpha ** 2) - 4 * exp(-2 * alpha ** 2)
    return 1 + num / (8 * beta ** 2)

def fwd_snake_correction(alpha, beta):
    # Supporting only the 'std' correction mode by design
    return sqrt(snake_beta_variance(alpha, beta))


def snake_gain(alpha='approx', beta='approx'):
    """
    Gain for Kaiming-style initialisations.

    alpha/beta can be:
      'approx' – ignores the sin² term (gain = 1)
      <float>  – use the exact second moment at that alpha value
    """
    if alpha == 'approx' and beta == 'approx':
        return 1
    elif isinstance(alpha, float) and isinstance(beta, float):
        return 1 / sqrt(snake_beta_second_moment(alpha, beta))
    else:
        raise ValueError(" Both alpha and beta passed to snake_kaiming inits ( normal / uniform ) must be either floats or 'approx' strings.")


def snake_kaiming_uniform_(tensor, init_value='approx', mode='fan_in'):
    alpha = beta = init_value
    fan   = init._calculate_correct_fan(tensor, mode)
    gain  = snake_gain(alpha, beta)
    std   = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def snake_kaiming_normal_(tensor, init_value='approx', mode='fan_in'):
    alpha = beta = init_value
    fan   = init._calculate_correct_fan(tensor, mode)
    gain  = snake_gain(alpha, beta)
    std   = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)




import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Forward kernel
# out = x + sin²(alpha * x) / beta
# optionally divided by correction cr
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _snake_beta_fwd_triton(X, OUT, ALPHA, BETA, CR,
                      X_stride1, X_stride2, X_stride3,
                      OUT_stride1, OUT_stride2, OUT_stride3,
                      A_stride, B_stride, C_stride, C, N,
                      CORR: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx   = pid // C
    channel_idx = pid % C
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets     = block_start + tl.arange(0, BLOCK_SIZE)

    X = X + batch_idx * X_stride1 + channel_idx * X_stride2
    x = tl.load(X + offsets * X_stride3, mask=offsets < N)

    alpha = tl.load(ALPHA + channel_idx * A_stride)
    beta  = tl.load(BETA  + channel_idx * B_stride)

    sinax = tl.sin(alpha * x)
    out   = x + sinax * sinax / (beta + 1e-9)

    if CORR:
        cr  = tl.load(CR + channel_idx * C_stride)
        out = out / cr

    OUT = OUT + batch_idx * OUT_stride1 + channel_idx * OUT_stride2
    tl.store(OUT + offsets * OUT_stride3, out, mask=offsets < N)

def snake_beta_fwd(x, alpha, beta, cr=None, out=None):
    if out is None:
        out = torch.empty_like(x)
    B, C, N = x.shape
    cr_ = default(cr, x)
    BLOCK_SIZE = min(triton.next_power_of_2(N), 2 ** 14)
    grid = lambda meta: (B * C, triton.cdiv(N, meta['BLOCK_SIZE']))
    _snake_beta_fwd_triton[grid](
        x, out, alpha, beta, cr_,
        x.stride(0),   x.stride(1),   x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        alpha.stride(0), beta.stride(0), cr_.stride(0),
        C, N, exists(cr), BLOCK_SIZE,
    )
    return out

# ---------------------------------------------------------------------------
# Backward kernel
#
# out     = x + sin²(ax) / b          [no correction]
# out_c   = out / cr                   [with correction]
#
# dydx    = (1 + (a/b)*sin(2ax)) * grad   [/cr if correction]
# dyda    = sum( sin(2ax)*x/b  * grad )   [/cr if correction]
# dydb    = sum(-sin²(ax)/b²   * grad )   [/cr if correction]
# dydc    = -sum(out_c * grad) / cr
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    reset_to_zero=['DYDA', 'DYDB', 'DYDC'],
    key=['N'],
)
@triton.jit
def _snake_beta_bwd_triton(X, OUT, ALPHA, BETA, CR, GRAD,
                      DYDX, DYDA, DYDB, DYDC,
                      X_stride1,    X_stride2,    X_stride3,
                      OUT_stride1,  OUT_stride2,  OUT_stride3,
                      GRAD_stride1, GRAD_stride2, GRAD_stride3,
                      DYDX_stride1, DYDX_stride2, DYDX_stride3,
                      DYDA_stride, DYDB_stride, DYDC_stride,
                      ALPHA_stride, BETA_stride, CR_stride,
                      C, N,
                      CORR:             tl.constexpr,
                      X_NEEDS_GRAD:     tl.constexpr,
                      ALPHA_NEEDS_GRAD: tl.constexpr,
                      BETA_NEEDS_GRAD:  tl.constexpr,
                      CR_NEEDS_GRAD:    tl.constexpr,
                      # Fix 1a: explicit constexprs so Triton sees them as
                      # compile-time constants, rather than a Python-level OR
                      # expression that is non-standard in Triton JIT.
                      NEED_XAB:         tl.constexpr,   # x + alpha + beta loads
                      NEED_SINAX:       tl.constexpr,   # sin(ax) for dydb only
                      BLOCK_SIZE:       tl.constexpr):

    pid         = tl.program_id(0)
    batch_idx   = pid // C
    channel_idx = pid % C
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets     = block_start + tl.arange(0, BLOCK_SIZE)

    GRAD = GRAD + batch_idx * GRAD_stride1 + channel_idx * GRAD_stride2
    grad = tl.load(GRAD + offsets * GRAD_stride3, mask=offsets < N, other=0)

    # Load correction once if needed
    cr = tl.zeros([1], dtype=grad.dtype)   # dummy; overwritten below
    if CORR:
        cr = tl.load(CR + channel_idx * CR_stride)

    # Fix 1b: each load group guarded by its own explicit constexpr,
    # so dead loads are provably eliminated at compile time.
    if NEED_XAB:
        X_ptr  = X + batch_idx * X_stride1 + channel_idx * X_stride2
        x      = tl.load(X_ptr + offsets * X_stride3, mask=offsets < N, other=0)
        alpha  = tl.load(ALPHA + channel_idx * ALPHA_stride)
        beta   = tl.load(BETA  + channel_idx * BETA_stride)
        ax     = alpha * x
        sin2ax = tl.sin(2.0 * ax)
    # Fix 1c: sinax is only needed for dydb — compute it separately so it
    # is not wasted when only dydx or dyda is requested.
    if NEED_SINAX:
        if not NEED_XAB:   # recompute ax if x/alpha/beta weren't loaded above
            X_ptr  = X + batch_idx * X_stride1 + channel_idx * X_stride2
            x      = tl.load(X_ptr + offsets * X_stride3, mask=offsets < N, other=0)
            alpha  = tl.load(ALPHA + channel_idx * ALPHA_stride)
            beta   = tl.load(BETA  + channel_idx * BETA_stride)
            ax     = alpha * x
        sinax = tl.sin(ax)

    # ── dydx ──────────────────────────────────────────────────────────
    if X_NEEDS_GRAD:
        dydx = (1.0 + (alpha / (beta + 1e-9)) * sin2ax) * grad
        if CORR:
            dydx = dydx / cr
        DYDX_ptr = DYDX + batch_idx * DYDX_stride1 + channel_idx * DYDX_stride2
        tl.store(DYDX_ptr + offsets * DYDX_stride3, dydx, mask=offsets < N)

    # ── dyda ──────────────────────────────────────────────────────────
    if ALPHA_NEEDS_GRAD:
        dyda = tl.sum(sin2ax * x / (beta + 1e-9) * grad, axis=0)
        if CORR:
            dyda = dyda / cr
        tl.atomic_add(DYDA + channel_idx * DYDA_stride, dyda)

    # ── dydb ──────────────────────────────────────────────────────────
    if BETA_NEEDS_GRAD:
        beta_s = beta + 1e-9
        dydb = tl.sum(-(sinax * sinax) / (beta_s * beta_s) * grad, axis=0)
        if CORR:
            dydb = dydb / cr
        tl.atomic_add(DYDB + channel_idx * DYDB_stride, dydb)

    # ── dydc ──────────────────────────────────────────────────────────
    if CR_NEEDS_GRAD:
        OUT_ptr  = OUT + batch_idx * OUT_stride1 + channel_idx * OUT_stride2
        out      = tl.load(OUT_ptr + offsets * OUT_stride3, mask=offsets < N, other=0)
        outgrad  = tl.sum(out * grad, axis=0)
        dydc     = -outgrad / cr
        tl.atomic_add(DYDC + channel_idx * DYDC_stride, dydc)

def snake_beta_bwd(x, alpha, beta, cr, out, grad,
              x_needs_grad, alpha_needs_grad, beta_needs_grad, cr_needs_grad):
    B, C, N = x.shape

    dydx = torch.empty_like(x,     dtype=grad.dtype)  if x_needs_grad     else None
    dyda = torch.zeros_like(alpha, dtype=alpha.dtype) if alpha_needs_grad else None
    dydb = torch.zeros_like(beta,  dtype=beta.dtype)  if beta_needs_grad  else None
    dydc = torch.zeros_like(cr,    dtype=cr.dtype)    if cr_needs_grad    else None

    # Triton needs non-None pointers — provide safe dummies
    dyda_ = default(dyda, default(dydb, x.new_empty((C,))))
    dydb_ = default(dydb, default(dyda, x.new_empty((C,))))
    dydc_ = default(dydc, x.new_empty((C,)))
    cr_   = default(cr,   x.new_empty((C,)))

    # dydx must be allocated even as dummy (stride taken)
    dydx_ = dydx if dydx is not None else torch.empty_like(x)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 2 ** 14)
    grid = lambda meta: (B * C, triton.cdiv(N, meta['BLOCK_SIZE']))

    need_xab   = x_needs_grad or alpha_needs_grad or beta_needs_grad
    need_sinax = beta_needs_grad  # sin(ax) exclusively for dydb
    _snake_beta_bwd_triton[grid](
        x, out, alpha, beta, cr_, grad,
        dydx_, dyda_, dydb_, dydc_,
        x.stride(0),    x.stride(1),    x.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        grad.stride(0), grad.stride(1), grad.stride(2),
        dydx_.stride(0), dydx_.stride(1), dydx_.stride(2),
        dyda_.stride(0), dydb_.stride(0), dydc_.stride(0),
        alpha.stride(0), beta.stride(0),  cr_.stride(0),
        C, N,
        exists(cr), x_needs_grad, alpha_needs_grad, beta_needs_grad, cr_needs_grad,
        need_xab, need_sinax,
        BLOCK_SIZE,
    )
    return dydx, dyda, dydb, dydc


# ---------------------------------------------------------------------------
# Autograd Function
# Signature: (x, alpha, beta, correction)
# ---------------------------------------------------------------------------

class SnakeBetaFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha, beta, correction=None):
        out = snake_beta_fwd(x, alpha, beta, correction)
        ctx.save_for_backward(x, alpha, beta, out)
        # Fix 3: save_for_backward only accepts Tensors; store the optional
        # correction tensor (or None) as a plain attribute instead.
        ctx.correction = correction
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta, out = ctx.saved_tensors
        cr = ctx.correction
        x_ng, a_ng, b_ng, c_ng = ctx.needs_input_grad
        dydx, dyda, dydb, dydc = snake_beta_bwd(
            x, alpha, beta, cr, out, grad_output,
            x_ng, a_ng, b_ng, c_ng,
        )
        return dydx, dyda, dydb, dydc


# ---------------------------------------------------------------------------
# SnakeBeta module
# ---------------------------------------------------------------------------

class SnakeBeta(nn.Module):
    def __init__(
        self,
        num_channels: int,
        init=1.0,
        synced_start: bool = False, # Only useful for mean/gamma inits
        correction: bool = True
    ):
        super().__init__()
        self.correction = correction

        # ── sample initial values ───────────────────────────────────────────────
        def _get_init_vals(init):
            if init == 'periodic':
                gamma = torch.distributions.Gamma(concentration=1.5, rate=0.1)
                return gamma.sample((num_channels,))
            elif init == 'mean_1':
                gamma = torch.distributions.Gamma(concentration=1.5, rate=1.5)
                return gamma.sample((num_channels,))
            elif init == 'mean_2':
                gamma = torch.distributions.Gamma(concentration=1.5, rate=0.75)
                return gamma.sample((num_channels,))
            elif init == 'mean_3':
                gamma = torch.distributions.Gamma(concentration=1.5, rate=0.5)
                return gamma.sample((num_channels,))
            else:
                # For flat init values
                return float(init) * torch.ones(num_channels)

        # ── sample and store initial values / parameters ────────────────────────
        if synced_start: # Both start from the exact same distribution to ensure stability ( I guess? )
            alpha_vals = _get_init_vals(init) # A

            self.alpha = nn.Parameter(torch.log(alpha_vals.clamp(min=1e-4)))         # A
            self.beta = nn.Parameter(torch.log(alpha_vals.clamp(min=1e-4).clone()))  # A's clone
        else:
            alpha_vals = _get_init_vals(init) # A 
            beta_vals = _get_init_vals(init)  # B

            self.alpha = nn.Parameter(torch.log(alpha_vals.clamp(min=1e-4)))  # A
            self.beta = nn.Parameter(torch.log(beta_vals.clamp(min=1e-4)))    # B


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)

        correction = None
        if self.correction:
            correction = fwd_snake_correction(alpha, beta)
            #correction = fwd_snake_correction(alpha, beta).detach()

        alpha = alpha.expand(x.size(1))
        beta = beta.expand(x.size(1))

        if correction is not None:
            correction = tensor_like(correction, alpha).expand(x.size(1))

        return SnakeBetaFunction.apply(x, alpha, beta, correction)
