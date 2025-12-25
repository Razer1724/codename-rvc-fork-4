import torch
import torch.nn as nn
import torch.nn.init as init
import math

# --- Helpers ---

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

def snake_variance(alpha, beta=None):
    b = beta if beta is not None else alpha
    # Variance approx for y = x + 1/b * sin^2(ax)
    num = 1 + exp(-8 * alpha ** 2) - 2 * exp(-4 * alpha ** 2)
    return 1 + num / (8 * b ** 2 + 1e-9)

def snake_second_moment(alpha, beta=None):
    b = beta if beta is not None else alpha
    num = 3 + exp(-8 * alpha ** 2) - 4 * exp(-2 * alpha ** 2)
    return 1 + num / (8 * b ** 2 + 1e-9)

# Standard init constants based on alpha=0.5, beta=1.0
alpha_max_var = 0.5604532115
max_std = sqrt(snake_variance(alpha_max_var, 1.0))

alpha_max_second_moment = 0.65797
max_second_moment_sqrt = sqrt(snake_second_moment(alpha_max_second_moment, 1.0))

def snake_correction(alpha, beta=None, kind=None):
    if kind == 'std':
        return sqrt(snake_variance(alpha, beta))
    elif kind == 'max':
        return max_std
    elif isinstance(kind, (int, float, torch.Tensor)):
        return kind
    return None

def snake_gain(alpha, beta=None, kind='approx'):
    if kind == 'approx':
        return 1.0
    elif kind == 'max':
        return 1 / max_second_moment_sqrt
    else:
        # Expecting kind to be a value or alpha itself for exact second moment
        return 1 / sqrt(snake_second_moment(alpha, beta))

def snake_kaiming_uniform_(tensor, alpha=0.5, beta=1.0, kind='approx', correction=None, mode='fan_in'):
    fan = init._calculate_correct_fan(tensor, mode)
    corr_val = snake_correction(alpha, beta, correction)
    gain = snake_gain(alpha, beta, kind)
    if corr_val is not None and isinstance(corr_val, (int, float)):
        gain = (corr_val ** 2) * gain
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def snake_kaiming_normal_(tensor, alpha=0.5, beta=1.0, kind='approx', correction=None, mode='fan_in'):
    fan = init._calculate_correct_fan(tensor, mode)
    corr_val = snake_correction(alpha, beta, correction)
    gain = snake_gain(alpha, beta, kind)
    if corr_val is not None and isinstance(corr_val, (int, float)):
        gain = (corr_val ** 2) * gain
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)

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
    def _snake_fwd_triton(
        X, OUT, ALPHA, BETA, CR, 
        X_stride1, X_stride2, X_stride3, 
        OUT_stride1, OUT_stride2, OUT_stride3,
        A_stride, B_stride, C_stride, 
        C, N, CORR: tl.constexpr, LOG_SCALE: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        batch_idx = pid // C
        channel_idx = pid % C
        block_start = tl.program_id(1) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        mask = offsets < N
        
        # Load parameters (1D tensors indexed by channel)
        alpha = tl.load(ALPHA + channel_idx * A_stride)
        beta = tl.load(BETA + channel_idx * B_stride)

        # Apply log_scale if enabled
        if LOG_SCALE:
            alpha = tl.exp(alpha)
            beta = tl.exp(beta)

        # Load input (3D tensor)
        X_ptr = X + batch_idx * X_stride1 + channel_idx * X_stride2 + offsets * X_stride3
        x = tl.load(X_ptr, mask=mask)
        
        # y = x + (1/beta) * sin^2(alpha * x)
        sinax = tl.sin(alpha * x)
        out = x + (sinax * sinax) / (beta + 1e-9)

        if CORR:
            cr = tl.load(CR + channel_idx * C_stride)
            out = out / cr

        OUT_ptr = OUT + batch_idx * OUT_stride1 + channel_idx * OUT_stride2 + offsets * OUT_stride3
        tl.store(OUT_ptr, out, mask=mask)

    def snake_fwd(x, alpha, beta, cr=None, out=None, log_scale=False):
        if out is None:
            out = torch.empty_like(x)
        B, C, N = x.shape
        # Use dummy 1D tensor for strides if cr is None
        cr_ptr = default(cr, alpha) 

        BLOCK_SIZE = min(triton.next_power_of_2(N), 2 ** 14)
        grid = lambda meta: (B * C, triton.cdiv(N, meta['BLOCK_SIZE']))

        _snake_fwd_triton[grid](
            x, out, alpha, beta, cr_ptr,
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            alpha.stride(0), beta.stride(0), cr_ptr.stride(0),
            C, N, exists(cr), log_scale, BLOCK_SIZE
        )
        return out

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
    def _snake_bwd_triton(
        X, OUT, ALPHA, BETA, CR, GRAD,
        DYDX, DYDA, DYDB, DYDC,
        X_stride1, X_stride2, X_stride3,
        OUT_stride1, OUT_stride2, OUT_stride3,
        GRAD_stride1, GRAD_stride2, GRAD_stride3,
        DYDX_stride1, DYDX_stride2, DYDX_stride3,
        DYDA_stride, DYDB_stride, DYDC_stride,
        ALPHA_stride, BETA_stride, CR_stride, 
        C, N, CORR: tl.constexpr, LOG_SCALE: tl.constexpr,
        X_NEEDS_GRAD: tl.constexpr, ALPHA_NEEDS_GRAD: tl.constexpr,
        BETA_NEEDS_GRAD: tl.constexpr, CR_NEEDS_GRAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        batch_idx = pid // C
        channel_idx = pid % C
        block_start = tl.program_id(1) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        GRAD_ptr = GRAD + batch_idx * GRAD_stride1 + channel_idx * GRAD_stride2 + offsets * GRAD_stride3
        grad = tl.load(GRAD_ptr, mask=mask, other=0.0)

        # Handle correction scaling for the incoming gradient
        if CORR:
            cr = tl.load(CR + channel_idx * CR_stride)
            grad_internal = grad / cr
        else:
            grad_internal = grad

        alpha_raw = tl.load(ALPHA + channel_idx * ALPHA_stride)
        beta_raw = tl.load(BETA + channel_idx * BETA_stride)

        # Apply log_scale if enabled
        if LOG_SCALE:
            alpha = tl.exp(alpha_raw)
            beta = tl.exp(beta_raw)
        else:
            alpha = alpha_raw
            beta = beta_raw

        beta_safe = beta + 1e-9
        X_ptr = X + batch_idx * X_stride1 + channel_idx * X_stride2 + offsets * X_stride3
        x = tl.load(X_ptr, mask=mask, other=0.0)

        # Derivatives
        ax = alpha * x
        sinax = tl.sin(ax)
        sin2ax = tl.sin(2.0 * ax)

        if X_NEEDS_GRAD:
            # dy/dx = 1 + (alpha / beta_safe) * sin(2ax)
            dydx = grad_internal * (1.0 + (alpha / beta_safe) * sin2ax)
            DYDX_ptr = DYDX + batch_idx * DYDX_stride1 + channel_idx * DYDX_stride2 + offsets * DYDX_stride3
            tl.store(DYDX_ptr, dydx, mask=mask)

        if ALPHA_NEEDS_GRAD:
            # dy/da = (x / beta_safe) * sin(2ax)
            dyda_val = tl.sum(grad_internal * (x / beta_safe) * sin2ax, axis=0)

            # Chain rule: dL/d(raw) = dL/da * da/d(raw) = dL/da * exp(raw) = dL/da * a
            if LOG_SCALE:
                dyda_val = dyda_val * alpha

            tl.atomic_add(DYDA + channel_idx * DYDA_stride, dyda_val)

        if BETA_NEEDS_GRAD:
            # dy/db = -sin^2(ax) / (beta_safe)^2
            dydb_val = tl.sum(grad_internal * (-(sinax * sinax) / (beta_safe * beta_safe)), axis=0)

            # Chain rule for beta
            if LOG_SCALE:
                dydb_val = dydb_val * beta

            tl.atomic_add(DYDB + channel_idx * DYDB_stride, dydb_val)

        if CR_NEEDS_GRAD:
            # dL/dcr = dL/dy_corr * (-y_corr / cr)
            OUT_ptr = OUT + batch_idx * OUT_stride1 + channel_idx * OUT_stride2 + offsets * OUT_stride3
            out_final = tl.load(OUT_ptr, mask=mask, other=0.0)
            dydc_val = tl.sum(-out_final * grad_internal, axis=0)
            tl.atomic_add(DYDC + channel_idx * DYDC_stride, dydc_val)

    def snake_bwd(x, alpha, beta, cr, out, grad, 
                  x_needs_grad, alpha_needs_grad, beta_needs_grad, cr_needs_grad, log_scale=False):
        B, C, N = x.shape

        # Allocated required gradient tensors
        dydx = torch.empty_like(x) if x_needs_grad else None
        dyda = torch.zeros_like(alpha) if alpha_needs_grad else None
        dydb = torch.zeros_like(beta) if beta_needs_grad else None
        dydc = torch.zeros_like(cr) if (cr_needs_grad and exists(cr)) else None
        
        # Prepare safety dummies for Triton call to avoid None.stride()
        dyda_ = default(dyda, alpha.new_empty((1,)))
        dydb_ = default(dydb, beta.new_empty((1,)))
        dydc_ = default(dydc, alpha.new_empty((1,)))
        cr_ptr = default(cr, alpha) 
        BLOCK_SIZE = min(triton.next_power_of_2(N), 2 ** 14)
        grid = lambda meta: (B * C, triton.cdiv(N, meta['BLOCK_SIZE']))

        _snake_bwd_triton[grid](
            x, out, alpha, beta, cr_ptr, grad,
            dydx, dyda_, dydb_, dydc_,
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            grad.stride(0), grad.stride(1), grad.stride(2),
            dydx.stride(0) if x_needs_grad else 0, 
            dydx.stride(1) if x_needs_grad else 0, 
            dydx.stride(2) if x_needs_grad else 0,
            dyda_.stride(0), dydb_.stride(0), dydc_.stride(0),
            alpha.stride(0), beta.stride(0), cr_ptr.stride(0),
            C, N, exists(cr), log_scale,
            x_needs_grad, alpha_needs_grad, beta_needs_grad, cr_needs_grad,
            BLOCK_SIZE
        )
        return dydx, dyda, dydb, dydc

except ImportError:
    # TorchScript Fallback
    @torch.jit.script
    def snake_fwd_jit(x, alpha, beta):
        return x + (torch.sin(alpha * x) ** 2) / beta

    @torch.jit.script
    def snake_fwd_c_jit(x, alpha, beta, correction):
        return (x + (torch.sin(alpha * x) ** 2) / beta) / correction

    # Backward Gradients
    @torch.jit.script
    def snake_bwd_jit(x, alpha, beta, cr, out, grad, 
                      x_needs_grad: bool, a_needs_grad: bool, b_needs_grad: bool, c_needs_grad: bool):
        # correction is (1, C, 1) or None
        g_int = grad / cr if cr is not None else grad
        
        dydx = (1.0 + (alpha / beta) * torch.sin(2.0 * alpha * x)) * g_int if x_needs_grad else None
        dyda = torch.sum(g_int * (x / beta) * torch.sin(2.0 * alpha * x), dim=(0, 2)) if a_needs_grad else None
        dydb = torch.sum(g_int * (-(torch.sin(alpha * x)**2) / (beta**2)), dim=(0, 2)) if b_needs_grad else None
        dydc = torch.sum(-out * g_int, dim=(0, 2)) if (c_needs_grad and cr is not None) else None
        
        return dydx, dyda, dydb, dydc

    @torch.cuda.amp.autocast(enabled=False)
    def snake_fwd(x, alpha, beta, cr=None, log_scale=False):
        if log_scale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        a, b = alpha.view(1, -1, 1), beta.view(1, -1, 1)
        if cr is None:
            return snake_fwd_jit(x, a, b)
        return snake_fwd_c_jit(x, a, b, cr.view(1, -1, 1))

    def snake_bwd(x, alpha, beta, cr, out, grad, x_ng, a_ng, b_ng, c_ng, log_scale=False):
        alpha_raw, beta_raw = alpha, beta
        if log_scale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        a, b = alpha.view(1, -1, 1), beta.view(1, -1, 1)
        c = cr.view(1, -1, 1) if cr is not None else None
        dydx, dyda, dydb, dydc = snake_bwd_jit(x, a, b, c, out, grad, x_ng, a_ng, b_ng, c_ng)

        if log_scale:
            if dyda is not None: dyda = dyda * alpha_raw.exp()
            if dydb is not None: dydb = dydb * beta_raw.exp()
            
        return dydx, dyda, dydb, dydc

# --- Autograd and Module ---

class SnakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, beta, correction=None, log_scale=False):
        out = snake_fwd(x, alpha, beta, correction, log_scale=log_scale)
        ctx.save_for_backward(x, alpha, beta, correction, out)
        ctx.log_scale = log_scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta, cr, out = ctx.saved_tensors
        # ctx.needs_input_grad is (x, alpha, beta, correction, log_scale)
        # Note: log_scale is not a tensor, so it doesn't need grad
        dydx, dyda, dydb, dydc = snake_bwd(
            x, alpha, beta, cr, out, grad_output, 
            ctx.needs_input_grad[0], 
            ctx.needs_input_grad[1], 
            ctx.needs_input_grad[2], 
            ctx.needs_input_grad[3],
            log_scale=ctx.log_scale
        )
        return dydx, dyda, dydb, dydc, None

class SnakeBeta(nn.Module):
    def __init__(self, num_channels, init='periodic', beta_init=1.0, correction=None, log_scale=False):
        super().__init__()
        self.log_scale = log_scale
        self.correction = correction

        # Alpha ( frequency )
        if init == 'periodic':
            gamma = torch.distributions.Gamma(concentration=1.5, rate=0.1)
            alpha_val = gamma.sample((num_channels,))
        elif callable(init):
            alpha_val = init(num_channels)
        else:
            alpha_val = torch.full((num_channels,), float(init))

        # Beta (Magnitude)
        if beta_init == 'periodic':
            beta_val = torch.ones(num_channels)
        elif callable(beta_init):
            beta_val = beta_init(num_channels)
        else:
            beta_val = torch.full((num_channels,), float(beta_init))

        # Apply log if using log_scale to ensure initial state matches desired values
        if self.log_scale:
            # Add small epsilon to avoid log(0) if init generated exact zeros (though Gamma/Ones won't)
            self.alpha = nn.Parameter(torch.log(alpha_val + 1e-9))
            self.beta = nn.Parameter(torch.log(beta_val + 1e-9))
        else:
            self.alpha = nn.Parameter(alpha_val)
            self.beta = nn.Parameter(beta_val)

    def forward(self, x):
        # x shape: (B, C, N)
        # Parameters are (C,)

        curr_alpha = self.alpha.exp() if self.log_scale else self.alpha
        curr_beta = self.beta.exp() if self.log_scale else self.beta

        corr_tensor = None
        if self.correction is not None:
            cv = snake_correction(curr_alpha, curr_beta, self.correction)
            if isinstance(cv, torch.Tensor):
                corr_tensor = cv
            elif isinstance(cv, (int, float)):
                corr_tensor = torch.full_like(self.alpha, float(cv))
        
        return SnakeFunction.apply(x, self.alpha, self.beta, corr_tensor, self.log_scale)

    def __repr__(self):
        return f'Snake(channels={self.alpha.shape[0]}, correction={self.correction}, log_scale={self.log_scale})'