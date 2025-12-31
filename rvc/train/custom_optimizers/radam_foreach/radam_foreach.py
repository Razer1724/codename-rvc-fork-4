import torch
from torch.optim import Optimizer
import numpy as np

class RAdamForEach(Optimizer):
    """
    RAdam optimizer with Foreach implementation for speed.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, decouple_wd=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                        decouple_wd=decouple_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            steps = []

            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            decouple_wd = group['decouple_wd']

            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('RAdamForEach does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]

                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'] += 1
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    steps.append(state['step'])

            if not params:
                continue

            if weight_decay != 0:
                if decouple_wd:
                    wd_scalar = 1.0 - lr * weight_decay
                    torch._foreach_mul_(params, wd_scalar)
                else:
                    torch._foreach_add_(grads, params, alpha=weight_decay)

            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

            steps_arr = np.array(steps, dtype=np.int32)

            bias_correction1 = 1 - beta1 ** steps_arr
            bias_correction2_sq = 1 - beta2 ** steps_arr

            step_size_unrect = -lr / bias_correction1

            rho_inf = 2 / (1 - beta2) - 1
            rho_t = rho_inf - 2 * steps_arr * (beta2 ** steps_arr) / (bias_correction2_sq)

            rect_mask = rho_t > 5

            rect_coef = np.sqrt(
                ((rho_t - 4) * (rho_t - 2) * rho_inf) / 
                ((rho_inf - 4) * (rho_inf - 2) * rho_t),
                where=rect_mask
            )

            denom = torch._foreach_sqrt(exp_avg_sqs)

            bc2_sqrt = np.sqrt(bias_correction2_sq)
            torch._foreach_div_(denom, bc2_sqrt.tolist())
            torch._foreach_add_(denom, eps)

            rect_scalar = step_size_unrect * np.where(rect_mask, rect_coef, 0.0)
            unrect_scalar = np.where(~rect_mask, step_size_unrect, 0.0)

            if np.any(rect_mask):
                torch._foreach_addcdiv_(params, exp_avgs, denom, scalars=rect_scalar.tolist())

            if np.any(~rect_mask):
                scaled_exp_avgs = torch._foreach_mul(exp_avgs, unrect_scalar.tolist())
                torch._foreach_add_(params, scaled_exp_avgs)

        return loss