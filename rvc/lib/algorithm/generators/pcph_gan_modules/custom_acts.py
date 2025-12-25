import torch
import torch.nn as nn


@torch.compile
def fused_smooth_lrelu(x: torch.Tensor, alpha: float = 0.1, b: float = 0.01) -> torch.Tensor:
    """
    Fused kernel for Smooth Leaky ReLU.
    Matches Leaky ReLU with slope 'alpha' but rounds the corner using Squareplus.
    
    Args:
        x: Input tensor
        alpha: Negative slope (default 0.1)
        b: Smoothness scale (controls the 'blue box' rounding)
    """
    return alpha * x + (1 - alpha) * 0.5 * (x + torch.sqrt(x**2 + 4 * b**2 + 1e-7))



class ParametricSmoothLeakyReLU(nn.Module):
    def __init__(self, num_parameters=1, init_alpha=0.1, init_b=0.1):
        super().__init__()
        # Constraints: 0.0 to 0.5
        initial_alpha_logit = torch.log(torch.tensor(init_alpha / (0.5 - init_alpha)))
        self.alpha_logit = nn.Parameter(torch.full((num_parameters,), initial_alpha_logit))

        # Constraints: 0.001 to 0.5 ( Safety for anti-aliasing )
        initial_b_logit = torch.log(torch.tensor(init_b / (0.5 - init_b)))
        self.b_logit = nn.Parameter(torch.full((num_parameters,), initial_b_logit))

        self.num_parameters = num_parameters

    @torch.compile
    def forward(self, x):
        # constraints using Sigmoid
        # Alpha is bounded [0, 0.5], B is bounded [0.001, 0.5]
        a = torch.sigmoid(self.alpha_logit) * 0.5
        b = torch.sigmoid(self.b_logit) * 0.2 # * 0.5 + 0.001

        if self.num_parameters > 1:
            dims = [1] * x.ndim
            dims[1] = self.num_parameters
            a = a.view(*dims)
            b = b.view(*dims)

        smooth_term = 0.5 * (x + torch.sqrt(x**2 + 4 * b**2 + 1e-7))
        out = a * x + (1 - a) * smooth_term

        # x^2 + 4b^2 is always positive for numerical stability
        return a * x + (1 - a) * 0.5 * (x + torch.sqrt(x**2 + 4 * b**2))