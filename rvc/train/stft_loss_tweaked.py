import torch
import scipy.signal
import numpy as np
from typing import List, Any

"""
Taken from the auraloss repository by Christian Steinmetz:
    https://github.com/csteinmetz1/auraloss/blob/main/auraloss/freq.py
    https://github.com/csteinmetz1/auraloss/blob/main/auraloss/utils.py

Modifications for Codename-RVC-Fork-4:
    - Removed unused loss classes / components
    - STFTLoss: removed w_lin_mag and w_phs terms
    - Original license: MIT (https://github.com/csteinmetz1/auraloss/blob/main/LICENSE)
"""


def apply_reduction(losses, reduction="none"):
    """Apply reduction to collection of losses."""
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    return losses


def get_window(win_type: str, win_length: int):
    """Return a window function.

    Args:
        win_type (str): Window type. Can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
        win_length (int): Window length

    Returns:
        win: The window as a 1D torch tensor
    """

    try:
        win = getattr(torch, win_type)(win_length)
    except:
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length))

    return win


class STFTMagnitudeLoss(torch.nn.Module):
    """STFT magnitude loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)

    Computes L1 or L2 distance between log10-scaled STFT magnitudes of two signals.
    Log-magnitudes are calculated with `log10(log_fac * x + log_eps)`, where `log_fac`
    controls compression strength and `log_eps` ensures numerical stability and controls
    the output range floor.
    For the purpose of use in Codename-RVC-Fork-4 we're using log10 (rather than natural log)

    Args:
        log (bool, optional): Use log10-scaled magnitudes. If False, uses linear magnitude.
            Default: True
        log_eps (float, optional): Constant added to magnitudes before log for numerical
            stability. Default: 1e-5
        log_fac (float, optional): Constant multiplier applied to magnitudes before log.
            Default: 1.0
        distance (str, optional): Distance function, one of ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction applied to loss elements, one of
            ["none", "mean", "sum"]. Default: "mean"
    """

    def __init__(self, log=True, log_eps=1e-5, log_fac=1.0, distance="L1", reduction="mean"):
        super(STFTMagnitudeLoss, self).__init__()

        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac

        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            log10 = torch.log(torch.tensor(10.0, device=x_mag.device))
            x_mag = torch.log(self.log_fac * x_mag.clamp(min=self.log_eps)) / log10
            y_mag = torch.log(self.log_fac * y_mag.clamp(min=self.log_eps)) / log10
        return self.distance(x_mag, y_mag)


class STFTLoss(torch.nn.Module):
    """STFT loss module.

    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).


    Args:
        fft_size (int, optional): FFT size in samples. Default: 1024
        hop_size (int, optional): Hop size of the FFT in samples. Default: 256
        win_length (int, optional): Length of the FFT analysis window. Default: 1024
        window (str, optional): Window to apply before FFT, can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
            Default: 'hann_window'
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            Default: 'mean'
        mag_distance (str, optional): Distance function ["L1", "L2"] for the magnitude loss terms.
        device (str, optional): Place the filterbanks on specified device. Default: None

    Returns:
        loss: Aggreate loss term.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
        w_log_mag: float = 1.0,
        sample_rate: float = None,
        fmin: float = None,
        fmin_weight: float = 0.1,
        eps: float = 1e-8,
        log_eps: float = 1e-5,
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Any = None,
        **kwargs
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", get_window(window, win_length).float())
        self.w_log_mag = w_log_mag
        self.sample_rate = sample_rate
        self.eps = eps
        self.reduction = reduction
        self.mag_distance = mag_distance
        self.device = device

        # Frequency emphasis curve
        n_bins = fft_size // 2 + 1
        if fmin is not None and sample_rate is not None:
            fmin_bin = int(np.ceil(fmin * fft_size / sample_rate))
            fmin_low = max(0, int(np.ceil((fmin * 0.5) * fft_size / sample_rate)))

            freq_weights = torch.ones(n_bins)
            freq_weights[:fmin_low] = fmin_weight
            if fmin_bin > fmin_low:
                ramp = torch.linspace(fmin_weight, 1.0, fmin_bin - fmin_low)
                freq_weights[fmin_low:fmin_bin] = ramp
            self.register_buffer("freq_weights", freq_weights.unsqueeze(0).unsqueeze(-1))
            self.use_freq_weights = True
        else:
            self.use_freq_weights = False

        self.logstft = STFTMagnitudeLoss(
            log=True,
            log_eps=log_eps,
            reduction=reduction,
            distance=mag_distance,
            **kwargs
        )

    def stft(self, x):
        """Perform STFT.
        Args:
            x (Tensor): Input signal tensor (B, T).

        Returns:
            Tensor: x_mag
                Magnitude spectra (B, fft_size // 2 + 1, frames).
        """
        window = self.window.to(x.device)
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            window,
            return_complex=True,
        )

        x_mag = torch.sqrt(torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=1e-6))
        return x_mag

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        bs, chs, seq_len = input.size()

        # STFT
        x_mag = self.stft(input.view(-1, seq_len))
        y_mag = self.stft(target.view(-1, seq_len))

        # Apply frequency emphasis if configured
        if self.use_freq_weights:
            fw = self.freq_weights.to(x_mag.device)
            x_mag = x_mag * fw
            y_mag = y_mag * fw

        # Log magnitude loss
        log_mag_loss = self.logstft(x_mag, y_mag)
        loss = self.w_log_mag * log_mag_loss
        loss = apply_reduction(loss, reduction=self.reduction)

        return loss
