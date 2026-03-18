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
    - SpectralConvergenceLoss: On silent / mute batches, we skip the sc calculation
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


class FIRFilter(torch.nn.Module):
    """FIR pre-emphasis filtering module.

    Args:
        filter_type (str): Shape of the desired FIR filter ("hp", "fd", "aw"). Default: "hp"
        coef (float): Coefficient value for the filter tap (only applicable for "hp" and "fd"). Default: 0.85
        ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
        plot (bool): Plot the magnitude respond of the filter. Default: False

    Based upon the perceptual loss pre-empahsis filters proposed by
    [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    A-weighting filter - "aw"
    First-order highpass - "hp"
    Folded differentiator - "fd"

    Note that the default coefficeint value of 0.85 is optimized for
    a sampling rate of 44.1 kHz, considering adjusting this value at differnt sampling rates.
    """

    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=101, plot=False):
        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot

        import scipy.signal

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul(
                [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
            )
            DENs = np.polymul(
                np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]
            )

            # convert analog filter to digital filter
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(
                1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
            )
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

            if plot:
                from .plotting import compare_filters
                compare_filters(b, a, taps, fs=fs)

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """
        weight = self.fir.weight.data.to(input.device)
        input = torch.nn.functional.conv1d(
            input, weight, padding=self.ntaps // 2
        )
        target = torch.nn.functional.conv1d(
            target, weight, padding=self.ntaps // 2
        )
        return input, target


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """

    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        y_norm = torch.norm(y_mag, p="fro")
        if y_norm < 1e-6:  # skip SC for mutes
            print(f"[SC] silent target detected, returning 0")
            return torch.tensor(0.0, device=x_mag.device)
        return torch.norm(y_mag - x_mag, p="fro") / y_norm

class STFTMagnitudeLoss(torch.nn.Module):
    """STFT magnitude loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)

    Log-magnitudes are calculated with `log(log_fac*x + log_eps)`, where `log_fac` controls the
    compression strength (larger value results in more compression), and `log_eps` can be used
    to control the range of the compressed output values (e.g., `log_eps>=1` ensures positive
    output values). The default values `log_fac=1` and `log_eps=0` correspond to plain log-compression.

    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        log_eps (float, optional): Constant value added to the magnitudes before evaluating the logarithm.
            Default: 0.0
        log_fac (float, optional): Constant multiplication factor for the magnitudes before evaluating the logarithm.
            Default: 1.0
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(self, log=True, log_eps=0.0, log_fac=1.0, distance="L1", reduction="mean"):
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
            x_mag = torch.log(self.log_fac * x_mag + self.log_eps)
            y_mag = torch.log(self.log_fac * y_mag + self.log_eps)
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
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of scaling frequency bins. Default: None.
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            Default: 'mean'
        mag_distance (str, optional): Distance function ["L1", "L2"] for the magnitude loss terms.
        device (str, optional): Place the filterbanks on specified device. Default: None

    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'. By default.
        loss, sc_mag_loss, log_mag_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        sample_rate: float = None,
        scale: str = None,
        perceptual_weighting: bool = False,
        n_bins: int = None,
        eps: float = 1e-8,
        output: str = "loss",
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Any = None,
        **kwargs
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = get_window(window, win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.sample_rate = sample_rate
        self.scale = scale
        self.perceptual_weighting = perceptual_weighting
        self.n_bins = n_bins
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.mag_distance = mag_distance
        self.device = device

        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(
            log=True,
            reduction=reduction,
            distance=mag_distance,
            **kwargs
        )

        # Perceptual / A-weighting
        if self.perceptual_weighting:
            if sample_rate is None:
                raise ValueError("sample_rate required for perceptual_weighting")
            self.prefilter = FIRFilter(filter_type="aw", fs=sample_rate)
        else:
            self.prefilter = None

        # setup mel filterbank
        if scale is not None:
            try:
                import librosa.filters
            except Exception as e:
                print(e)
                print("Try `pip install auraloss[all]`.")

            if self.scale == "mel":
                assert sample_rate != None  # Must set sample rate to use mel scale
                assert n_bins <= fft_size  # Must be more FFT bins than Mel bins
                fb = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
                fb = torch.tensor(fb).unsqueeze(0)

            elif self.scale == "chroma":
                assert sample_rate != None  # Must set sample rate to use chroma scale
                assert n_bins <= fft_size  # Must be more FFT bins than chroma bins
                fb = librosa.filters.chroma(
                    sr=sample_rate, n_fft=fft_size, n_chroma=n_bins
                )

            else:
                raise ValueError(
                    f"Invalid scale: {self.scale}. Must be 'mel' or 'chroma'."
                )

            self.register_buffer("fb", fb)

        if scale is not None and device is not None:
            self.fb = self.fb.to(self.device)  # move filterbank to device

    def stft(self, x):
        """Perform STFT.
        Args:
            x (Tensor): Input signal tensor (B, T).

        Returns:
            Tensor: x_mag, x_phs
                Magnitude and phase spectra (B, fft_size // 2 + 1, frames).
        """
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=self.eps)
        )

        return x_mag

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        bs, chs, seq_len = input.size()

        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(input.device)

        # apply A-weighting prefilter before STFT if enabled
        if self.prefilter is not None:
            input_f, target_f = self.prefilter(
                input.view(-1, 1, input.size(-1)),
                target.view(-1, 1, target.size(-1))
            )
            input_f = input_f.squeeze(1)
            target_f = target_f.squeeze(1)
        else:
            input_f = input.view(-1, input.size(-1))
            target_f = target.view(-1, target.size(-1))

        x_mag = self.stft(input_f)
        y_mag = self.stft(target_f)

        # apply relevant transforms
        if self.scale is not None:
            self.fb = self.fb.to(input.device)
            x_mag = torch.matmul(self.fb, x_mag)
            y_mag = torch.matmul(self.fb, y_mag)

        # compute loss terms
        # SC is skipped for silent samples / mutes, log magnitude still runs
        is_silent = target.abs().max() < 1e-6
        sc_mag_loss = 0.0 if is_silent else (self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0)
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0

        # combine loss terms
        loss = (
            (self.w_sc * sc_mag_loss)
            + (self.w_log_mag * log_mag_loss)
        )

        loss = apply_reduction(loss, reduction=self.reduction)

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sc_mag_loss, log_mag_loss
