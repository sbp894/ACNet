"""
Gammatone filterbank coefficients and filtering (PyTorch).

Vendored and trimmed for the standalone ACNet release. Only the pieces used by
the NEMS-style gammatonegram front end are kept (ERB spacing, filter-coefficient
construction, and the vectorised IIR filterbank). Original source: sppy.PT_gtg_filters.
"""

import torch
import torchaudio.functional as AF

DEFAULT_FILTER_NUM = 100
DEFAULT_LOW_FREQ = 100
DEFAULT_HIGH_FREQ = 44100 / 4


def erb_point(low_freq, high_freq, fraction):
    """Single point on an ERB scale between `low_freq` and `high_freq`."""
    ear_q = 9.26449
    min_bw = 24.7

    return (
        -ear_q * min_bw
        + torch.exp(
            fraction * (-torch.log(high_freq + ear_q * min_bw) + torch.log(low_freq + ear_q * min_bw))
        ) * (high_freq + ear_q * min_bw)
    )


def erb_space(low_freq=DEFAULT_FILTER_NUM, high_freq=DEFAULT_HIGH_FREQ, num=DEFAULT_FILTER_NUM):
    """`num` center frequencies equally spaced on an ERB scale (filterbank spacing)."""
    return erb_point(low_freq, high_freq, torch.arange(1, num + 1) / num)


def erb_space_hilb(low_freq=DEFAULT_FILTER_NUM, high_freq=DEFAULT_HIGH_FREQ, num=DEFAULT_FILTER_NUM):
    """`num` center frequencies equally spaced on an ERB scale (CF reporting)."""
    return erb_point(low_freq, high_freq, torch.arange(0, num) / (num - 1))


def make_erb_filters(fs, centre_freqs, width=1.0, device='cpu'):
    """
    Compute the filter coefficients for a bank of Gammatone filters.

    :param fs: Sampling frequency
    :param centre_freqs: Center frequencies of the filters (tensor)
    :param width: Width parameter for the filters
    :param device: 'cpu' or 'cuda'
    :return: fcoefs, a matrix of filter coefficients
    """
    T = 1 / fs
    ear_q = 9.26449  # Glasberg and Moore parameters
    min_bw = 24.7
    order = 1

    centre_freqs = centre_freqs.to(device)

    erb = width * ((centre_freqs / ear_q) ** order + min_bw ** order) ** (1 / order)
    tpi = torch.tensor(torch.pi).to(device)
    B = 1.019 * 2 * tpi * erb

    arg = 2 * centre_freqs * tpi * T
    vec = torch.exp(2j * arg)

    A0 = T
    A2 = torch.tensor(0.0, device=device)
    B0 = torch.tensor(1.0, device=device)
    B1 = -2 * torch.cos(arg) / torch.exp(B * T)
    B2 = torch.exp(-2 * B * T)

    rt_pos = torch.sqrt(torch.tensor(3.0 + 2 ** 1.5, device=device))
    rt_neg = torch.sqrt(torch.tensor(3.0 - 2 ** 1.5, device=device))

    common = -T * torch.exp(-(B * T))

    k11 = torch.cos(arg) + rt_pos * torch.sin(arg)
    k12 = torch.cos(arg) - rt_pos * torch.sin(arg)
    k13 = torch.cos(arg) + rt_neg * torch.sin(arg)
    k14 = torch.cos(arg) - rt_neg * torch.sin(arg)

    A11 = common * k11
    A12 = common * k12
    A13 = common * k13
    A14 = common * k14

    gain_arg = torch.exp(1j * arg - B * T)

    gain = torch.abs(
        (vec - gain_arg * k11)
        * (vec - gain_arg * k12)
        * (vec - gain_arg * k13)
        * (vec - gain_arg * k14)
        * (T * torch.exp(B * T)
           / (-1 / torch.exp(B * T) + 1 + vec * (1 - torch.exp(B * T))))
        ** 4
    )

    allfilts = torch.ones_like(centre_freqs, device=device)

    fcoefs = torch.stack([
        A0 * allfilts, A11, A12, A13, A14, A2 * allfilts,
        B0 * allfilts, B1, B2,
        gain
    ], dim=1)

    return fcoefs


def erb_filterbank(wave, coefs, device='cpu'):
    """
    Process an input waveform with a gammatone filter bank (vectorised IIR).

    :param wave: input data (1D torch tensor)
    :param coefs: gammatone filter coefficients (2D torch tensor)
    :param device: 'cpu' or 'cuda'
    :return: 2D tensor where each row corresponds to a filter's output.
    """
    wave = wave.to(device)
    coefs = coefs.to(device)

    num_channels = coefs.shape[0]

    gain = coefs[:, 9]

    Bs1 = coefs[:, (0, 1, 5)]  # A0, A11, A2
    Bs2 = coefs[:, (0, 2, 5)]  # A0, A12, A2
    Bs3 = coefs[:, (0, 3, 5)]  # A0, A13, A2
    Bs4 = coefs[:, (0, 4, 5)]  # A0, A14, A2
    As = coefs[:, 6:9]         # B0, B1, B2

    # Scale coefficients across channels for numerical stability
    scale_all_coeffs = As[:, 0:1]
    As = As / scale_all_coeffs
    Bs1, Bs2, Bs3, Bs4 = (Bs1 / scale_all_coeffs, Bs2 / scale_all_coeffs,
                          Bs3 / scale_all_coeffs, Bs4 / scale_all_coeffs)

    # Vectorise the filtering operations across all channels
    y1 = AF.lfilter(wave.unsqueeze(0).repeat(num_channels, 1), b_coeffs=Bs1, a_coeffs=As)
    y2 = AF.lfilter(y1, b_coeffs=Bs2, a_coeffs=As)
    y3 = AF.lfilter(y2, b_coeffs=Bs3, a_coeffs=As)
    y4 = AF.lfilter(y3, b_coeffs=Bs4, a_coeffs=As)

    return y4 / gain[:, None]