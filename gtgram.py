"""
NEMS-style gammatonegram (PyTorch).

Vendored and trimmed for the standalone ACNet release. Only the NEMS
(summed-energy) gammatonegram path is kept, since ACNet was trained with
`gtg_mode='nems'`. Original source: sppy.PT_gtgram.
"""

import torch
from gtg_filters import erb_space, make_erb_filters, erb_filterbank


def cast2tensor(wave, fs, channels, f_min, f_max, fs_gtg=None, device='cpu'):
    if not torch.is_tensor(wave):
        wave = torch.tensor(wave, dtype=torch.float32)
    if not torch.is_tensor(fs):
        fs = torch.tensor(fs, dtype=torch.float32)
    if (fs_gtg is not None) and (not torch.is_tensor(fs_gtg)):
        fs_gtg = torch.tensor(fs_gtg, dtype=torch.float32)
    if not torch.is_tensor(channels):
        channels = torch.tensor(channels, dtype=torch.int32)
    if not torch.is_tensor(f_min):
        f_min = torch.tensor(f_min, dtype=torch.float32)
    if not torch.is_tensor(f_max):
        f_max = torch.tensor(f_max, dtype=torch.float32)

    wave = wave.to(device)

    return wave, fs, channels, f_min, f_max, fs_gtg


def round_half_away_from_zero(num):
    return torch.floor(num + 0.5 * torch.sign(num))


def gtgram_strides(fs, window_time, hop_time, filterbank_cols):
    nwin = int(round_half_away_from_zero(window_time * fs))
    hop_samples = int(round_half_away_from_zero(hop_time * fs))
    columns = 1 + int(torch.floor(torch.tensor((filterbank_cols - nwin) / hop_samples)))
    return nwin, hop_samples, columns


def gtgram_xe(wave, fs, channels, f_min, f_max=None, device='cpu'):
    cfs = erb_space(f_min, f_max or fs / 2, channels).to(device)
    fcoefs = torch.flip(make_erb_filters(fs, cfs, device=device), dims=[0])
    xf = erb_filterbank(wave.to(device), fcoefs, device=device)
    xe = torch.pow(xf, 2)
    return xe


def gtgram(wave, fs, window_time, hop_time, channels, f_min, f_max=None, device='cpu'):
    wave, fs, channels, f_min, f_max, _ = cast2tensor(wave, fs, channels, f_min, f_max, device=device)

    xe = gtgram_xe(wave, fs, channels, f_min, f_max, device=device)
    nwin, hop_samples, ncols = gtgram_strides(fs, window_time, hop_time, xe.shape[1])

    # Extract every window and take the RMS energy over it.
    start_indices = torch.arange(0, ncols * hop_samples, hop_samples, device=xe.device)
    segments = torch.stack([xe[:, start_idx:start_idx + nwin] for start_idx in start_indices], dim=2)
    y = torch.sqrt(segments.mean(dim=1))

    return y
