"""
ACNet: a multi-task 1D-ResNet model of ferret auditory cortex.

Standalone, dependency-light release. This module contains everything needed to
load ACNet and run it on your own audio:

  * A gammatonegram front end (`GammatoneFilterbankProcessing`).
  * The shared ResNet backbone (`layers_shared`) that produces the auditory
    "manifold" embeddings.
  * A single linear + DEXP readout over all 3124 recorded neurons. In the
    original research code these were 62 separate site-specific heads; here they
    are concatenated into one readout layer that predicts every neuron at once.

Typical use:

    from acnet_model import load_acnet
    model = load_acnet()
    model.update_audio_process({'overall_db': 60})   # set stimulus level (dB SPL)
    emb, gtg = model.get_mf_embeddings('my_sound.wav')     # embeddings (default)
    psth = model.predict_psth('my_sound.wav')              # optional neural PSTHs

Only the gammatonegram path and the DEXP nonlinearity are included, because
those are what the released ACNet weights use.
"""

import os
import struct
from collections import OrderedDict

# Import torch before numpy. In some MKL/OpenMP setups (e.g. conda numpy + pip
# torch), importing numpy first deadlocks the subsequent `import torch`.
import torch
import torch.nn as nn
import torchaudio  # only for the Resample transform; wav decoding uses stdlib struct
import numpy as np

from gtgram import gtgram
from gtg_filters import erb_space_hilb


WEIGHTS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'acnet_v1.pt')

# Model release version. Bump this (and ship a new acnet_v<N>.pt) for future
# retrains, e.g. v2. The version is also stored inside each checkpoint, so a
# loaded model reports the version it was actually trained/exported as.
ACNET_VERSION = "1.0.0"
MODEL_NAME = "ACNet"

_USABLE_CUDA = None


def best_device():
    """Return the best *usable* torch device: ``cuda`` only if a kernel can
    actually run on this GPU + torch-build combination, else ``cpu``.

    ``torch.cuda.is_available()`` returns True whenever a GPU is merely present,
    even when the installed torch wheel ships no kernels for it (e.g. an old
    GPU with a newer CUDA build). Using it to pick a device then fails at
    runtime with ``cudaErrorNoKernelImageForDevice``. This probes usability by
    launching one tiny kernel, caches the result, and falls back to CPU on any
    error. Override by passing an explicit device to ``load_acnet`` / ``.to()``.
    """
    global _USABLE_CUDA
    if _USABLE_CUDA is None:
        _USABLE_CUDA = False
        if torch.cuda.is_available():
            try:
                _ = (torch.zeros(1, device='cuda') + 1).item()  # force launch + sync
                _USABLE_CUDA = True
            except Exception:
                _USABLE_CUDA = False
    return torch.device('cuda' if _USABLE_CUDA else 'cpu')


# --------------------------------------------------------------------------- #
# Audio preprocessing
# --------------------------------------------------------------------------- #
def remove_clicks(w, max_threshold=50, verbose=False):
    """Log-compress samples beyond 67% of `max_threshold` (LBHB-mode only)."""
    w_clean = w.clone()
    crossover = 0.67 * max_threshold
    ii = w > crossover
    w_clean[ii] = crossover + torch.log(w_clean[ii] - crossover + 1).to(w_clean.dtype)
    jj = w < -crossover
    w_clean[jj] = -crossover - torch.log(-w_clean[jj] - crossover + 1).to(w_clean.dtype)
    if verbose:
        print(f'bins compressed down: {ii.sum().item()} up: {jj.sum().item()}')
    return w_clean


def _load_wav(path, int16_scale=32768.0):
    """Decode a WAV file to a mono ``(1, n_samples)`` float32 tensor in [-1, 1].

    Parses the RIFF container directly so the package needs no audio-decoding
    dependency (soundfile / torchcodec / ffmpeg). Supports integer PCM
    (8/16/24/32-bit) and IEEE-float (32/64-bit) WAV, including WAVE_FORMAT_EXTENSIBLE.

    `int16_scale` is the divisor for 16-bit PCM. The default 32768 is the
    full-scale convention; pass 32767 to reproduce NEMS (`nems_lbhb.runclass`
    divides int16 by 32767). The two differ by 3e-5 relative.
    """
    with open(path, 'rb') as f:
        riff = f.read()
    if riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
        raise ValueError(f"Not a RIFF/WAVE file: {path}")

    fmt_tag = n_channels = fs = bits = None
    raw = None
    pos = 12
    while pos + 8 <= len(riff):
        cid = riff[pos:pos + 4]
        csize = struct.unpack('<I', riff[pos + 4:pos + 8])[0]
        body = riff[pos + 8:pos + 8 + csize]
        if cid == b'fmt ':
            fmt_tag, n_channels, fs, _, _, bits = struct.unpack('<HHIIHH', body[:16])
            if fmt_tag == 0xFFFE and csize >= 26:   # EXTENSIBLE: real tag is in the subformat GUID
                fmt_tag = struct.unpack('<H', body[24:26])[0]
        elif cid == b'data':
            raw = body
        pos += 8 + csize + (csize & 1)  # chunks are word-aligned

    if fmt_tag is None or raw is None:
        raise ValueError(f"Missing fmt/data chunk in {path}")

    if fmt_tag == 1:            # integer PCM
        if bits == 8:          # 8-bit PCM is unsigned
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif bits == 16:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / int16_scale
        elif bits == 24:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            ints[ints >= 1 << 23] -= 1 << 24       # sign-extend
            data = ints.astype(np.float32) / (1 << 23)
        elif bits == 32:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported PCM bit depth {bits} in {path}")
    elif fmt_tag == 3:         # IEEE float
        dtype = np.float32 if bits == 32 else np.float64
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV format tag {fmt_tag} in {path}")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)  # mix to mono
    return torch.from_numpy(np.ascontiguousarray(data))[None, :], fs


def nems_audio_preprocess(sig, fs_stim, fs_gtg, f_max=20e3, duration=None, fixed_amp_scale=250,
                          lbhb_mode=False, overall_db=65, level_mode='exact', device='cpu',
                          verbose=False, nems_match=False):
    """
    Resample, ramp and level-scale a waveform prior to the gammatone filterbank.

    For inference outside LBHB set `lbhb_mode=False`; then `level_mode` must be
    'exact' and the signal is scaled so its RMS matches `overall_db` dB SPL
    (reference 20 uPa).

    `nems_match=False` (the default) is the path the released ACNet weights were
    trained on: a torchaudio polyphase resampler and a periodic Hann ramp.
    `nems_match=True` swaps in the two choices `nems_lbhb.runclass.NAT_stim`
    makes instead -- `scipy.signal.resample` (FFT) and a symmetric `np.hanning`
    ramp -- so this function reproduces a NEMS recording's `stim` signal to
    float32 precision. It needs scipy and is only for validating against NEMS;
    it does NOT reproduce the model's training inputs.
    """
    sig = sig.to(device)
    sig = sig.squeeze(0)  # assume mono

    window_time = 1 / fs_gtg
    hop_time = 1 / fs_gtg
    fs0 = f_max * 2  # nyquist for the NEMS path

    padbins = torch.tensor((window_time - hop_time) / 2 * fs0)
    if padbins != torch.round(padbins):
        raise ValueError("Need to adjust wav resampling rate")

    if duration is not None and len(sig) < fs_stim * duration:
        sig = torch.nn.functional.pad(sig, (0, int(fs_stim * duration) - len(sig)))

    if fs_stim != fs0:
        if nems_match:
            # scipy's FFT resampler, as used by NAT_stim. Differs audibly little
            # from the polyphase one (waveform r=0.9993) but enough to move the
            # gammatonegram by a few percent, so it must be matched exactly.
            from scipy.signal import resample as _scipy_resample
            n_out = int(len(sig) / fs_stim * fs0)
            sig_np = _scipy_resample(sig.detach().cpu().numpy().astype(np.float64), n_out)
            sig = torch.tensor(sig_np, dtype=torch.float32, device=device)
        else:
            resampler = torchaudio.transforms.Resample(orig_freq=fs_stim, new_freq=fs0).to(device)
            sig = resampler(sig)

    if duration is not None:
        # NAT_stim truncates to floor(Duration*fs0) after resampling, and only then
        # ramps -- so the offset ramp has to land on the truncated end, not before it.
        sig = sig[:int(np.floor(duration * fs0))]

    # 5 ms onset/offset ramp
    if nems_match:
        ramp_np = np.hanning(.005 * fs0 * 2)  # symmetric, as in NAT_stim
        ramp = torch.tensor(ramp_np[:int(np.floor(len(ramp_np) / 2))],
                            dtype=sig.dtype, device=device)
    else:
        ramp = torch.hann_window(int(.005 * fs0 * 2))[:int(.005 * fs0)].to(device)
    sig[:len(ramp)] *= ramp
    sig[-len(ramp):] *= torch.flip(ramp, [0])

    if lbhb_mode:
        sig = remove_clicks(sig * fixed_amp_scale, 15)
        pre_dbspl = 20 * torch.log10(torch.sqrt(torch.mean(sig ** 2)) / 20e-6)
        if overall_db is not None:
            if level_mode == 'approx':
                sf = 10 ** ((80 - overall_db) / 20)
            elif level_mode == 'exact':
                sf = 10 ** ((pre_dbspl - overall_db) / 20)
            sig = sig / sf
    else:
        pre_dbspl = 20 * torch.log10(torch.sqrt(torch.mean(sig ** 2)) / 20e-6)
        if overall_db is not None:
            assert level_mode == 'exact', "If lbhb_mode=False, then `level_mode` must be exact!"
            sf = 10 ** ((pre_dbspl - overall_db) / 20)
            sig = sig / sf

    if verbose:
        post_dbspl = 20 * torch.log10(torch.sqrt(torch.mean(sig ** 2)) / 20e-6)
        print(f"lbhb={lbhb_mode}: dbspl pre={pre_dbspl:.2f} post={post_dbspl:.2f}")

    return sig, fs0


class GammatoneFilterbankProcessing(nn.Module):
    """Wraps waveform preprocessing + NEMS gammatonegram + compression."""

    def __init__(self, num_cfs=32, f_min=200.0, f_max=20e3, fs_gtg=100.0, overall_db=65.0,
                 fixed_amp_scale=250.0, keep_pre_s=0., stim_dur_after_onset=None,
                 compress='log10x', level_mode='exact', lbhb_mode=False, nems_match=False,
                 stim_duration_s=None, verbose=False):
        super().__init__()
        self.num_cfs = num_cfs
        self.f_min = f_min
        self.f_max = f_max
        self.fs_gtg = fs_gtg
        self.overall_db = overall_db
        self.fixed_amp_scale = fixed_amp_scale
        self.keep_pre_s = keep_pre_s
        self.stim_dur_after_onset = stim_dur_after_onset
        self.compress = compress
        self.level_mode = level_mode
        self.lbhb_mode = lbhb_mode
        self.nems_match = nems_match
        # Zero-pad the WAVEFORM out to this many seconds before filtering, the way
        # NAT_stim pads every clip to the trial's `Duration`. Not the same as
        # `stim_dur_after_onset`, which zero-pads the finished gammatonegram: padding
        # the waveform lets the filterbank ring out into the added bins, padding the
        # gammatonegram does not. BNT wavs are 17.79 s against an 18 s Duration, so
        # this is what makes ACNet's output line up with a NEMS stim signal.
        self.stim_duration_s = stim_duration_s
        self.device = best_device()

    def apply_compress(self, x_mag):
        """Apply `self.compress` to a raw (uncompressed) gammatonegram magnitude."""
        if self.compress == 'sqrt':
            return torch.sqrt(torch.abs(x_mag))
        elif self.compress == 'log10x':
            return 0.5 * torch.log(1 + 10 * x_mag)
        elif self.compress == 'log50x':
            return 0.5 * torch.log10(1 + 50 * x_mag)
        elif self.compress == 'log2x':
            return torch.log(1 + 2 * x_mag)
        raise ValueError(f"Unknown compression '{self.compress}'")

    def forward(self, sig, fs_stim, overall_db=None, fixed_amp_scale=None):
        if overall_db is not None:
            self.overall_db = overall_db
        if fixed_amp_scale is not None:
            self.fixed_amp_scale = fixed_amp_scale

        w, fs_stim = nems_audio_preprocess(
            sig, fs_stim, self.fs_gtg, self.f_max, overall_db=self.overall_db, lbhb_mode=self.lbhb_mode,
            fixed_amp_scale=self.fixed_amp_scale, device=self.device, level_mode=self.level_mode,
            nems_match=self.nems_match, duration=self.stim_duration_s)

        x_gtg_pt = gtgram(w, fs_stim, window_time=1 / self.fs_gtg, hop_time=1 / self.fs_gtg,
                          channels=self.num_cfs, f_min=self.f_min, f_max=self.f_max, device=self.device)

        x_gtg_pt = self.apply_compress(x_gtg_pt)

        # Optional trimming / padding around stimulus onset
        if self.stim_dur_after_onset is not None:
            post_pad_gtg = int(self.stim_dur_after_onset * self.fs_gtg) - x_gtg_pt.shape[1]
            if post_pad_gtg < 0:
                x_gtg_pt = x_gtg_pt[:, :post_pad_gtg]
                post_pad_gtg = 0
        else:
            post_pad_gtg = 0

        pre_pad_gtg = int(self.keep_pre_s * self.fs_gtg)
        x_gtg_pt = torch.cat((torch.zeros((self.num_cfs, pre_pad_gtg), device=x_gtg_pt.device),
                              x_gtg_pt, torch.zeros((self.num_cfs, post_pad_gtg), device=x_gtg_pt.device)),
                             dim=1).T
        return x_gtg_pt

    def get_cfs(self):
        cfs = erb_space_hilb(torch.tensor(self.f_min), torch.tensor(self.f_max), torch.tensor(self.num_cfs))
        cfs = torch.flip(cfs, [0])
        return cfs.numpy().squeeze()


# --------------------------------------------------------------------------- #
# Backbone building blocks
# --------------------------------------------------------------------------- #
class Batchnorm1d_txp(nn.Module):
    """BatchNorm over the feature dim of a (batch, time, feature) tensor."""

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.norm_layer = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.norm_layer(x.transpose(-1, -2)).transpose(-1, -2)


class Conv1DRowLayer(nn.Module):
    """Causal depthwise (per-frequency-channel) FIR convolution over time."""

    def __init__(self, kernel_size, in_channels=1, out_channels=1, num_filters=20,
                 pool_size=1, normalize=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        assert in_channels == out_channels == 1, "Optimized code is not set up for channels > 1"
        assert normalize is False, "Not set up for normalization"

        self.conv = nn.Conv1d(in_channels=num_filters, out_channels=num_filters,
                              kernel_size=kernel_size, groups=num_filters, padding=0)
        self.pad = nn.ConstantPad1d((kernel_size - 1, 0), 0)  # causal padding
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size) if pool_size > 1 else None

    def forward(self, x):
        if x.dim() == 2:
            x = x[None, ...]
        if x.dim() == 3:
            x = x[:, None, :, :]
        # x: [batch, 1, time, freq]
        x = x.squeeze(1).permute(0, 2, 1)  # [batch, freq, time]
        x = self.pad(x)
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        return x.permute(0, 2, 1)  # [batch, time, freq]


class ResBlock_CNN1d_W_NL_v1(nn.Module):
    """FIR-conv -> linear -> (batchnorm) -> ReLU residual block."""

    def __init__(self, input_dim, output_dim, kernel_size, in_channels=1, out_channels=1,
                 time_pool=1, norm_1d=False, postfix='', skip=True, res_scale=1.0):
        super().__init__()
        self.params = {"input_dim": input_dim, "output_dim": output_dim, "kernel_size": kernel_size,
                       "in_channels": in_channels, "out_channels": out_channels, "time_pool": time_pool,
                       "norm_1d": norm_1d, "skip": skip}

        self.shortcut = self._make_shortcut(input_dim, output_dim, time_pool) if skip else nn.Identity()

        self.main = nn.Sequential()
        self.main.add_module(
            f"conv{postfix}", Conv1DRowLayer(kernel_size=kernel_size, in_channels=1, out_channels=1,
                                             pool_size=time_pool, num_filters=input_dim))
        self.main.add_module(f"linear{postfix}", nn.Linear(input_dim, output_dim))
        if norm_1d:
            self.main.add_module(f"bn{postfix}", Batchnorm1d_txp(output_dim))

        self.layer_NL = nn.Sequential(OrderedDict([(f"nl{postfix}", nn.ReLU())]))

        self.res_scale = res_scale
        self.gamma = nn.Parameter(torch.ones(1))
        self.register_buffer('running_grad_norm', torch.zeros(1))

    def _make_shortcut(self, in_dim, out_dim, pool):
        layers = [nn.Linear(in_dim, out_dim)]
        if pool > 1:
            layers.append(nn.AvgPool1d(pool, stride=pool))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        if self.params['skip']:
            out = self.gamma * self.res_scale * out + self.shortcut(x)
        out = self.layer_NL(out)
        return out


class DEXP(nn.Module):
    """
    Double-exponential output nonlinearity, applied per neuron.

        y = base + amp * exp(-exp(-exp(kappa) * x))
    """

    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.base = nn.Parameter(torch.zeros(num_nodes, dtype=torch.float32))
        self.amp = nn.Parameter(1.5 * torch.ones(num_nodes, dtype=torch.float32))
        self.kappa = nn.Parameter(0.7 * torch.ones(num_nodes, dtype=torch.float32))

    def forward(self, x):
        return self.base + self.amp * torch.exp(-torch.exp(-torch.exp(self.kappa) * x))


# --------------------------------------------------------------------------- #
# ACNet model
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "version": ACNET_VERSION,
    "input_dim": 32,           # number of gammatone frequency channels
    "hidden_dim": [75, 100, 125, 150, 175, 200],
    "kernel_size": [7, 7, 7, 7, 7, 7],
    "n_neurons": 3124,         # concatenated over all 62 recording sites
    "norm_1d": True,
    "res_scale": 1.0,
    "skip": True,
    # audio front end
    "fs_gtg": 100,
    "num_cfs": 32,
    "f_min": 200.0,
    "f_max": 20e3,
    "overall_db": 65.0,
    "fixed_amp_scale": 250.0,
    "keep_pre_s": 0.0,
    "stim_dur_after_onset": None,
    "compress": "log10x",
    "level_mode": "exact",
    "lbhb_mode": False,
    "nems_match": False,
    "stim_duration_s": None,
}


class ACNet(nn.Module):
    """
    Shared ResNet backbone + a single concatenated readout over all neurons.

    `get_mf_embeddings` returns the backbone ("manifold") embeddings and the
    gammatonegram. `predict_psth` additionally applies the neural readout to
    predict every recorded neuron's PSTH.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg
        self.version = cfg.get('version', ACNET_VERSION)
        self.model_name = MODEL_NAME

        input_dim = cfg["input_dim"]
        hidden_dim = cfg["hidden_dim"]
        kernel_size = cfg["kernel_size"]
        norm_1d = cfg["norm_1d"]
        res_scale = cfg["res_scale"]
        skip = cfg["skip"]

        self.device = best_device()

        self.audio_process = GammatoneFilterbankProcessing(
            num_cfs=cfg["num_cfs"], f_min=cfg["f_min"], f_max=cfg["f_max"], fs_gtg=cfg["fs_gtg"],
            overall_db=cfg["overall_db"], fixed_amp_scale=cfg["fixed_amp_scale"],
            keep_pre_s=cfg["keep_pre_s"], stim_dur_after_onset=cfg["stim_dur_after_onset"],
            compress=cfg["compress"], level_mode=cfg["level_mode"], lbhb_mode=cfg["lbhb_mode"],
            nems_match=cfg["nems_match"], stim_duration_s=cfg["stim_duration_s"])

        # Shared backbone: names match the original checkpoint (BlockCWR{i}).
        self.layers_shared = nn.Sequential()
        self.layers_shared.add_module("BlockCWR0", ResBlock_CNN1d_W_NL_v1(
            input_dim=input_dim, output_dim=hidden_dim[0], kernel_size=kernel_size[0], norm_1d=norm_1d,
            time_pool=1, postfix='0', in_channels=1, out_channels=1, skip=False))
        for in_idx in range(len(hidden_dim) - 1):
            out_idx = in_idx + 1
            self.layers_shared.add_module(f"BlockCWR{out_idx}", ResBlock_CNN1d_W_NL_v1(
                input_dim=hidden_dim[in_idx], output_dim=hidden_dim[out_idx], kernel_size=kernel_size[out_idx],
                norm_1d=norm_1d, time_pool=1, postfix=str(out_idx), in_channels=1, out_channels=1,
                skip=skip, res_scale=res_scale))

        # Single concatenated neural readout (was 62 site-specific heads).
        self.readout_linear = nn.Linear(hidden_dim[-1], cfg["n_neurons"])
        self.readout_nl = DEXP(cfg["n_neurons"])

    # --------------------------------------------------------------------- #
    def to(self, *args, **kwargs):
        """Move the model and keep the audio front end on the same device."""
        super().to(*args, **kwargs)
        dev = next(self.parameters()).device
        self.device = dev
        self.audio_process.device = dev
        return self

    def _device(self):
        """Device the model's weights currently live on."""
        return next(self.parameters()).device

    def update_audio_process(self, params_audio_proc=None, verbose=False):
        """Update a subset of audio-front-end params (e.g. `overall_db`)."""
        if params_audio_proc:
            allowed = {'overall_db', 'fixed_amp_scale', 'level_mode', 'lbhb_mode', 'keep_pre_s',
                       'stim_dur_after_onset', 'compress', 'f_min', 'f_max', 'fs_gtg', 'num_cfs',
                       'nems_match', 'stim_duration_s'}
            for key, val in params_audio_proc.items():
                if key in allowed:
                    setattr(self.audio_process, key, val)
                    self.config[key] = val
                    if verbose:
                        print(f"\t -> Updated {key} to {val}")
        self.audio_process.device = self.device

    def set_bnt_mode(self, overall_db, fixed_amp_scale, nems_match=False, verbose=False):
        """Configure the front end the way the BNT training stimuli were built.

        The BNT/baphy path is `remove_clicks(w * FixedAmpScale, 15)` followed by
        a fixed `10**((80 - OveralldB)/20)` gain -- i.e. `lbhb_mode=True` with
        `level_mode='approx'`. Both differ from the defaults, which target
        arbitrary wavs at a known dB SPL. Take `overall_db` / `fixed_amp_scale`
        from the recording's exptparams (`OveralldB`, `ReferenceHandle`
        `FixedAmpScale`), never from a hardcoded constant -- they vary by site.
        """
        self.update_audio_process(
            {'lbhb_mode': True, 'level_mode': 'approx', 'overall_db': overall_db,
             'fixed_amp_scale': fixed_amp_scale, 'nems_match': nems_match}, verbose=verbose)

    def _to_gtg(self, x, fs=None):
        # Always compute the gammatonegram on the same device as the weights,
        # regardless of prior .to()/.cpu()/.cuda() calls.
        dev = self._device()
        self.device = dev
        self.audio_process.device = dev

        if isinstance(x, str):
            # 32767 rather than 32768 when reproducing NEMS bit-for-bit; see _load_wav.
            int16_scale = 32767.0 if self.audio_process.nems_match else 32768.0
            waveform, fs = _load_wav(x, int16_scale=int16_scale)
        elif isinstance(x, torch.Tensor) and fs is not None:
            waveform = x
        else:
            raise ValueError("Input must be a path to a wav file or a (waveform, fs) pair.")
        return self.audio_process(waveform.to(dev), fs)

    def get_mf_embeddings(self, x, fs=None):
        """
        Return (manifold_rep, gtg).

          manifold_rep -> (batch, time, embedding_dim)
          gtg          -> (time, frequency)
        """
        gtg = self._to_gtg(x, fs)
        shared_rep = self.layers_shared(gtg)
        return shared_rep, gtg

    def gtg_to_model_input(self, gtg, gtg_compress='sqrt'):
        """Turn an externally computed gammatonegram into ACNet's model input.

        Use this for the `stim` signal of a NEMS recording, which
        `nems_lbhb.runclass.NAT_stim` returns already **sqrt-compressed**
        (`np.abs(s)**0.5`), whereas ACNet's input is `compress(magnitude)` with
        `compress='log10x'`. `gtg_compress` names the compression already
        applied to `gtg` -- it is undone to recover magnitude, then the model's
        own `self.audio_process.compress` is applied.

        Parameters
        ----------
        gtg : array or tensor, (time, num_cfs) or (num_cfs, time).
            Orientation is inferred from `num_cfs`; if both dims equal
            `num_cfs`, (time, num_cfs) is assumed.
        gtg_compress : {'sqrt', 'none'}.
            'sqrt' for a NEMS `stim` signal, 'none' for a raw magnitude gtgram.

        Returns
        -------
        torch.Tensor, (time, num_cfs), on the model's device.
        """
        dev = self._device()
        if not isinstance(gtg, torch.Tensor):
            gtg = torch.as_tensor(np.asarray(gtg, dtype=np.float32))
        gtg = gtg.to(dev).float().squeeze()
        if gtg.dim() != 2:
            raise ValueError(f"Expected a 2-D gammatonegram, got shape {tuple(gtg.shape)}")

        n_cfs = self.audio_process.num_cfs
        if gtg.shape[-1] != n_cfs:
            if gtg.shape[0] != n_cfs:
                raise ValueError(f"Neither axis of {tuple(gtg.shape)} matches num_cfs={n_cfs}")
            gtg = gtg.T  # (num_cfs, time) -> (time, num_cfs)

        if gtg_compress == 'sqrt':
            x_mag = gtg ** 2
        elif gtg_compress in ('none', None, 'linear'):
            x_mag = gtg
        else:
            raise ValueError(f"Unknown gtg_compress '{gtg_compress}'")
        return self.audio_process.apply_compress(x_mag)

    def get_mf_embeddings_from_gtg(self, gtg, gtg_compress='sqrt'):
        """`get_mf_embeddings` for a gammatonegram computed outside the model.

        Returns (manifold_rep, model_input); see `gtg_to_model_input`.
        """
        x = self.gtg_to_model_input(gtg, gtg_compress)
        return self.layers_shared(x), x

    def predict_psth_from_gtg(self, gtg, gtg_compress='sqrt', return_embeddings=False):
        """`predict_psth` for a gammatonegram computed outside the model."""
        shared_rep, x = self.get_mf_embeddings_from_gtg(gtg, gtg_compress)
        psth = self.readout_nl(self.readout_linear(shared_rep)).squeeze(0)
        if return_embeddings:
            return psth, shared_rep, x
        return psth

    def predict_psth(self, x, fs=None, return_embeddings=False):
        """
        Predict every recorded neuron's PSTH for the input audio.

        Returns psth -> (time, n_neurons); optionally also (manifold_rep, gtg).
        """
        shared_rep, gtg = self.get_mf_embeddings(x, fs)
        psth = self.readout_nl(self.readout_linear(shared_rep))
        psth = psth.squeeze(0)
        if return_embeddings:
            return psth, shared_rep, gtg
        return psth


def load_acnet(weights_path=None, map_location=None):
    """
    Load ACNet from a standalone checkpoint.

    Returns (model, cell_rtest) where `cell_rtest` is the per-neuron
    test-set prediction correlation (Pearson r) reported in the paper.

    The loaded model exposes `model.version` (the version the checkpoint was
    exported as) and `model.model_name`.
    """
    if weights_path is None:
        weights_path = WEIGHTS_DEFAULT
    if map_location is None:
        map_location = best_device()

    try:
        ckpt = torch.load(weights_path, map_location=map_location, weights_only=True)
    except Exception:
        # Fallback for older torch or checkpoints with non-tensor payloads.
        ckpt = torch.load(weights_path, map_location=map_location, weights_only=False)

    model = ACNet(ckpt.get('config'))
    model.load_state_dict(ckpt['state_dict'])
    model.to(map_location)  # place params + audio front end on the usable device
    model.eval()

    # Version metadata (checkpoint top-level wins over the config copy).
    model.version = ckpt.get('version', model.version)
    model.model_name = ckpt.get('model_name', MODEL_NAME)
    print(f"Loaded {model.model_name} v{model.version}")

    cell_rtest = ckpt.get('cell_rtest')
    if isinstance(cell_rtest, torch.Tensor):
        cell_rtest = cell_rtest.cpu().numpy()
    elif cell_rtest is not None:
        cell_rtest = np.asarray(cell_rtest)
    return model, cell_rtest
