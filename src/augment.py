"""Training-only augmentation.

Waveform level (precomputed as fixed variants, see features.build_feature_banks):
    time stretch, pitch shift, random gain, additive Gaussian noise.
Spectrogram level (on the fly, training DataLoader only):
    SpecAugment-style frequency and time masking (Park et al., 2019).
"""
from __future__ import annotations

import numpy as np
import torch


def waveform_augment(y: np.ndarray, sr: int, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """One random augmented copy of a (preprocessed) waveform. Deterministic given rng."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    # draw every random number unconditionally so the stream does not depend on which ops fire
    ts, ps, g, nz = cfg["time_stretch"], cfg["pitch_shift"], cfg["gain"], cfg["noise"]
    u = rng.random(4)
    rate = rng.uniform(ts["min_rate"], ts["max_rate"])
    steps = rng.uniform(-ps["max_semitones"], ps["max_semitones"])
    gain_db = rng.uniform(-g["max_db"], g["max_db"])
    snr_db = rng.uniform(nz["min_snr_db"], nz["max_snr_db"])
    noise_seed = int(rng.integers(0, 2**31 - 1))

    if u[0] < ts["p"]:
        y = librosa.effects.time_stretch(y, rate=float(rate))
    if u[1] < ps["p"]:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(steps), res_type="soxr_hq")
    if u[2] < g["p"]:
        y = y * np.float32(10.0 ** (gain_db / 20.0))
    if u[3] < nz["p"]:
        p_sig = float(np.mean(y.astype(np.float64) ** 2))
        p_noise = p_sig / (10.0 ** (snr_db / 10.0))
        y = y + np.random.default_rng(noise_seed).normal(0.0, np.sqrt(p_noise), size=y.shape)
    return np.asarray(y, dtype=np.float32)


def spec_augment(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """x: (T, F) standardised features. Masked values are set to 0 (= training mean)."""
    T, F = x.shape
    x = x.clone()
    fw_max = max(1, int(round(cfg.get("freq_mask_frac", 0.125) * F)))
    for _ in range(int(cfg.get("n_freq_masks", 2))):
        w = int(torch.randint(0, fw_max + 1, (1,)).item())
        if w == 0:
            continue
        f0 = int(torch.randint(0, F - w + 1, (1,)).item())
        x[:, f0:f0 + w] = 0.0
    tw_max = min(int(cfg.get("time_mask_max", 20)), int(cfg.get("time_mask_frac", 0.1) * T))
    for _ in range(int(cfg.get("n_time_masks", 2))):
        if tw_max < 1:
            break
        w = int(torch.randint(0, tw_max + 1, (1,)).item())
        if w == 0:
            continue
        t0 = int(torch.randint(0, T - w + 1, (1,)).item())
        x[t0:t0 + w, :] = 0.0
    return x
