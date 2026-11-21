"""Preprocessing (mono, 48k->16k, trim, peak-normalise), log-mel, MFCC+D+DD,
precomputed feature banks, and train-only standardisation."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import augment, utils

KINDS = ("logmel", "mfcc")


# ============================================================================
# Audio
# ============================================================================
def load_wave(path, sr: int = 16000):
    """Mono, resampled to sr. Returns (y float32, original_sr, original_duration_s)."""
    import librosa

    y, sr0 = librosa.load(str(path), sr=None, mono=True)
    dur0 = len(y) / sr0
    if sr0 != sr:
        y = librosa.resample(y, orig_sr=sr0, target_sr=sr, res_type="soxr_hq")
    return y.astype(np.float32), int(sr0), float(dur0)


def preprocess(y: np.ndarray, audio_cfg: dict):
    """Trim leading/trailing silence, then peak-normalise. Returns (y, (start, end))."""
    import librosa

    yt, (s, e) = librosa.effects.trim(y, top_db=audio_cfg["trim_top_db"])
    if len(yt) < audio_cfg["sr"] // 4:  # safety: never trim below 0.25 s
        yt, s, e = y, 0, len(y)
    peak = float(np.max(np.abs(yt))) if len(yt) else 0.0
    if peak > 0:
        yt = yt / peak * float(audio_cfg.get("peak", 1.0))
    return yt.astype(np.float32), (int(s), int(e))


# ============================================================================
# Features  (25 ms Hann window, 10 ms hop at 16 kHz -> n_fft=400, hop=160)
# ============================================================================
def _mel_power(y, sr, fc, n_mels):
    import librosa

    return librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=fc["n_fft"], win_length=fc["win_length"], hop_length=fc["hop_length"],
        window="hann", center=True, power=2.0, n_mels=n_mels, fmin=fc["fmin"], fmax=fc["fmax"])


def logmel(y, sr, fc) -> np.ndarray:
    """(T, n_mels) log-mel in dB (absolute ref=1.0, clamp top_db below the max)."""
    import librosa

    S = _mel_power(y, sr, fc, fc["n_mels"])
    L = librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=fc["top_db"])
    return L.T.astype(np.float32)


def mfcc_deltas(y, sr, fc) -> np.ndarray:
    """(T, 3*n_mfcc): 40 MFCC + delta + delta-delta."""
    import librosa

    S = _mel_power(y, sr, fc, fc["mfcc_n_mels"])
    M = librosa.feature.mfcc(S=librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=fc["top_db"]),
                             n_mfcc=fc["n_mfcc"])
    T = M.shape[1]
    w = min(int(fc["delta_width"]), T if T % 2 == 1 else T - 1)
    if w >= 3:
        d1 = librosa.feature.delta(M, width=w, order=1)
        d2 = librosa.feature.delta(M, width=w, order=2)
    else:
        d1 = np.zeros_like(M)
        d2 = np.zeros_like(M)
    return np.concatenate([M, d1, d2], axis=0).T.astype(np.float32)


def extract(kind: str, y, sr, fc) -> np.ndarray:
    if kind == "logmel":
        return logmel(y, sr, fc)
    if kind == "mfcc":
        return mfcc_deltas(y, sr, fc)
    raise ValueError(kind)


# ============================================================================
# Ragged storage: one concatenated array + offsets
# ============================================================================
def save_ragged(path, arrays) -> None:
    lens = np.array([a.shape[0] for a in arrays], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    data = np.concatenate(arrays, axis=0).astype(np.float32)
    np.savez(path, data=data, offsets=offsets)


def load_ragged(path) -> list:
    z = np.load(path)
    data, off = z["data"], z["offsets"]
    return [data[off[i]:off[i + 1]] for i in range(len(off) - 1)]


# ============================================================================
# Feature banks (clean + V precomputed augmented variants per utterance)
# ============================================================================
def _process_file(rel_path, raw_dir, cfg, file_index, n_variants, aug_seed, kinds):
    sr = cfg["audio"]["sr"]
    fc = cfg["features"]
    y, sr0, dur0 = load_wave(Path(raw_dir) / rel_path, sr)
    y, (s, e) = preprocess(y, cfg["audio"])
    out = {"sr0": sr0, "raw_dur": dur0, "trim_dur": len(y) / sr, "trim_start_s": s / sr,
           "waves": [y], "feats": {k: [extract(k, y, sr, fc)] for k in kinds}}
    for v in range(1, n_variants + 1):
        rng = np.random.default_rng([int(aug_seed), int(file_index), int(v)])
        ya = augment.waveform_augment(y, sr, rng, cfg["augment"])
        out["waves"].append(ya)
        for k in kinds:
            out["feats"][k].append(extract(k, ya, sr, fc))
    return out


def _wave_aug_cfg(cfg):
    a = cfg["augment"]
    return {k: a[k] for k in ("enabled", "n_variants", "aug_seed", "time_stretch", "pitch_shift", "gain", "noise")}


def cache_dir(cfg) -> Path:
    return utils.resolve(cfg["data"]["cache_dir"])


def build_feature_banks(cfg, meta: pd.DataFrame, n_jobs: int = -1, kinds=KINDS) -> Path:
    """Compute waves + features for all 1440 files (clean + augmented variants) in parallel.

    Variants are computed for every utterance, but Datasets only use them when train=True,
    so validation/test data are never augmented.
    """
    from joblib import Parallel, delayed

    t0 = time.time()
    raw_dir = utils.resolve(cfg["data"]["raw_dir"])
    cdir = utils.ensure_dir(cache_dir(cfg))
    V = int(cfg["augment"]["n_variants"]) if cfg["augment"]["enabled"] else 0
    seed = int(cfg["augment"]["aug_seed"])
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_file)(rel, str(raw_dir), cfg, i, V, seed, tuple(kinds))
        for i, rel in enumerate(meta["rel_path"]))
    for v in range(V + 1):
        save_ragged(cdir / f"wave_v{v}.npz", [r["waves"][v] for r in res])
        for k in kinds:
            save_ragged(cdir / f"{k}_v{v}.npz", [r["feats"][k][v] for r in res])
    info = pd.DataFrame({
        "file_id": meta["file_id"], "orig_sr": [r["sr0"] for r in res],
        "raw_duration_s": [r["raw_dur"] for r in res], "trimmed_duration_s": [r["trim_dur"] for r in res],
        "trim_start_s": [r["trim_start_s"] for r in res],
        "n_frames": [r["feats"][kinds[0]][0].shape[0] for r in res],
    })
    info.to_csv(cdir / "preprocess_info.csv", index=False)
    import librosa

    manifest = {
        "file_ids": meta["file_id"].tolist(), "n_variants": V, "kinds": list(kinds),
        "audio": cfg["audio"], "features": cfg["features"], "augment_wave": _wave_aug_cfg(cfg),
        "librosa": librosa.__version__, "build_time_s": round(time.time() - t0, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    utils.save_json(manifest, cdir / "manifest.json")
    print(f"Feature banks written to {cdir} in {manifest['build_time_s']} s (V={V})")
    return cdir


def load_bank(cfg, kind: str, meta: pd.DataFrame, use_variants: bool = True) -> list:
    """bank[v][i] -> array for utterance i (metadata row order), variant v (0 = clean)."""
    cdir = cache_dir(cfg)
    mpath = cdir / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"No feature cache in {cdir}. Run: python -m src.prepare features")
    man = utils.load_json(mpath)
    if man["file_ids"] != meta["file_id"].tolist():
        raise RuntimeError("Feature cache order != metadata order. Rebuild: python -m src.prepare features")
    for key, cur in (("audio", cfg["audio"]), ("features", cfg["features"]), ("augment_wave", _wave_aug_cfg(cfg))):
        if man[key] != cur:
            raise RuntimeError(f"Feature cache built with different '{key}' settings:\n cache={man[key]}\n "
                               f"config={cur}\nRebuild: python -m src.prepare features")
    stem = "wave" if kind == "wave" else kind
    V = man["n_variants"] if use_variants else 0
    return [load_ragged(cdir / f"{stem}_v{v}.npz") for v in range(V + 1)]


# ============================================================================
# Standardisation — fitted on TRAINING utterances only, per round
# ============================================================================
class Standardizer:
    def __init__(self, mean, std):
        import torch

        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self._mt = torch.from_numpy(self.mean)
        self._st = torch.from_numpy(self.std)

    @classmethod
    def fit(cls, arrays, eps: float = 1e-5) -> "Standardizer":
        """Per-dimension mean/std over ALL frames of the given (training, clean) utterances."""
        n = 0
        s = None
        ss = None
        for a in arrays:
            a = np.asarray(a, dtype=np.float64)
            s = a.sum(0) if s is None else s + a.sum(0)
            ss = (a ** 2).sum(0) if ss is None else ss + (a ** 2).sum(0)
            n += a.shape[0]
        mean = s / n
        var = np.maximum(ss / n - mean ** 2, 0.0)
        return cls(mean, np.maximum(np.sqrt(var), eps))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def transform_torch(self, x):
        return (x - self._mt) / self._st

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
