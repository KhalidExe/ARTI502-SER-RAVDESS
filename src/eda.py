"""Data audit + EDA for template section 5.1 (counts, durations, acoustic descriptors, association with emotion).

All tables -> results/tables/, figures -> results/figures/ (prefix eda_).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as feat, utils
from .utils import EMOTIONS

TAB = utils.RESULTS_DIR / "tables"
FIG = utils.RESULTS_DIR / "figures"


def audit_files(meta: pd.DataFrame, raw_dir) -> pd.DataFrame:
    """Header of every WAV: sample rate, channels, subtype, duration."""
    import soundfile as sf

    raw_dir = utils.resolve(raw_dir)
    rows = []
    for fid, rel in zip(meta["file_id"], meta["rel_path"]):
        i = sf.info(str(raw_dir / rel))
        rows.append({"file_id": fid, "samplerate": i.samplerate, "channels": i.channels, "subtype": i.subtype,
                     "duration_s": i.frames / i.samplerate})
    df = pd.DataFrame(rows)
    utils.ensure_dir(TAB)
    df.to_csv(TAB / "audit_wav_headers.csv", index=False)
    return df


def count_tables(meta: pd.DataFrame) -> dict:
    utils.ensure_dir(TAB)
    t = {
        "by_emotion": meta.groupby("emotion").size().reindex(EMOTIONS).rename("n").to_frame()
                          .assign(share=lambda d: (d["n"] / d["n"].sum()).round(4)),
        "by_actor": meta.groupby(["actor", "gender"]).size().rename("n").reset_index(),
        "by_gender": meta.drop_duplicates("actor").groupby("gender").size().rename("actors").to_frame()
                          .join(meta.groupby("gender").size().rename("files")),
        "emotion_x_intensity": pd.crosstab(meta["emotion"], meta["intensity"]).reindex(EMOTIONS),
        "emotion_x_gender": pd.crosstab(meta["emotion"], meta["gender"]).reindex(EMOTIONS),
    }
    for k, v in t.items():
        v.to_csv(TAB / f"counts_{k}.csv")
    return t


def _descriptors_one(y, sr):
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    f0, voiced, _ = librosa.pyin(y, fmin=65.0, fmax=600.0, sr=sr, frame_length=1024, hop_length=160)
    f0v = f0[voiced & ~np.isnan(f0)] if f0 is not None else np.array([])
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=400, hop_length=160)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=400, hop_length=160)[0]
    rms_db = 20 * np.log10(rms + 1e-8)
    return {
        "duration_s": len(y) / sr,
        "rms_db_mean": float(rms_db.mean()), "rms_db_std": float(rms_db.std()),
        "f0_mean_hz": float(f0v.mean()) if len(f0v) else np.nan,
        "f0_std_hz": float(f0v.std()) if len(f0v) else np.nan,
        "f0_range_hz": float(np.percentile(f0v, 95) - np.percentile(f0v, 5)) if len(f0v) else np.nan,
        "voiced_fraction": float(voiced.mean()) if voiced is not None else np.nan,
        "spectral_centroid_hz": float(cent.mean()), "zcr_mean": float(zcr.mean()),
    }


def acoustic_descriptors(cfg, meta: pd.DataFrame, n_jobs=-1) -> pd.DataFrame:
    """Descriptors on the PREPROCESSED clean waveform (16 kHz, trimmed, peak-normalised).
    Note: peak normalisation removes absolute level, so RMS reflects dynamics, not loudness."""
    from joblib import Parallel, delayed

    sr = cfg["audio"]["sr"]
    waves = feat.load_bank(cfg, "wave", meta, use_variants=False)[0]
    rows = Parallel(n_jobs=n_jobs, verbose=2)(delayed(_descriptors_one)(np.asarray(w), sr) for w in waves)
    df = pd.concat([meta[["file_id", "emotion", "label", "intensity", "statement", "actor", "gender"]].reset_index(drop=True),
                    pd.DataFrame(rows)], axis=1)
    utils.ensure_dir(TAB)
    df.to_csv(TAB / "acoustic_descriptors.csv", index=False)
    return df


DESCRIPTORS = ["duration_s", "rms_db_mean", "rms_db_std", "f0_mean_hz", "f0_std_hz", "f0_range_hz",
               "voiced_fraction", "spectral_centroid_hz", "zcr_mean"]


def descriptor_stats(desc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Template 5.1.1: mean, median, std, max, min — overall and per emotion."""
    agg = ["mean", "median", "std", "max", "min"]
    overall = desc[DESCRIPTORS].agg(agg).T
    by_emo = desc.groupby("emotion")[DESCRIPTORS].agg(["mean", "median", "std"]).reindex(EMOTIONS)
    overall.to_csv(TAB / "descriptor_stats_overall.csv")
    by_emo.to_csv(TAB / "descriptor_stats_by_emotion.csv")
    return overall, by_emo


def descriptor_association(desc: pd.DataFrame) -> pd.DataFrame:
    """Audio analogue of 'correlation of attributes with target' (target is nominal, 8 classes):
    eta^2 (correlation ratio, one-way ANOVA effect size) and Kruskal-Wallis H with epsilon^2."""
    from scipy import stats

    rows = []
    for d in DESCRIPTORS:
        sub = desc[["emotion", d]].dropna()
        groups = [g[d].to_numpy() for _, g in sub.groupby("emotion")]
        grand = sub[d].mean()
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_t = ((sub[d] - grand) ** 2).sum()
        H, p = stats.kruskal(*groups)
        n, k = len(sub), len(groups)
        rows.append({"descriptor": d, "eta_squared": ss_b / ss_t if ss_t > 0 else np.nan,
                     "kruskal_H": H, "kruskal_p": p, "epsilon_squared": (H - k + 1) / (n - k), "n": n})
    df = pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
    df.to_csv(TAB / "descriptor_association.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_class_distribution(meta):
    import matplotlib.pyplot as plt

    c = meta.groupby("emotion").size().reindex(EMOTIONS)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(c.index, c.values, color="#4C72B0")
    for i, v in enumerate(c.values):
        ax.text(i, v + 2, str(v), ha="center", fontsize=8)
    ax.set_ylabel("utterances")
    ax.set_title("RAVDESS speech: utterances per emotion")
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(utils.ensure_dir(FIG) / "eda_class_distribution.png", dpi=200)
    return fig


def plot_durations(prep_info: pd.DataFrame, meta: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].hist(prep_info["raw_duration_s"], bins=40, alpha=0.6, label="raw (48 kHz file)")
    axes[0].hist(prep_info["trimmed_duration_s"], bins=40, alpha=0.6, label="after silence trim")
    axes[0].set_xlabel("duration (s)")
    axes[0].set_ylabel("files")
    axes[0].legend()
    d = prep_info.merge(meta[["file_id", "emotion"]], on="file_id")
    axes[1].boxplot([d.loc[d.emotion == e, "trimmed_duration_s"] for e in EMOTIONS])
    axes[1].set_xticks(range(1, len(EMOTIONS) + 1), EMOTIONS)
    axes[1].set_ylabel("trimmed duration (s)")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Utterance durations")
    fig.tight_layout()
    fig.savefig(FIG / "eda_durations.png", dpi=200)
    return fig


def plot_logmel_per_emotion(cfg, meta, actor=1):
    """One utterance per emotion: same actor, statement 1, normal intensity, repetition 1."""
    import matplotlib.pyplot as plt

    bank = feat.load_bank(cfg, "logmel", meta, use_variants=False)[0]
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), sharey=True)
    for ax, e in zip(axes.ravel(), EMOTIONS):
        sel = meta[(meta.actor == actor) & (meta.emotion == e) & (meta.statement == 1) &
                   (meta.intensity == 1) & (meta.repetition == 1)]
        i = sel.index[0]
        S = bank[i].T
        hop_s = cfg["features"]["hop_length"] / cfg["audio"]["sr"]
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, S.shape[1] * hop_s, 0, S.shape[0]])
        ax.set_title(f"{e} ({sel.file_id.iloc[0]})", fontsize=8)
        ax.set_xlabel("time (s)")
    axes[0, 0].set_ylabel("mel band")
    axes[1, 0].set_ylabel("mel band")
    fig.suptitle(f"64-band log-mel spectrograms, actor {actor:02d}, statement 1, normal intensity")
    fig.tight_layout()
    fig.savefig(FIG / "eda_logmel_per_emotion.png", dpi=200)
    return fig


def plot_descriptor_boxplots(desc):
    import matplotlib.pyplot as plt

    cols = ["f0_mean_hz", "f0_std_hz", "rms_db_std", "duration_s", "spectral_centroid_hz", "voiced_fraction"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.boxplot([desc.loc[desc.emotion == e, c].dropna() for e in EMOTIONS])
        ax.set_xticks(range(1, len(EMOTIONS) + 1), EMOTIONS)
        ax.set_title(c)
        ax.tick_params(axis="x", rotation=40, labelsize=7)
    fig.suptitle("Acoustic descriptors per emotion (preprocessed clean audio)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_descriptor_boxplots.png", dpi=200)
    return fig


def duration_stats(prep_info: pd.DataFrame) -> pd.DataFrame:
    s = prep_info[["raw_duration_s", "trimmed_duration_s"]].agg(["mean", "median", "std", "min", "max"]).T
    s.to_csv(utils.ensure_dir(TAB) / "duration_stats.csv")
    return s
