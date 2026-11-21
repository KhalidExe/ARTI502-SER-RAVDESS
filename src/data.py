"""RAVDESS download, metadata parsing, fold creation, leakage checks, Datasets and collate."""
from __future__ import annotations

import hashlib
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from . import augment, utils
from .utils import EMOTIONS, N_CLASSES

ZIP_NAME = "Audio_Speech_Actors_01-24.zip"
EXPECTED_N = 1440
METADATA_PATH = utils.ROOT / "data" / "metadata.csv"
FOLDS_PATH = utils.SPLITS_DIR / "folds.json"
RANDOM_SPLIT_PATH = utils.SPLITS_DIR / "random_split.json"
LABEL_MAP_PATH = utils.SPLITS_DIR / "label_map.json"


# ============================================================================
# Download
# ============================================================================
def file_md5(path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_ravdess(url: str, raw_dir) -> Path:
    """Download + extract Audio_Speech_Actors_01-24.zip (Zenodo 1188976). Idempotent."""
    raw_dir = utils.ensure_dir(utils.resolve(raw_dir))
    zip_path = raw_dir / ZIP_NAME
    if not zip_path.exists():
        tmp = zip_path.with_name(ZIP_NAME + ".part")
        print(f"Downloading {url}\n  -> {zip_path}")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done, next_report = 0, 0
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if done >= next_report:
                    pct = f"{100 * done / total:.0f}%" if total else ""
                    print(f"  {done / 1e6:.0f} MB {pct}", flush=True)
                    next_report += 20 * (1 << 20)
        tmp.replace(zip_path)
    wavs = [p for p in raw_dir.rglob("*.wav") if "__MACOSX" not in p.parts]
    if len(wavs) < EXPECTED_N:
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
    info = {"url": url, "zip": str(zip_path), "bytes": zip_path.stat().st_size,
            "md5": file_md5(zip_path)}
    utils.save_json(info, raw_dir / "download_info.json")
    print(f"zip md5 = {info['md5']}, size = {info['bytes'] / 1e6:.1f} MB")
    return raw_dir


# ============================================================================
# Metadata
# ============================================================================
def parse_filename(path) -> dict:
    """MM-VV-EE-II-SS-RR-AA.wav -> fields (see README)."""
    stem = Path(path).stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS file name: {stem}")
    modality, vocal, emo, inten, stmt, rep, actor = (int(p) for p in parts)
    return {
        "file_id": stem,
        "modality": modality,
        "vocal_channel": vocal,
        "emotion_code": emo,
        "emotion": EMOTIONS[emo - 1],
        "label": emo - 1,
        "intensity": inten,
        "statement": stmt,
        "repetition": rep,
        "actor": actor,
        "gender": "F" if actor % 2 == 0 else "M",
    }


def build_metadata(raw_dir, save: bool = True) -> pd.DataFrame:
    raw_dir = utils.resolve(raw_dir)
    files = sorted(p for p in raw_dir.rglob("*.wav") if "__MACOSX" not in p.parts)
    rows = []
    for p in files:
        r = parse_filename(p)
        r["rel_path"] = p.relative_to(raw_dir).as_posix()
        rows.append(r)
    df = pd.DataFrame(rows)
    df = df[(df.modality == 3) & (df.vocal_channel == 1)]
    df = df.sort_values("file_id").reset_index(drop=True)
    if save:
        utils.ensure_dir(METADATA_PATH.parent)
        df.to_csv(METADATA_PATH, index=False)
        utils.save_json({"label_to_index": utils.LABEL_MAP,
                         "index_to_label": {i: e for i, e in enumerate(EMOTIONS)},
                         "rule": "index = RAVDESS emotion code - 1"}, LABEL_MAP_PATH)
    return df


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"{METADATA_PATH} missing. Run: python -m src.prepare metadata")
    df = pd.read_csv(METADATA_PATH)
    assert df["file_id"].is_monotonic_increasing, "metadata must be sorted by file_id"
    return df


# ============================================================================
# Splits
# ============================================================================
def make_actor_folds(seed: int = 2026, n_folds: int = 6) -> dict:
    """Actor-wise, gender-balanced folds: 6 folds x (2 F + 2 M)."""
    rng = np.random.default_rng(seed)
    females = rng.permutation(np.arange(2, 25, 2))
    males = rng.permutation(np.arange(1, 25, 2))
    per = len(females) // n_folds
    folds = []
    for k in range(n_folds):
        f = females[k * per:(k + 1) * per]
        m = males[k * per:(k + 1) * per]
        folds.append(sorted(int(a) for a in np.concatenate([f, m])))
    rounds = []
    for k in range(n_folds):
        v = (k + 1) % n_folds
        tr = [j for j in range(n_folds) if j not in (k, v)]
        rounds.append({
            "round": k, "test_fold": k, "val_fold": v, "train_folds": tr,
            "test_actors": folds[k], "val_actors": folds[v],
            "train_actors": sorted(a for j in tr for a in folds[j]),
        })
    return {
        "seed": seed, "n_folds": n_folds,
        "method": "numpy default_rng(seed); permute 12 F and 12 M actors separately; "
                  "fold k = F[2k:2k+2] + M[2k:2k+2]; round k: test=fold k, val=fold (k+1)%6, train=rest",
        "folds": folds, "rounds": rounds,
    }


def make_random_split(meta: pd.DataFrame, seed: int = 2026, sizes=(960, 240, 240)) -> dict:
    """Stratified (by emotion) random UTTERANCE-level split; actors are shared. Q3 diagnostic only."""
    from sklearn.model_selection import train_test_split

    ids = meta["file_id"].to_numpy()
    y = meta["label"].to_numpy()
    rest, test = train_test_split(ids, test_size=sizes[2], stratify=y, random_state=seed)
    y_rest = meta.set_index("file_id").loc[rest, "label"].to_numpy()
    train, val = train_test_split(rest, test_size=sizes[1], stratify=y_rest, random_state=seed)
    return {
        "seed": seed, "sizes": list(sizes),
        "method": "sklearn train_test_split twice, stratified by emotion (test first, then val)",
        "train": sorted(train.tolist()), "val": sorted(val.tolist()), "test": sorted(test.tolist()),
    }


def load_splits() -> dict:
    for p in (FOLDS_PATH, RANDOM_SPLIT_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run: python -m src.prepare splits")
    return {"folds": utils.load_json(FOLDS_PATH), "random": utils.load_json(RANDOM_SPLIT_PATH)}


def split_indices(meta: pd.DataFrame, protocol: str, fold, splits: dict) -> dict:
    """Row indices (into metadata == feature-bank order) of train/val/test."""
    if protocol == "actor":
        r = splits["folds"]["rounds"][int(fold)]
        get = lambda actors: np.flatnonzero(meta["actor"].isin(actors).to_numpy())
        idx = {"train": get(r["train_actors"]), "val": get(r["val_actors"]), "test": get(r["test_actors"])}
    elif protocol == "random":
        pos = {fid: i for i, fid in enumerate(meta["file_id"])}
        rs = splits["random"]
        idx = {k: np.array(sorted(pos[f] for f in rs[k])) for k in ("train", "val", "test")}
    else:
        raise ValueError(protocol)
    assert_split_ok(meta, idx, protocol)
    return idx


def assert_split_ok(meta: pd.DataFrame, idx: dict, protocol: str) -> None:
    """Hard guard called before EVERY run."""
    s = {k: set(v.tolist()) for k, v in idx.items()}
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"]), \
        "utterance overlap between subsets"
    assert [len(idx[k]) for k in ("train", "val", "test")] == [960, 240, 240], \
        f"unexpected sizes {[len(idx[k]) for k in ('train', 'val', 'test')]}"
    if protocol == "actor":
        act = {k: set(meta["actor"].to_numpy()[v].tolist()) for k, v in idx.items()}
        assert not (act["train"] & act["val"]), "LEAKAGE: actor in train and val"
        assert not (act["train"] & act["test"]), "LEAKAGE: actor in train and test"
        assert not (act["val"] & act["test"]), "LEAKAGE: actor in val and test"


def _chk(name, passed, detail="") -> dict:
    return {"check": name, "passed": bool(passed), "detail": str(detail)}


def verify_data(meta: pd.DataFrame) -> list[dict]:
    out = []
    out.append(_chk("data: exactly 1440 files", len(meta) == EXPECTED_N, f"n={len(meta)}"))
    counts = meta["emotion"].value_counts().reindex(EMOTIONS).fillna(0).astype(int).to_dict()
    ok = counts["neutral"] == 96 and all(counts[e] == 192 for e in EMOTIONS[1:])
    out.append(_chk("data: neutral=96, other emotions=192", ok, counts))
    per_actor = meta.groupby("actor").size()
    out.append(_chk("data: 60 files per actor (24 actors)",
                    len(per_actor) == 24 and (per_actor == 60).all(), per_actor.unique().tolist()))
    g = meta.drop_duplicates("actor")["gender"].value_counts().to_dict()
    out.append(_chk("data: 12 F / 12 M actors", g.get("F") == 12 and g.get("M") == 12, g))
    ok = (meta["label"] == meta["emotion_code"] - 1).all() and \
        all(EMOTIONS[l] == e for l, e in zip(meta["label"], meta["emotion"]))
    if LABEL_MAP_PATH.exists():
        ok = ok and utils.load_json(LABEL_MAP_PATH)["label_to_index"] == utils.LABEL_MAP
    out.append(_chk("data: label map fixed (index = code-1) and saved", ok, utils.LABEL_MAP))
    return out


def verify_splits(meta: pd.DataFrame, splits: dict) -> list[dict]:
    out = []
    fo = splits["folds"]
    gender = dict(zip(meta["actor"], meta["gender"]))
    folds = fo["folds"]
    all_actors = sorted(a for f in folds for a in f)
    out.append(_chk("folds: 6 folds x 4 actors, every actor exactly once",
                    len(folds) == 6 and all(len(f) == 4 for f in folds) and all_actors == list(range(1, 25)),
                    folds))
    ok = all(sum(gender[a] == "F" for a in f) == 2 and sum(gender[a] == "M" for a in f) == 2 for f in folds)
    out.append(_chk("folds: each fold 2 F + 2 M", ok,
                    [[f"{a}{gender[a]}" for a in f] for f in folds]))
    disjoint, sizes = True, []
    for r in fo["rounds"]:
        try:
            idx = split_indices(meta, "actor", r["round"], splits)
            sizes.append([len(idx[k]) for k in ("train", "val", "test")])
        except AssertionError as e:
            disjoint = False
            sizes.append(str(e))
    out.append(_chk("folds: train/val/test actors pairwise disjoint in every round", disjoint))
    out.append(_chk("folds: sizes 960/240/240 in every round",
                    all(s == [960, 240, 240] for s in sizes), sizes))
    test_counts = pd.Series([a for r in fo["rounds"] for a in r["test_actors"]]).value_counts()
    out.append(_chk("folds: every actor in test exactly once across 6 rounds",
                    len(test_counts) == 24 and (test_counts == 1).all()))
    rs = splits["random"]
    st = {k: set(rs[k]) for k in ("train", "val", "test")}
    ok = [len(rs[k]) for k in ("train", "val", "test")] == [960, 240, 240] and \
        not (st["train"] & st["val"]) and not (st["train"] & st["test"]) and not (st["val"] & st["test"])
    out.append(_chk("random split: 960/240/240, utterances disjoint", ok))
    a = meta.set_index("file_id")["actor"]
    tr_act, te_act = set(a.loc[rs["train"]]), set(a.loc[rs["test"]])
    shared = sorted(te_act & tr_act)
    out.append(_chk("random split: actors ARE shared (leakage present by design)", len(shared) > 0,
                    f"{len(shared)}/{len(te_act)} test actors also in train"))
    lab = meta.set_index("file_id")["emotion"]
    strat = {k: lab.loc[rs[k]].value_counts().reindex(EMOTIONS).fillna(0).astype(int).tolist()
             for k in ("train", "val", "test")}
    out.append(_chk("random split: stratified by emotion (counts per subset)", True, strat))
    return out


# ============================================================================
# Datasets
# ============================================================================
class _AugMixin:
    """Shared D-01 logic: ONE version per utterance per epoch (clean or one of the V augmented copies),
    so an epoch always has exactly len(indices) samples. Augmented copies inherit the actor ID of their
    source utterance and may only be drawn for TRAINING actors of the current round (asserted)."""

    def _init_aug(self, bank, indices, train, aug_cfg, actors, aug_allowed_actors):
        self.train = bool(train)
        aug_on = bool(train and aug_cfg and aug_cfg.get("enabled", False))
        self.p_variant = float(aug_cfg.get("p_variant", 0.0)) if aug_on and len(bank) > 1 else 0.0
        self.variant_draws = 0          # how many samples used an augmented copy
        self.aug_used = set()           # utterance indices that received an augmented copy
        if self.p_variant > 0:
            if actors is None or aug_allowed_actors is None:
                raise ValueError("augmentation needs actors + aug_allowed_actors (train actors of the round)")
            used_actors = set(np.asarray(actors)[self.indices].tolist())
            assert used_actors <= set(aug_allowed_actors), \
                f"LEAKAGE: augmented copies requested for non-training actors {sorted(used_actors - set(aug_allowed_actors))}"
        return aug_on

    def _pick_version(self, idx):
        if self.p_variant > 0 and torch.rand(1).item() < self.p_variant:
            self.variant_draws += 1
            self.aug_used.add(idx)
            return int(torch.randint(1, len(self.bank), (1,)).item())
        return 0


class FeatureDataset(_AugMixin, Dataset):
    """Frame-level features (T, F) for M1-M3.

    bank[0] = clean features of all 1440 utterances; bank[v>=1] = precomputed augmented variants.
    Augmentation (variant choice + SpecAugment) happens ONLY when train=True.
    """

    def __init__(self, bank, indices, labels, standardizer=None, train=False, aug_cfg=None,
                 actors=None, aug_allowed_actors=None):
        self.bank = bank
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels)
        self.std = standardizer
        aug_on = self._init_aug(bank, self.indices, train, aug_cfg, actors, aug_allowed_actors)
        spec = (aug_cfg or {}).get("specaug", {}) if aug_on else {}
        self.spec = spec if spec.get("enabled", False) else None
        self.lengths = [int(bank[0][i].shape[0]) for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        v = self._pick_version(idx)
        x = torch.from_numpy(np.ascontiguousarray(self.bank[v][idx], dtype=np.float32))
        if self.std is not None:
            x = self.std.transform_torch(x)
        if self.spec is not None:
            x = augment.spec_augment(x, self.spec)
        return x, int(self.labels[idx]), idx


class WaveDataset(_AugMixin, Dataset):
    """Raw 16 kHz waveform (N,) for M4. Variants (waveform aug) only when train=True."""

    def __init__(self, bank, indices, labels, train=False, aug_cfg=None, max_seconds=None, sr=16000,
                 actors=None, aug_allowed_actors=None):
        self.bank = bank
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels)
        self._init_aug(bank, self.indices, train, aug_cfg, actors, aug_allowed_actors)
        self.max_len = int(max_seconds * sr) if max_seconds else None
        self.lengths = [int(bank[0][i].shape[0]) for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        v = self._pick_version(idx)
        x = torch.from_numpy(np.ascontiguousarray(self.bank[v][idx], dtype=np.float32))
        if self.max_len and x.shape[0] > self.max_len:  # only a safety cap; RAVDESS clips are short
            start = int(torch.randint(0, x.shape[0] - self.max_len + 1, (1,)).item()) if self.train else 0
            x = x[start:start + self.max_len]
        return x, int(self.labels[idx]), idx


class AccessCounter(Dataset):
    """Wraps the TEST dataset to prove it is read exactly once (after training)."""

    def __init__(self, ds):
        self.ds = ds
        self.count = 0
        self.lengths = getattr(ds, "lengths", None)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        self.count += 1
        return self.ds[i]


def collate_pad(batch) -> dict:
    """Pad to the longest item; mask[b, t] = True for real frames/samples."""
    xs, ys, idxs = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    T = int(lengths.max())
    if xs[0].dim() == 2:
        out = torch.zeros(len(xs), T, xs[0].shape[1], dtype=torch.float32)
    else:
        out = torch.zeros(len(xs), T, dtype=torch.float32)
    for b, x in enumerate(xs):
        out[b, : x.shape[0]] = x
    mask = torch.arange(T)[None, :] < lengths[:, None]
    return {"x": out, "mask": mask, "lengths": lengths,
            "y": torch.tensor(ys, dtype=torch.long), "idx": torch.tensor(idxs, dtype=torch.long)}


class BucketBatchSampler(Sampler):
    """Batches of similar length (less padding). Used for M4."""

    def __init__(self, lengths, batch_size, shuffle=True, generator=None, bucket_mult=50):
        self.lengths = list(lengths)
        self.bs = int(batch_size)
        self.shuffle = shuffle
        self.g = generator
        self.chunk = self.bs * bucket_mult

    def __iter__(self):
        n = len(self.lengths)
        order = torch.randperm(n, generator=self.g).tolist() if self.shuffle else list(range(n))
        batches = []
        for s in range(0, n, self.chunk):
            chunk = sorted(order[s:s + self.chunk], key=lambda i: self.lengths[i])
            batches.extend(chunk[j:j + self.bs] for j in range(0, len(chunk), self.bs))
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=self.g).tolist()
            batches = [batches[p] for p in perm]
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.bs)
