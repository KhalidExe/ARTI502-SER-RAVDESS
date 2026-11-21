"""Shared helpers: paths, label map, seeding, config loading, environment info."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ----------------------------------------------------------------------------
# Paths (everything is relative to 02_code/, independent of the working dir)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
SPLITS_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"

# ----------------------------------------------------------------------------
# Labels — FIXED mapping, used everywhere (training, evaluation, plots).
# label index = RAVDESS emotion code - 1   (01 neutral -> 0, ..., 08 surprised -> 7)
# ----------------------------------------------------------------------------
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
N_CLASSES = len(EMOTIONS)
LABEL_MAP = {name: i for i, name in enumerate(EMOTIONS)}
MODELS = ["m1", "m2", "m3", "m4"]
# which input each model accepts (first = default)
MODEL_FEATURES = {"m1": ["logmel", "mfcc"], "m2": ["logmel"], "m3": ["logmel"], "m4": ["wave"]}


def hard_exit(code: int = 0) -> None:
    """Flush and exit without interpreter teardown. On Windows the CUDA/joblib teardown can crash
    (exit 0xC0000409) after all results are already written, which would look like a failed step."""
    print(f"[done] exit code {int(code)}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":  # TerminateProcess skips the DLL detach routines where the crash happens
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.TerminateProcess(k32.GetCurrentProcess(), int(code))
    os._exit(int(code))


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_json(obj, path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_int_list(spec) -> list[int]:
    """'0-5' -> [0..5], '0,2,4' -> [0,2,4], 3 -> [3]."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return [int(s) for s in spec]
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ----------------------------------------------------------------------------
# Seeding
# ----------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int) -> None:  # DataLoader worker_init_fn
    import torch

    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_generator(seed: int):
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ----------------------------------------------------------------------------
# Configs: configs/base.yaml  <-  configs/<model>.yaml  <-  extra files  <-  CLI overrides
# ----------------------------------------------------------------------------
def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def get_by_path(cfg: dict, dotted: str):
    cur = cfg
    for key in dotted.split("."):
        cur = cur[key]
    return cur


def set_by_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = cfg
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def _parse_value(s: str):
    import yaml

    v = yaml.safe_load(s)
    if isinstance(v, str):  # PyYAML reads '1e-3' as a string (YAML 1.1); fix that
        try:
            return float(v)
        except ValueError:
            pass
    return v


def apply_overrides(cfg: dict, overrides: Iterable | dict | None) -> dict:
    if not overrides:
        return cfg
    if isinstance(overrides, dict):
        items = overrides.items()
    else:
        items = []
        for o in overrides:
            if isinstance(o, str):
                k, v = o.split("=", 1)
                items.append((k.strip(), _parse_value(v)))
            else:
                items.append(tuple(o))
    for k, v in items:
        set_by_path(cfg, k, v)
    return cfg


def load_config(model: str, overrides=None, extra_files: Iterable | None = None) -> dict:
    import yaml

    with open(CONFIG_DIR / "base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(CONFIG_DIR / f"{model}.yaml", "r", encoding="utf-8") as f:
        deep_update(cfg, yaml.safe_load(f))
    for path in extra_files or []:
        path = resolve(path)
        with open(path, "r", encoding="utf-8") as f:
            extra = json.load(f) if str(path).endswith(".json") else yaml.safe_load(f)
        # best-hparam files from hparam_search store {"overrides": {"train.lr": ...}}
        if "overrides" in extra:
            apply_overrides(cfg, extra["overrides"])
        else:
            deep_update(cfg, extra)
    apply_overrides(cfg, overrides)
    return cfg


def config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------
def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        pass
    return "nogit"


def _version(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not installed"


def env_info() -> dict:
    """Everything the paper needs about the environment (hardware + versions)."""
    info: dict[str, Any] = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    for m in ["numpy", "scipy", "pandas", "sklearn", "librosa", "soundfile", "soxr",
              "transformers", "matplotlib", "joblib", "yaml"]:
        info[m] = _version(m)
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        info["ram_gb"] = "unknown (psutil not installed)"
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_capability"] = f"sm_{p.major}{p.minor}"
            info["gpu_mem_gb"] = round(p.total_memory / 1024**3, 1)
            info["torch_arch_list"] = torch.cuda.get_arch_list()
    except Exception as e:  # pragma: no cover
        info["torch"] = f"error: {e}"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        info["nvidia_smi"] = out.stdout.strip()
    except Exception:
        info["nvidia_smi"] = "unavailable"
    return info
