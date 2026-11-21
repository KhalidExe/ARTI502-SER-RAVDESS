# Speech Emotion Recognition on RAVDESS — LSTM vs. Transformer (speaker-independent)

ARTI 502 Deep Learning, Group 1 (IAU, Term 1 2026/27). Code for the group's approved project proposal.
Repository: https://github.com/KhalidExe/ARTI502-SER-RAVDESS

> **Status (21 Nov 2026):** all results in `results/` are **preliminary**: 1 seed, default hyper-parameters
> (not tuned), tag `prelim`. The leakage (random-split) numbers are a single run per model (diagnostic only).
> `results/hparam_search/` is a **partial snapshot**: M3 and M2 complete, M1 in progress, M4 not started.
> It will be updated, together with the 3-seed tuned protocol, before the final report.

Models (all share the same folds, preprocessing and evaluation code):

| ID | Model | Input |
|---|---|---|
| M1 | 2-layer BiLSTM (128/dir) + attention pooling | 64-band log-mel (and 40 MFCC+Δ+ΔΔ for the ablation) |
| M2 | 3 conv blocks → BiLSTM → attention pooling | log-mel |
| M3 | Transformer encoder from scratch (4 layers, 4 heads, d=256) | log-mel |
| M4 | wav2vec 2.0 Base fine-tuned (`facebook/wav2vec2-base`) | raw 16 kHz waveform |

Protocol: actor-wise, gender-balanced 6-fold CV (test = fold k, val = fold k+1, train = other 4; 960/240/240),
3 seeds, early stopping on **validation UAR**, test fold used once. Q3 diagnostic: stratified random utterance split (M1–M3).
All choices not fixed by the proposal are logged in **`DECISIONS.md`**.

---

## 1. Setup (PC 1: RTX 5080 16 GB)

Python 3.11 or 3.12 recommended. Run everything from the `02_code` folder.

**Windows (PowerShell, PC 1 — the machine used for this project)**
```powershell
cd <path-to-repo>
py -3.12 -m venv .venv
# if activation is blocked once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**VS Code (notebooks)**
1. Install the *Python* and *Jupyter* extensions; open the repository folder.
2. Open a notebook → *Select Kernel* → *Python Environments* → `.venv`.
3. Notebooks find the project root themselves (works whether VS Code starts them in `notebooks/` or in `02_code/`).
4. Long batches (full runs, search): run them in the VS Code **terminal** (`.venv` active), not inside a notebook,
   e.g. `.\scripts\run_progress_tonight.ps1`. Logs: `results\logs\`.
5. Optional: `setx HF_HUB_DISABLE_SYMLINKS_WARNING 1` (harmless Hugging Face cache warning on Windows).

**Ubuntu**
```bash
cd <path-to-repo>
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

RTX 5080 is Blackwell (sm_120): the PyTorch wheel **must** be cu128 or newer; older wheels fail with
"no kernel image is available". `notebooks/00_env_check.ipynb` verifies this and freezes versions.

---

## 2. Run order

| Step | Command / notebook | Output |
|---|---|---|
| 1. Environment | `notebooks/00_env_check.ipynb` | `results/env/env_info.json`, `pip_freeze.txt` |
| 2. Data | `python -m src.prepare all` | `data/raw/` (Zenodo 1188976), `data/metadata.csv`, `splits/*.json`, `data/cache/*` |
| 2b. Audit + EDA | `notebooks/01_data_audit_and_eda.ipynb` | `results/tables/*`, `results/figures/eda_*.png` |
| 3. Checks | `python -m src.checks --overfit` (or notebook 02) | `results/checks/checks.md` |
| 4. Smoke test | `notebooks/02_smoke_test.ipynb` | `results/smoke/` (separate from real results) |
| 5. HP search | `python -m src.hparam_search --model m1` (m2, m3; m4 with `--rounds 0,3`) | `results/hparam_search/<model>_<features>/` |
| 6. Full runs | see below | `results/runs.csv`, `predictions/`, `curves/`, `runs/` |
| 7. Analysis | `notebooks/03_results_analysis.ipynb` | `results/tables/`, `results/figures/` |

**Priority for the progress report (21 Nov):** steps 1–4, then preliminary (untuned) numbers:
```
python -m src.run_experiments --model m1 --folds 0-5 --seeds 0 --tag prelim
python -m src.run_experiments --model m3 --folds 0-5 --seeds 0 --tag prelim
python -m src.run_experiments --model m2 --folds 0-5 --seeds 0 --tag prelim
```
or all of it (checks + the three batches) in one go: `.\scripts\run_progress_tonight.ps1`

**Full protocol (after the search):**
```
# main: 4 models x 6 folds x 3 seeds = 72 runs
python -m src.run_experiments --model m1 --tag main --hparams results/hparam_search/m1_logmel/best.json
python -m src.run_experiments --model m2 --tag main --hparams results/hparam_search/m2_logmel/best.json
python -m src.run_experiments --model m3 --tag main --hparams results/hparam_search/m3_logmel/best.json
python -m src.run_experiments --model m4 --tag main --hparams results/hparam_search/m4_wave/best.json
# MFCC ablation: 18 runs
python -m src.run_experiments --model m1 --features mfcc --tag main --hparams results/hparam_search/m1_logmel/best.json
# leakage diagnostic: M1-M3 x 3 seeds = 9 runs
python -m src.run_experiments --model m1 --protocol random --tag main --hparams results/hparam_search/m1_logmel/best.json
python -m src.run_experiments --model m2 --protocol random --tag main --hparams results/hparam_search/m2_logmel/best.json
python -m src.run_experiments --model m3 --protocol random --tag main --hparams results/hparam_search/m3_logmel/best.json
```
Add `--skip-existing` to resume an interrupted batch. Overrides: `--override train.lr=5.0e-4`.

---

## 3. Layout

```
<repository root>/
├── configs/            base.yaml + m1..m4.yaml + hparam_search.yaml
├── src/
│   ├── utils.py        paths, fixed label map, seeding, config loading, env info
│   ├── data.py         download, metadata, folds, random split, leakage asserts, Datasets, padding+masks
│   ├── features.py     16 kHz / trim / peak-norm, log-mel, MFCC+Δ+ΔΔ, feature cache, train-only standardiser
│   ├── augment.py      waveform augmentation, SpecAugment
│   ├── models.py       M1–M4 (all mask-aware)
│   ├── train.py        training loop, early stopping on val UAR, test evaluation, run logging
│   ├── evaluate.py     accuracy / UAR / macro-F1 / confusion, paired tests, trivial baselines
│   ├── checks.py       validation checklist (section 4 below)
│   ├── prepare.py      data CLI
│   ├── run_experiments.py   batch CLI
│   ├── hparam_search.py     validation-only search
│   ├── verify_runs.py       checks every run left complete outputs (runs.csv row, curves, 240 predictions)
│   ├── eda.py, analysis.py  tables and figures
├── notebooks/          00 env · 00a CUDA torch · 01 EDA · 02 checks+smoke · 03 analysis
│                       04 pipeline runner · 05 extra prelim runs · 06 hyper-parameter search
├── scripts/            run_progress_tonight.ps1 (Windows batch), scrub_paths.py (pre-push path scrubbing)
├── splits/             folds.json, random_split.json, label_map.json (fixed once, committed)
├── results/            runs.csv, predictions/, curves/, runs/, tables/, figures/, checks/, env/, logs/,
│                       hparam_search/ (PARTIAL snapshot: M3 and M2 done, M1 in progress; to be updated)
├── data/, checkpoints/ (git-ignored)
└── DECISIONS.md
```

`results/runs.csv`: one row per (model, features, protocol, fold, seed, tag) with best epoch, val/test UAR,
accuracy, macro-F1, 8 per-class recalls, parameter counts, train time, GPU, config hash, git commit, hyper-parameters.
`results/predictions/<run_id>.csv`: file id, actor, true, predicted, 8 class probabilities.
Label map (fixed): `neutral 0, calm 1, happy 2, sad 3, angry 4, fearful 5, disgust 6, surprised 7` (= emotion code − 1).

---

## 4. Validation checklist

Status: **run on 21 Nov 2026** (PC 1, RTX 5080). `python -m src.checks --overfit` → **28/28 PASS** (`results/checks/checks.md`). M4-specific checks (`--with-m4`) not run yet.

| Check | How it is verified | Result |
|---|---|---|
| 1,440 files; neutral 96, others 192; 60/actor; 12 F / 12 M | `data.verify_data` | PASS |
| Label map fixed and saved | `splits/label_map.json`, `verify_data` | PASS |
| Train/val/test actors pairwise disjoint in every round | `data.assert_split_ok` (runs before **every** training run) + `verify_splits` | PASS (also asserted before each of the 33 preliminary runs) |
| Each fold 2 F + 2 M; sizes 960/240/240 | `verify_splits` | PASS |
| Every actor in test exactly once | `verify_splits` | PASS |
| Random split shares actors (count reported) | `verify_splits` | PASS — 24/24 test actors also in train |
| Standardisation fitted on training folds only, per round | `checks.check_pipeline`; `std_fit_actors` in every `runs/*.json` | PASS |
| Augmentation only in the training DataLoader | `checks.check_pipeline` | PASS |
| D-01: augmented copies only for training actors; one version per utterance per epoch (960 samples) | Dataset assertion + `augmentation_audit` in `runs/*.json` + `checks.check_pipeline` | PASS (check) + audit in all 33 preliminary runs: augmented actors == training actors |
| Padding masks: padded-in-batch output == alone (padding filled with garbage) | `checks.check_mask_invariance` — **strict for M1–M3**, approximate/informational for M4 (D-09) | PASS M1–M3 (max diff ≤ 8.8e-06); M4 not run |
| M4 masked pooling uses `_get_feat_extract_output_lengths` | `checks.check_models --with-m4` | not run yet |
| HP search never touches the test fold (D-10) | assertion in `src.hparam_search` | assertion active (search in progress) |
| Test fold untouched before final evaluation | built after training; `AccessCounter` asserts one pass; `test_access_count` in `runs/*.json` | PASS — test_access_count = 240 in all 33 runs |
| UAR == `balanced_accuracy_score`, macro-F1 == `f1_score(macro)`, mean recalls == UAR | `checks.check_metrics` | PASS (|diff| = 0 on 3 test vectors) |
| Seeds (python/numpy/torch/CUDA/workers); rerun ≈ same | `utils.set_seed`; notebook 02 → `results/checks/reproducibility.json` | PASS — identical curves and test UAR on rerun (`results/checks/reproducibility.json`) |
| Versions pinned, hardware + run times logged | `results/env/`, `runs.csv` | PASS — `results/env/` (torch 2.11.0+cu128, Python 3.10.11, RTX 5080) |
| Each model overfits a tiny batch | `checks --overfit` | PASS M1/logmel, M1/mfcc, M2, M3 (train acc 1.000); M4 not run |
| Above chance (UAR 12.5 %) + trivial baselines | `results/tables/baselines.csv` | PASS — all models ≥ 51.7 % UAR; baselines in `results/tables/baselines.csv` |

---

## 5. Dataset

The data are **not included** in this repository.

- **Source:** RAVDESS, audio-only speech subset, Zenodo record 1188976 (https://zenodo.org/records/1188976),
  file `Audio_Speech_Actors_01-24.zip` (1,440 WAV files, 24 actors, 48 kHz, 16-bit).
- **Download:** `python -m src.prepare download` fetches and extracts the zip into `data/raw/`
  (git-ignored), then `python -m src.prepare all` builds the metadata, splits and feature cache.
- **Licence:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).
- **Citation:**
  S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS):
  A dynamic, multimodal set of facial and vocal expressions in North American English," *PLoS ONE*, vol. 13,
  no. 5, e0196391, 2018, doi: 10.1371/journal.pone.0196391.
  Dataset: S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song
  (RAVDESS)," Zenodo, 2018, doi: 10.5281/zenodo.1188976.

---

## 6. Publishing to GitHub (pre-push path scrubbing)

Notebook outputs and logs contain absolute paths of the training machine. Before every push, export a copy
(respecting `.gitignore`) and scrub it; never push the working folder directly:
```
python scripts/scrub_paths.py <export_copy>            # replaces machine paths with <repo>, <project>, <home>
python scripts/scrub_paths.py <export_copy> --check    # must report 0 files
```
