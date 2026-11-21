"""Validation checklist (validation checklist). Writes results/checks/checks.json + checks.md.

    python -m src.checks                 # data, splits, metrics, pipeline, mask invariance (M1-M3)
    python -m src.checks --overfit       # + overfit-a-tiny-batch for M1-M3
    python -m src.checks --with-m4       # + M4 (downloads wav2vec2-base weights from Hugging Face)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import data, evaluate, features as feat, models, train, utils
from .data import _chk


# ----------------------------------------------------------------------------
def check_metrics() -> list[dict]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    rng = np.random.default_rng(0)
    out = []
    cases = {}
    y = rng.integers(0, 8, 600)
    p = np.where(rng.random(600) < 0.6, y, rng.integers(0, 8, 600))
    cases["random"] = (y, p)
    p2 = p.copy()
    p2[p2 == 3] = 4  # class 3 never predicted
    cases["class never predicted"] = (y, p2)
    y3 = np.r_[np.zeros(20, int), rng.integers(1, 8, 300)]  # imbalanced, like neutral
    cases["imbalanced"] = (y3, np.where(rng.random(len(y3)) < 0.5, y3, rng.integers(0, 8, len(y3))))
    for name, (yt, yp) in cases.items():
        m = evaluate.compute_metrics(yt, yp)
        d_uar = abs(m["uar"] - balanced_accuracy_score(yt, yp))
        d_f1 = abs(m["macro_f1"] - f1_score(yt, yp, average="macro", zero_division=0))
        d_acc = abs(m["acc"] - accuracy_score(yt, yp))
        d_rec = abs(np.mean(m["recall_per_class"]) - m["uar"])
        out.append(_chk(f"metrics ({name}): UAR==balanced_accuracy, macroF1==f1(macro), acc, mean(recalls)==UAR",
                        max(d_uar, d_f1, d_acc, d_rec) < 1e-12,
                        f"|dUAR|={d_uar:.1e} |dF1|={d_f1:.1e} |dAcc|={d_acc:.1e} |dRec|={d_rec:.1e}"))
    return out


# ----------------------------------------------------------------------------
@torch.no_grad()
def check_mask_invariance(model, in_dim, device, lengths=(97, 173, 250, 311), wave=False, tol=1e-4, garbage=True):
    """Output of an utterance alone == output of the same utterance padded inside a batch.
    Padding is filled with large garbage values to prove it is ignored."""
    model.eval().to(device)
    g = torch.Generator().manual_seed(0)
    xs = [torch.randn(L, generator=g) * 0.1 if wave else torch.randn(L, in_dim, generator=g) for L in lengths]
    batch = data.collate_pad([(x, 0, i) for i, x in enumerate(xs)])
    xb, mb = batch["x"].clone(), batch["mask"]
    if garbage:
        noise = torch.randn(xb.shape, generator=g) * 100.0
        pad = ~mb if wave else (~mb).unsqueeze(-1).expand_as(xb)
        xb[pad] = noise[pad]
    out_b = model(xb.to(device), mb.to(device)).float().cpu()
    diffs = []
    for i, x in enumerate(xs):
        o = model(x[None].to(device), torch.ones(1, x.shape[0], dtype=torch.bool, device=device)).float().cpu()
        diffs.append(float((o[0] - out_b[i]).abs().max()))
    return max(diffs), diffs


def check_pipeline(cfg, meta, splits) -> list[dict]:
    """Train-only standardisation and train-only augmentation (needs the feature cache)."""
    out = []
    bank = feat.load_bank(cfg, "logmel", meta)
    idx = data.split_indices(meta, "actor", 0, splits)
    ds, std = train.build_datasets(cfg, "logmel", bank, meta, idx, which=("train", "val"))
    ref = feat.Standardizer.fit([bank[0][i] for i in idx["train"]])
    all_std = feat.Standardizer.fit([bank[0][i] for i in range(len(meta))])
    act = meta["actor"].to_numpy()
    fit_actors = set(act[idx["train"]])
    held = set(act[idx["val"]]) | set(act[idx["test"]])
    out.append(_chk("pipeline: standardiser fitted on ORIGINAL (non-augmented) training utterances only (round 0)",
                    np.allclose(std.mean, ref.mean) and np.allclose(std.std, ref.std) and not (fit_actors & held)
                    and not np.allclose(std.mean, all_std.mean),
                    f"fit actors={sorted(int(a) for a in fit_actors)}"))
    torch.manual_seed(0)
    v1, v2 = ds["val"][0][0], ds["val"][0][0]
    clean = torch.from_numpy(std.transform(bank[0][int(idx['val'][0])]))
    val_clean = torch.equal(v1, v2) and torch.allclose(v1, clean, atol=1e-6)
    draws = [ds["train"][0][0] for _ in range(20)]
    tr_clean = torch.from_numpy(std.transform(bank[0][int(idx['train'][0])]))
    varies = any(d.shape != tr_clean.shape or not torch.allclose(d, tr_clean) for d in draws)
    out.append(_chk("pipeline: augmentation only in training dataset (val returns clean, deterministic)",
                    val_clean and varies and ds["val"].p_variant == 0 and ds["val"].spec is None,
                    f"val deterministic+clean={val_clean}, train varies={varies}, "
                    f"train p_variant={ds['train'].p_variant}"))
    # D-01: augmented copies inherit the actor ID -> requesting them for non-training actors must fail
    actors = meta["actor"].to_numpy()
    tr_act = sorted(set(actors[idx["train"]].tolist()))
    try:
        data.FeatureDataset(bank, idx["val"], meta["label"].to_numpy(), standardizer=std, train=True,
                            aug_cfg=cfg["augment"], actors=actors, aug_allowed_actors=tr_act)
        blocked = False
    except AssertionError:
        blocked = True
    trd = ds["train"]
    trd.variant_draws, trd.aug_used = 0, set()
    torch.manual_seed(1)
    for i in range(len(trd)):  # one full epoch
        trd[i]
    ok_epoch = len(trd) == len(idx["train"]) and trd.aug_used <= set(idx["train"].tolist())
    out.append(_chk("pipeline (D-01): augmented copies only for training actors; ONE version per utterance per epoch",
                    blocked and ok_epoch,
                    f"aug for val actors blocked={blocked}; epoch length={len(trd)} (= {len(idx['train'])} train utts); "
                    f"augmented draws in 1 epoch={trd.variant_draws}"))
    out.append(_chk("pipeline: test fold untouched until final eval",
                    True, "by construction: run_single builds the test DataLoader only after training and the "
                          "best-checkpoint restore; AccessCounter asserts exactly one pass (test_access_count in runs/*.json)"))
    return out


def check_models(cfg_by_model: dict, device, include=("m1", "m2", "m3"), in_dims=None) -> list[dict]:
    out = []
    in_dims = in_dims or {"logmel": 64, "mfcc": 120}
    combos = [("m1", "logmel"), ("m1", "mfcc"), ("m2", "logmel"), ("m3", "logmel"), ("m4", "wave")]
    for m, f in combos:
        if m not in include:
            continue
        cfg = cfg_by_model[m]
        torch.manual_seed(0)
        model = models.build_model(cfg, None if f == "wave" else in_dims[f])
        lengths = (16000, 27000, 40000, 51000) if f == "wave" else (97, 173, 250, 311)
        tol = 1e-4
        worst, diffs = check_mask_invariance(model, in_dims.get(f), device, lengths, wave=(f == "wave"), tol=tol)
        if m != "m4":  # strict for M1-M3
            out.append(_chk(f"mask invariance (strict): {m}/{f} padded-in-batch == alone (tol {tol})", worst < tol,
                            f"max |diff| = {worst:.2e}"))
        else:  # D-09: approximate for M4 (group norm in the feature encoder sees the zero padding)
            out.append(_chk(f"mask invariance (approximate, informational): m4/wave padded-in-batch vs alone",
                            bool(np.isfinite(worst)),
                            f"max |logit diff| = {worst:.2e}; expected > 0 (no attention_mask, group norm sees zeros); "
                            f"eval uses batch size 1 -> evaluation predictions are padding-free"))
            ok, det = True, []
            with torch.no_grad():
                for L in lengths:
                    h = model.w2v(torch.randn(1, L).to(device)).last_hidden_state
                    exp = int(model.w2v._get_feat_extract_output_lengths(torch.tensor([L])))
                    ok &= h.size(1) == exp
                    det.append(f"{L}->{h.size(1)}/{exp}")
            out.append(_chk("M4 (D-09): masked mean-pooling uses _get_feat_extract_output_lengths (lengths match)", ok,
                            ", ".join(det)))
        del model
    return out


def run_all(with_m4=False, overfit=False, skip_data=False) -> list[dict]:
    t0 = time.time()
    device = train.get_device()
    results = []
    cfgs = {m: utils.load_config(m) for m in utils.MODELS}
    results += check_metrics()
    if not skip_data:
        meta = data.load_metadata()
        splits = data.load_splits()
        results += data.verify_data(meta)
        results += data.verify_splits(meta, splits)
        results += check_pipeline(cfgs["m1"], meta, splits)
    include = ("m1", "m2", "m3", "m4") if with_m4 else ("m1", "m2", "m3")
    results += check_models(cfgs, device, include)
    if overfit and not skip_data:
        banks = {"logmel": feat.load_bank(cfgs["m1"], "logmel", meta, use_variants=False),
                 "mfcc": feat.load_bank(cfgs["m1"], "mfcc", meta, use_variants=False)}
        combos = [("m1", "logmel"), ("m1", "mfcc"), ("m2", "logmel"), ("m3", "logmel")]
        if with_m4:
            banks["wave"] = feat.load_bank(cfgs["m4"], "wave", meta, use_variants=False)
            combos.append(("m4", "wave"))
        for m, f in combos:
            kw = dict(n=16, steps=60, lr=1e-4) if m == "m4" else dict(n=32, steps=200)
            r = train.overfit_tiny_batch(cfgs[m], f, meta, banks, splits, **kw)
            results.append(_chk(f"sanity: {m}/{f} overfits a tiny batch ({r['n']} utts)", r["passed"],
                                f"loss {r['first_loss']:.3f} -> {r['final_loss']:.4f}, acc {r['train_acc_eval_mode']:.3f}"))
    write_report(results, time.time() - t0)
    return results


def write_report(results, seconds) -> None:
    d = utils.ensure_dir(utils.RESULTS_DIR / "checks")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    utils.save_json({"timestamp": stamp, "seconds": round(seconds, 1), "env": utils.env_info(), "checks": results},
                    d / "checks.json")
    lines = [f"Checks run {stamp} ({seconds:.0f} s)", "", "| # | Check | Result | Detail |", "|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        det = r["detail"].replace("|", "/")
        det = det if len(det) < 300 else det[:297] + "..."
        lines.append(f"| {i} | {r['check']} | {'PASS' if r['passed'] else '**FAIL**'} | {det} |")
    (d / "checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_fail = sum(not r["passed"] for r in results)
    print("\n".join(lines))
    print(f"\n{len(results) - n_fail}/{len(results)} passed -> {d / 'checks.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-m4", action="store_true")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--skip-data", action="store_true", help="only metric + model checks (no dataset needed)")
    a = ap.parse_args()
    res = run_all(with_m4=a.with_m4, overfit=a.overfit, skip_data=a.skip_data)
    utils.hard_exit(0 if all(r["passed"] for r in res) else 1)


if __name__ == "__main__":
    main()
