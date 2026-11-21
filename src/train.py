"""Training loop: AdamW, (warm-up), ReduceLROnPlateau on val UAR, early stopping on val UAR
(patience 15), restore best checkpoint, then ONE evaluation on the test fold."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data, evaluate, models, utils
from .features import Standardizer
from .utils import EMOTIONS

RUN_COLUMNS = (["run_id", "tag", "model", "features", "protocol", "fold", "seed", "best_epoch", "epochs_run",
                "val_uar", "val_acc", "val_macro_f1", "test_acc", "test_uar", "test_macro_f1"]
               + [f"recall_{e}" for e in EMOTIONS]
               + ["n_params", "n_trainable", "train_time_s", "device", "gpu", "config_hash", "git_commit",
                  "timestamp", "lr", "batch_size", "dropout", "hparams"])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(tc: dict):
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(tc.get("amp", "none")).lower())


def make_loader(ds, batch_size, shuffle, generator=None, bucket=False, num_workers=0):
    kw = dict(collate_fn=data.collate_pad, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              worker_init_fn=utils.seed_worker if num_workers > 0 else None)
    if bucket:
        sampler = data.BucketBatchSampler(ds.lengths, batch_size, shuffle=shuffle, generator=generator)
        return DataLoader(ds, batch_sampler=sampler, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None,
                      drop_last=False, **kw)


def make_optimizer(model, cfg):
    tc = cfg["train"]
    head_lr = tc.get("head_lr")
    if head_lr and hasattr(model, "head_parameters"):
        head = model.head_parameters()
        ids = {id(p) for p in head}
        body = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
        groups = [{"params": body, "lr": tc["lr"]}, {"params": head, "lr": head_lr}]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": tc["lr"]}]
    return torch.optim.AdamW(groups, lr=tc["lr"], weight_decay=tc["weight_decay"])


def build_datasets(cfg, features, bank, meta, idx, which=("train", "val")):
    """Standardiser is fitted on the CLEAN TRAINING utterances of this round only."""
    labels = meta["label"].to_numpy()
    actors = meta["actor"].to_numpy()
    train_actors = sorted(set(actors[idx["train"]].tolist()))  # augmented copies allowed only for these
    aug = cfg["augment"]
    out = {}
    common = dict(actors=actors, aug_allowed_actors=train_actors)
    if features == "wave":
        std = None
        for k in which:
            out[k] = data.WaveDataset(bank, idx[k], labels, train=(k == "train"), aug_cfg=aug,
                                      max_seconds=cfg["train"].get("max_seconds"), sr=cfg["audio"]["sr"], **common)
    else:
        # D-01: fitted on the ORIGINAL (clean, non-augmented) training utterances only
        std = Standardizer.fit([bank[0][i] for i in idx["train"]])
        for k in which:
            out[k] = data.FeatureDataset(bank, idx[k], labels, standardizer=std, train=(k == "train"),
                                         aug_cfg=aug, **common)
    if "train" in out:
        # D-01: one version per utterance per epoch -> epoch length = number of training utterances
        assert len(out["train"]) == len(idx["train"])
    for k in which:
        if k != "train":
            assert out[k].p_variant == 0 and getattr(out[k], "spec", None) is None, f"{k} must not be augmented"
    return out, std


@torch.no_grad()
def predict(model, loader, device, crit=None, adt=None) -> dict:
    model.eval()
    logits_all, ys, idxs, loss_sum = [], [], [], 0.0
    for b in loader:
        x, mask, y = b["x"].to(device), b["mask"].to(device), b["y"].to(device)
        with torch.autocast(device_type=device.type, dtype=adt, enabled=adt is not None):
            logits = model(x, mask).float()
        if crit is not None:
            loss_sum += float(crit(logits, y)) * len(y)
        logits_all.append(logits.cpu())
        ys.append(b["y"])
        idxs.append(b["idx"])
    logits = torch.cat(logits_all)
    y = torch.cat(ys).numpy()
    probs = torch.softmax(logits, 1).numpy()
    pred = probs.argmax(1)
    m = evaluate.compute_metrics(y, pred)
    m["loss"] = loss_sum / max(len(y), 1)
    m.update({"y": y, "pred": pred, "probs": probs, "idx": torch.cat(idxs).numpy()})
    return m


def train_model(model, cfg, train_loader, val_loader, device, max_epochs=None, verbose=True,
                log_prefix="") -> tuple[dict, list[dict]]:
    """Returns (best dict with state_dict, history). Uses ONLY train + val data."""
    tc = cfg["train"]
    max_epochs = int(max_epochs or tc["max_epochs"])
    adt = amp_dtype(tc)
    opt = make_optimizer(model, cfg)
    pc = tc["plateau"]
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=pc["factor"],
                                                       patience=pc["patience"], min_lr=pc["min_lr"])
    crit = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    accum = int(tc.get("grad_accum", 1))
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    warmup = int(round(float(tc.get("warmup_epochs", 0)) * steps_per_epoch))
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=(adt == torch.float16 and device.type == "cuda"))
    params = [p for p in model.parameters() if p.requires_grad]

    best = {"val_uar": -1.0, "epoch": 0, "state": None, "val": None}
    history, bad, step = [], 0, 0
    if warmup > 0:
        for g in opt.param_groups:
            g["lr"] = 0.0
    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        nb = len(train_loader)
        for bi, b in enumerate(train_loader):
            x, mask, y = b["x"].to(device, non_blocking=True), b["mask"].to(device), b["y"].to(device)
            with torch.autocast(device_type=device.type, dtype=adt, enabled=adt is not None):
                logits = model(x, mask)
            loss = crit(logits.float(), y)
            scaler.scale(loss / accum).backward()
            if (bi + 1) % accum == 0 or bi + 1 == nb:
                if warmup > 0 and step < warmup:
                    for g, lr0 in zip(opt.param_groups, base_lrs):
                        g["lr"] = lr0 * (step + 1) / warmup
                if tc.get("grad_clip"):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            tot_loss += float(loss.detach()) * len(y)
            tot_correct += int((logits.argmax(1) == y).sum())
            tot_n += len(y)
        val = predict(model, val_loader, device, crit, adt)
        lr_now = opt.param_groups[0]["lr"]
        if step >= warmup:
            sched.step(val["uar"])
        row = {"epoch": epoch, "lr": lr_now, "train_loss": tot_loss / tot_n, "train_acc": tot_correct / tot_n,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_uar": val["uar"], "val_macro_f1": val["macro_f1"],
               "epoch_time_s": round(time.time() - t_ep, 2)}
        history.append(row)
        improved = val["uar"] > best["val_uar"] + 1e-9
        if improved:
            best = {"val_uar": val["uar"], "epoch": epoch, "val": {k: val[k] for k in ("acc", "uar", "macro_f1", "loss")},
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        if verbose:
            print(f"{log_prefix}ep {epoch:3d} lr {lr_now:.2e} | train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}"
                  f" | val loss {val['loss']:.3f} UAR {val['uar']:.3f}{' *' if improved else ''}", flush=True)
        if bad >= tc["patience"]:
            break
    return best, history


def run_single(cfg, features, protocol, fold, seed, meta, banks, splits, out_root="results", tag="main",
               eval_test=True, max_epochs=None, save_checkpoint=None, verbose=True, runs_csv_name="runs.csv") -> dict:
    """One (model, features, protocol, fold, seed) run. Writes curves, predictions, run json, runs.csv row."""
    t0 = time.time()
    cfg = copy.deepcopy(cfg)
    if max_epochs is not None:
        cfg["train"]["max_epochs"] = int(max_epochs)
    tc = cfg["train"]
    model_name = cfg["model"]["name"]
    out_root = utils.ensure_dir(utils.resolve(out_root))
    fold_str = str(fold) if protocol == "actor" else "R"
    run_id = f"{model_name}_{features}_{protocol}_f{fold_str}_s{seed}_{tag}"
    chash = utils.config_hash(cfg)

    utils.set_seed(seed, tc.get("deterministic", True))
    device = get_device()
    idx = data.split_indices(meta, protocol, fold, splits)  # asserts disjointness + sizes
    bank = banks[features]
    ds, std = build_datasets(cfg, features, bank, meta, idx, which=("train", "val"))
    g = utils.make_generator(seed)
    bucket = bool(tc.get("bucket_by_length", False))
    train_loader = make_loader(ds["train"], tc["batch_size"], True, g, bucket, tc.get("num_workers", 0))
    val_loader = make_loader(ds["val"], tc["eval_batch_size"], False, None, bucket, tc.get("num_workers", 0))

    in_dim = None if features == "wave" else int(bank[0][0].shape[1])
    model = models.build_model(cfg, in_dim).to(device)
    n_total, n_train = models.count_params(model)
    if verbose:
        print(f"== {run_id} | {device} | params {n_total:,} (trainable {n_train:,}) | cfg {chash}")

    best, history = train_model(model, cfg, train_loader, val_loader, device, verbose=verbose, log_prefix=f"[{run_id}] ")
    model.load_state_dict(best["state"])  # restore best (by val UAR)
    train_time = time.time() - t0

    # ---- D-01 audit: augmented copies were drawn only for training utterances / actors of this round
    tr_ds = ds["train"]
    aug_actors = sorted(set(meta["actor"].to_numpy()[sorted(tr_ds.aug_used)].tolist())) if tr_ds.aug_used else []
    assert tr_ds.aug_used <= set(idx["train"].tolist()), "LEAKAGE: augmented copy of a non-training utterance used"
    assert set(aug_actors) <= set(meta["actor"].to_numpy()[idx["train"]].tolist()), "LEAKAGE: augmented actor"
    aug_audit = {"variant_draws": tr_ds.variant_draws, "utterances_with_variant": len(tr_ds.aug_used),
                 "actors_with_variant": aug_actors, "tracked": tc.get("num_workers", 0) == 0}

    # ---- write curves
    curves_dir = utils.ensure_dir(out_root / "curves")
    pd.DataFrame(history).to_csv(curves_dir / f"{run_id}.csv", index=False)

    res = {"run_id": run_id, "tag": tag, "model": model_name, "features": features, "protocol": protocol,
           "fold": fold_str, "seed": seed, "best_epoch": best["epoch"], "epochs_run": len(history),
           "val_uar": best["val"]["uar"], "val_acc": best["val"]["acc"], "val_macro_f1": best["val"]["macro_f1"],
           "n_params": n_total, "n_trainable": n_train, "train_time_s": round(train_time, 1),
           "device": device.type, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
           "config_hash": chash, "git_commit": utils.git_commit(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "lr": tc["lr"], "batch_size": tc["batch_size"], "dropout": cfg["model"].get("dropout"),
           "hparams": json.dumps({"model": cfg["model"], "lr": tc["lr"], "head_lr": tc.get("head_lr"),
                                  "batch_size": tc["batch_size"], "grad_accum": tc.get("grad_accum", 1),
                                  "weight_decay": tc["weight_decay"], "max_epochs": tc["max_epochs"]}, default=str)}

    # ---- the ONLY contact with the test fold: after training, best checkpoint restored
    test_access = None
    if eval_test:
        labels = meta["label"].to_numpy()
        if features == "wave":
            base_test = data.WaveDataset(bank, idx["test"], labels, train=False,
                                         max_seconds=tc.get("max_seconds"), sr=cfg["audio"]["sr"])
        else:  # same standardiser as training (fitted on training utterances only)
            base_test = data.FeatureDataset(bank, idx["test"], labels, standardizer=std, train=False)
        test_ds = data.AccessCounter(base_test)
        test_loader = make_loader(test_ds, tc["eval_batch_size"], False, None, bucket, 0)
        te = predict(model, test_loader, device, None, amp_dtype(tc))
        test_access = test_ds.count
        assert test_access == len(idx["test"]), "test fold must be read exactly once"
        res.update({"test_acc": te["acc"], "test_uar": te["uar"], "test_macro_f1": te["macro_f1"]})
        res.update({f"recall_{e}": r for e, r in zip(EMOTIONS, te["recall_per_class"])})
        pred_dir = utils.ensure_dir(out_root / "predictions")
        pdf = pd.DataFrame({"file_id": meta["file_id"].to_numpy()[te["idx"]],
                            "actor": meta["actor"].to_numpy()[te["idx"]],
                            "y_true": te["y"], "y_pred": te["pred"]})
        for c, e in enumerate(EMOTIONS):
            pdf[f"p_{e}"] = te["probs"][:, c]
        pdf.to_csv(pred_dir / f"{run_id}.csv", index=False)
        res["test_confusion"] = te["confusion"]

    # ---- checkpoint + run json
    save_ckpt = tc.get("save_checkpoints", True) if save_checkpoint is None else save_checkpoint
    if save_ckpt:
        ck = utils.ensure_dir(utils.CHECKPOINT_DIR / out_root.name)
        torch.save({"state_dict": best["state"], "config": cfg, "standardizer": std.state() if std else None,
                    "run_id": run_id}, ck / f"{run_id}.pt")
    detail = {**res, "config": cfg, "split": {k: [int(i) for i in v] for k, v in idx.items()},
              "std_fit_actors": sorted(set(meta["actor"].to_numpy()[idx["train"]].tolist())),
              "test_access_count": test_access, "augmentation_audit": aug_audit, "history": history}
    utils.save_json(detail, out_root / "runs" / f"{run_id}.json")

    row = {c: res.get(c, "") for c in RUN_COLUMNS}
    csv_path = out_root / runs_csv_name
    df = pd.DataFrame([row], columns=RUN_COLUMNS)
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    if verbose:
        msg = f"   best epoch {best['epoch']} val UAR {res['val_uar']:.4f}"
        if eval_test:
            msg += f" | TEST UAR {res['test_uar']:.4f} acc {res['test_acc']:.4f} F1 {res['test_macro_f1']:.4f}"
        print(msg + f" | {train_time:.0f}s")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def overfit_tiny_batch(cfg, features, meta, banks, splits, n=32, steps=150, lr=None, seed=0, verbose=False) -> dict:
    """Sanity: a model must be able to memorise a tiny balanced batch (no augmentation)."""
    cfg = copy.deepcopy(cfg)
    utils.set_seed(seed)
    device = get_device()
    idx = data.split_indices(meta, "actor", 0, splits)
    y = meta["label"].to_numpy()
    tr = idx["train"]
    per = max(1, n // len(EMOTIONS))
    tiny = np.concatenate([tr[y[tr] == c][:per] for c in range(len(EMOTIONS))])
    bank = banks[features]
    labels = meta["label"].to_numpy()
    if features == "wave":
        ds = data.WaveDataset(bank, tiny, labels, train=False)
        in_dim = None
    else:
        std = Standardizer.fit([bank[0][i] for i in tiny])
        ds = data.FeatureDataset(bank, tiny, labels, standardizer=std, train=False)
        in_dim = int(bank[0][0].shape[1])
    batch = data.collate_pad([ds[i] for i in range(len(ds))])
    x, mask, yt = batch["x"].to(device), batch["mask"].to(device), batch["y"].to(device)
    model = models.build_model(cfg, in_dim).to(device)
    if lr is not None:
        cfg["train"]["lr"] = lr
        cfg["train"]["head_lr"] = None
    opt = make_optimizer(model, cfg)
    crit = nn.CrossEntropyLoss()
    adt = amp_dtype(cfg["train"])
    losses = []
    for s in range(steps):
        model.train()
        with torch.autocast(device_type=device.type, dtype=adt, enabled=adt is not None):
            logits = model(x, mask)
        loss = crit(logits.float(), yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if verbose and s % 25 == 0:
            print(f"step {s} loss {losses[-1]:.4f}")
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=adt, enabled=adt is not None):
        acc = float((model(x, mask).argmax(1) == yt).float().mean())
    return {"model": cfg["model"]["name"], "features": features, "n": len(tiny), "steps": steps,
            "first_loss": losses[0], "final_loss": losses[-1], "train_acc_eval_mode": acc,
            "passed": acc >= 0.95 and losses[-1] < 0.5 * losses[0]}
