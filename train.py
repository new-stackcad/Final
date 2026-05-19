"""
train.py  –  training loop with early stopping, checkpointing, logging.

Usage
-----
python src/train.py --data data/physics_train.csv --out checkpoints/
python src/train.py --data data/physics_train.csv --out checkpoints/ --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from config import BASE_CFG, DT, TRAIN_CFG
from dataset import CabinDataset, NormStats, build_loaders
from model import PhysicsLoss, PhysicsLSTM
from physics import precompute_road_array


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

_ZERO_METRICS = {
    "loss_total": 0.0, "loss_state": 0.0,
    "loss_accel": 0.0, "loss_rms":   0.0, "loss_phys": 0.0,
    "loss_ac":    0.0,
}

def _mean_metrics(ms: List[Dict]) -> Dict:
    if not ms:
        return dict(_ZERO_METRICS)
    return {k: float(np.mean([d[k] for d in ms])) for k in ms[0]}


def _run_epoch(model:   PhysicsLSTM,
               loader:  DataLoader,
               crit:    PhysicsLoss,
               opt:     torch.optim.Optimizer | None,
               device:  str,
               h_seat:  float,
               dt:      float,
               ) -> Dict:
    train = opt is not None
    model.train(train)
    metrics = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            road      = batch["road"].to(device)
            param     = batch["param"].to(device)
            state     = batch["state"].to(device)
            accel     = batch["accel"].to(device)
            rms       = batch["rms"].to(device)
            accel_raw = batch.get("accel_raw", None)
            a_std     = batch.get("a_std",     None)
            if accel_raw is not None: accel_raw = accel_raw.to(device)
            if a_std     is not None: a_std     = a_std.to(device)

            sp, ap = model(road, param)
            loss, m = crit(sp, ap, state, accel, rms,
                           h_seat=h_seat, dt=dt,
                           accel_raw=accel_raw, a_std=a_std)

            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            metrics.append(m)

    return _mean_metrics(metrics)


# ─────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────

def train(
    data_csv:   str,
    out_dir:    str   = "checkpoints",
    cfg_base:   Dict  = None,
    train_cfg:  Dict  = None,
    resume:     str   = None,   # path to checkpoint to resume from
) -> PhysicsLSTM:

    cfg_base   = cfg_base  or BASE_CFG
    train_cfg  = train_cfg or TRAIN_CFG
    out_path   = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    h_seat     = float(cfg_base.get("h_seat", 0.6))
    downsample = int(train_cfg.get("downsample", 4))
    dt_stored  = DT * downsample
    epochs     = int(train_cfg["epochs"])
    patience   = int(train_cfg["patience"])
    lr         = float(train_cfg["lr"])
    bs         = int(train_cfg["batch_size"])

    print(f"Device : {device}")
    print(f"Data   : {data_csv}")
    print(f"Output : {out_dir}")

    # ── pre-compute road signals ─────────────────────────────
    print("Pre-computing road signals …")
    import numpy as np
    t_eval   = np.arange(0.0, float(cfg_base["sim_duration_s"]) + DT, DT)[::downsample]
    road_arr = precompute_road_array(cfg_base, t_eval)   # [T, 6]  float32

    # ── normalisation stats ──────────────────────────────────
    df_train_rows = pd.read_csv(data_csv)
    if "split" in df_train_rows.columns:
        df_train_rows = df_train_rows[df_train_rows["split"] == "train"]
    ns = NormStats.from_dataframe(df_train_rows)
    ns.save(str(out_path / "norm_stats.npz"))

    # ── dataloaders ──────────────────────────────────────────
    train_ld, val_ld, test_ld = build_loaders(
        data_csv, ns, batch_size=bs,
        h_seat=h_seat, road_arr=road_arr,
        num_workers=0,
        cfg=cfg_base,
    )

    # ── model ────────────────────────────────────────────────
    model = PhysicsLSTM(
        road_dim=train_cfg["road_dim"],
        param_dim=train_cfg["param_dim"],
        lstm_hidden=train_cfg["lstm_hidden"],
        lstm_layers=train_cfg["lstm_layers"],
        dropout=train_cfg["dropout"],
    ).to(device)
    print(f"Model  : {model.n_params:,} trainable parameters")

    start_epoch = 1
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {resume}  (epoch {start_epoch-1})")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=float(train_cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-6)
    crit  = PhysicsLoss(
        lambda_state=float(train_cfg["lambda_state"]),
        lambda_accel=float(train_cfg["lambda_accel"]),
        lambda_rms=float(train_cfg["lambda_rms"]),
        lambda_phys=float(train_cfg["lambda_phys"]),
    )

    # ── training loop ────────────────────────────────────────
    best_val   = float("inf")
    no_improve = 0
    history    = []
    best_path  = out_path / "best.pt"

    for ep in range(start_epoch, start_epoch + epochs):
        t0 = time.perf_counter()

        tr_m = _run_epoch(model, train_ld, crit, opt,   device, h_seat, dt_stored)
        vl_m = _run_epoch(model, val_ld,   crit, None,  device, h_seat, dt_stored)
        sched.step()

        elapsed = time.perf_counter() - t0
        row = {"epoch": ep,
               **{f"tr_{k}": v for k, v in tr_m.items()},
               **{f"vl_{k}": v for k, v in vl_m.items()},
               "lr": sched.get_last_lr()[0],
               "elapsed_s": elapsed}
        history.append(row)

        val_loss = vl_m["loss_total"]
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({"model": model.state_dict(),
                        "epoch": ep,
                        "val_loss": best_val,
                        "train_cfg": train_cfg}, str(best_path))
        else:
            no_improve += 1

        if ep % 10 == 0 or ep == start_epoch:
            print(f"Ep {ep:4d}/{start_epoch+epochs-1}  "
                  f"tr={tr_m['loss_total']:.4e}  "
                  f"vl={vl_m['loss_total']:.4e}  "
                  f"rms={vl_m['loss_rms']:.4e}  "
                  f"ac={vl_m.get('loss_ac',0):.4e}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  "
                  f"({elapsed:.0f}s)")

        if no_improve >= patience:
            print(f"Early stopping at epoch {ep}  "
                  f"(best val={best_val:.4e}, no improvement for {patience} epochs)")
            break

    # ── save history ─────────────────────────────────────────
    hist_path = out_path / "history.csv"
    pd.DataFrame(history).to_csv(str(hist_path), index=False)
    print(f"\nHistory saved → {hist_path}")

    # ── load best weights ─────────────────────────────────────
    ckpt = torch.load(str(best_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Best model loaded from epoch {ckpt['epoch']}  "
          f"(val={ckpt['val_loss']:.4e})")

    # ── evaluate on test set ──────────────────────────────────
    print("\n=== Test set evaluation ===")
    test_m = _run_epoch(model, test_ld, crit, None, device, h_seat, dt_stored)
    print("  " + "  ".join(f"{k}={v:.4e}" for k, v in test_m.items()))
    with open(str(out_path / "test_metrics.json"), "w") as f:
        json.dump(test_m, f, indent=2)
    print(f"Test metrics saved → {out_path / 'test_metrics.json'}")

    return model


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PhysicsLSTM surrogate")
    parser.add_argument("--data",    required=True,
                        help="Path to physics_train.csv")
    parser.add_argument("--out",     default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--batch",   type=int, default=None)
    parser.add_argument("--resume",  type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override road CSV directory")
    args = parser.parse_args()

    cfg   = dict(BASE_CFG)
    trcfg = dict(TRAIN_CFG)
    if args.epochs: trcfg["epochs"]     = args.epochs
    if args.lr:     trcfg["lr"]         = args.lr
    if args.batch:  trcfg["batch_size"] = args.batch
    if args.data_dir:
        d = args.data_dir
        for side in ("front_left", "front_right",
                     "rear1_left", "rear1_right",
                     "rear2_left", "rear2_right"):
            key = ("axle" + ("front" if "front" in side else
                             "rear1" if "rear1" in side else "rear2")
                   + "_" + ("left" if "left" in side else "right") + "_csv")
            cfg[key] = f"{d}/{side}.csv"

    train(args.data, args.out, cfg_base=cfg, train_cfg=trcfg, resume=args.resume)
