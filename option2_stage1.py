"""
option2_stage1.py  -  Stage 1 of Option 2.

Retrains the surrogate with ONLY 4 parameters (Kf, Cf, K2, K3),
holding damper shape fixed at nominal values, then runs Bayesian
optimisation to find optimal spring/damper rates.

Usage
-----
# Generate new data with 4-param LHS (or reuse existing if bounds match):
python src/option2_stage1.py --mode generate --n_cases 80 --n_jobs -1

# Train 4-param surrogate:
python src/option2_stage1.py --mode train --data data/opt2_train.csv

# Optimise:
python src/option2_stage1.py --mode optimise
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import BASE_CFG, BAYES_CFG, DT, TRAIN_CFG
from data_gen import run_lhs_grid
from dataset import CabinDataset, NormStats, build_loaders
from model import PhysicsLSTM
from physics import precompute_road_array
from train import train
from optimise import RMSEmulator

# ─────────────────────────────────────────────────────────────
# Option 2 uses only 4 optimisation parameters.
# Damper shape params are fixed at nominal during Stage 1.
# ─────────────────────────────────────────────────────────────

OPT2_PARAM_NAMES = ["K_f", "C_f", "K_2", "K_3"]

OPT2_PARAM_BOUNDS = {
    "K_f": (0.70 * 474257,  1.30 * 474257),
    "C_f": (0.50 * 15000,   2.00 * 15000),
    "K_2": (0.70 * 1077620, 1.30 * 1077620),
    "K_3": (0.70 * 1077620, 1.30 * 1077620),
}

# Nominal damper shape held fixed during Stage 1
NOMINAL_DAMPER = {
    "cs_minus":   0.3,
    "asym_ratio": 3.0,
    "gamma_c":    0.12,
    "gamma_r":    0.09,
}


# ─────────────────────────────────────────────────────────────
# Patched NormStats for 4-param mode
# ─────────────────────────────────────────────────────────────

class NormStats4(NormStats):
    """NormStats subclass that uses only the 4 spring/damper params."""

    def __init__(self, base: NormStats):
        # Copy all fields from base
        super().__init__(
            r_mean=base.r_mean, r_std=base.r_std,
            s_mean=base.s_mean, s_std=base.s_std,
            a_mean=base.a_mean, a_std=base.a_std,
            p_lo=base.p_lo[:4], p_hi=base.p_hi[:4],
        )


# ─────────────────────────────────────────────────────────────
# 4-param dataset wrapper
# ─────────────────────────────────────────────────────────────

class CabinDataset4(CabinDataset):
    """
    Loads CSV but only uses the 4 spring/damper parameters.
    Damper shape columns may or may not be present in the CSV;
    they are ignored either way.
    """

    def __init__(self, csv_path, norm_stats, **kwargs):
        # Temporarily patch PARAM_NAMES globally for parent __init__
        import config as cfg_mod
        original = cfg_mod.PARAM_NAMES[:]
        cfg_mod.PARAM_NAMES = OPT2_PARAM_NAMES
        try:
            super().__init__(csv_path, norm_stats, **kwargs)
        finally:
            cfg_mod.PARAM_NAMES = original


# ─────────────────────────────────────────────────────────────
# 4-param surrogate model (n_params=4 instead of 8)
# ─────────────────────────────────────────────────────────────

def build_4param_model(train_cfg=None):
    tc = train_cfg or TRAIN_CFG
    return PhysicsLSTM(
        road_dim=tc["road_dim"],
        param_dim=tc["param_dim"],
        lstm_hidden=tc["lstm_hidden"],
        lstm_layers=tc["lstm_layers"],
        dropout=tc["dropout"],
        n_states=6,
        n_accels=3,
    )


# ─────────────────────────────────────────────────────────────
# 4-param RMS emulator
# ─────────────────────────────────────────────────────────────

class RMSEmulator4(RMSEmulator):
    """
    Emulator that only accepts the 4 spring/damper params.
    Uses OPT2_PARAM_NAMES and OPT2_PARAM_BOUNDS for normalisation.
    """

    def __init__(self, model, road_norm, ns, cfg, device="cpu"):
        super().__init__(model, road_norm, ns, cfg, device)
        # Override with 4-param bounds
        self.ns.p_lo = np.array(
            [OPT2_PARAM_BOUNDS[k][0] for k in OPT2_PARAM_NAMES], np.float32)
        self.ns.p_hi = np.array(
            [OPT2_PARAM_BOUNDS[k][1] for k in OPT2_PARAM_NAMES], np.float32)

    @torch.no_grad()
    def __call__(self, params: dict) -> float:
        p_raw  = np.array([float(params[k]) for k in OPT2_PARAM_NAMES], np.float32)
        p_norm = (p_raw - self.ns.p_lo) / (self.ns.p_hi - self.ns.p_lo + 1e-8)
        p_t    = torch.tensor(p_norm, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        _, ap  = self.model(self.road_t, p_t)
        a      = ap[0].cpu().numpy() * self.ns.a_std + self.ns.a_mean
        a_seat = a[:, 0] + self.h_seat * a[:, 2]
        return float(np.sqrt(np.mean(a_seat[self.n_skip:]**2)))

    def denormalise_params(self, p_norm):
        p_phys = self.ns.p_lo + p_norm * (self.ns.p_hi - self.ns.p_lo)
        return {k: float(p_phys[j]) for j, k in enumerate(OPT2_PARAM_NAMES)}


# ─────────────────────────────────────────────────────────────
# Stage 1 Bayesian optimisation (4 params)
# ─────────────────────────────────────────────────────────────

def run_opt2_botorch(emulator: RMSEmulator4,
                     n_init=12, n_iter=60, seed=42) -> dict:
    try:
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from scipy.stats import qmc
    except ImportError:
        raise ImportError("pip install botorch")

    dim    = len(OPT2_PARAM_NAMES)
    bounds = torch.zeros(2, dim, dtype=torch.double)
    bounds[1, :] = 1.0
    torch.manual_seed(seed)

    def eval_norm(x_np):
        return emulator(emulator.denormalise_params(x_np.astype(np.float32)))

    sobol = qmc.Sobol(d=dim, scramble=True, seed=seed)
    X0    = sobol.random(n_init).astype(np.float64)
    Y0    = np.array([[eval_norm(x)] for x in X0])
    print(f"  Initial {n_init} samples: RMS in [{Y0.min():.4f}, {Y0.max():.4f}]")

    train_X = torch.tensor(X0, dtype=torch.double)
    train_Y = -torch.tensor(Y0, dtype=torch.double)
    best_rms = float(Y0.min())
    best_x   = X0[int(Y0.argmin())]

    for it in range(n_iter):
        try:
            gp  = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            acqf = LogExpectedImprovement(gp, best_f=train_Y.max())
            cand, _ = optimize_acqf(acqf, bounds=bounds, q=1,
                                     num_restarts=10, raw_samples=256)
            x_new = cand[0].detach().numpy().astype(np.float64)
        except Exception as e:
            print(f"  BoTorch iter {it+1} failed ({e}); random fallback")
            x_new = np.random.default_rng(seed + it).random(dim)

        y_new = eval_norm(x_new)
        print(f"  iter {it+1:3d}/{n_iter}  RMS={y_new:.5f}  "
              f"best={min(best_rms, y_new):.5f}")

        train_X = torch.cat([train_X,
                              torch.tensor(x_new, dtype=torch.double).unsqueeze(0)])
        train_Y = torch.cat([train_Y,
                              torch.tensor([[-y_new]], dtype=torch.double)])
        if y_new < best_rms:
            best_rms = y_new; best_x = x_new

    best_params = emulator.denormalise_params(best_x.astype(np.float32))
    return {"params": best_params, "rms_surrogate": best_rms}


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def run_stage1(mode="all", n_cases=80, n_jobs=1,
               data_csv="data/opt2_train.csv",
               ckpt_dir="checkpoints/opt2",
               cfg_base=None, train_cfg=None):

    cfg_base  = cfg_base  or dict(BASE_CFG)
    train_cfg = train_cfg or dict(TRAIN_CFG)
    # Fix damper shape at nominal for all generated cases
    cfg_base.update(NOMINAL_DAMPER)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_int = int(train_cfg.get("downsample", 4))
    t_eval = np.arange(0.0, float(cfg_base["sim_duration_s"]) + DT, DT)[::ds_int]

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── generate ─────────────────────────────────────────────
    if mode in ("generate", "all"):
        print("\n=== Option 2 Stage 1: generating 4-param LHS data ===")

        # Temporarily override PARAM_BOUNDS and PARAM_NAMES in data_gen
        import config as cfg_mod
        import data_gen as dg_mod
        orig_names  = cfg_mod.PARAM_NAMES[:]
        orig_bounds = dict(cfg_mod.PARAM_BOUNDS)
        cfg_mod.PARAM_NAMES  = OPT2_PARAM_NAMES
        cfg_mod.PARAM_BOUNDS = OPT2_PARAM_BOUNDS
        dg_mod.PARAM_BOUNDS  = OPT2_PARAM_BOUNDS  # data_gen imports it at module level

        try:
            run_lhs_grid(n_cases, cfg_base, out_csv=data_csv,
                         seed=42, downsample=ds_int, n_jobs=n_jobs)
        finally:
            cfg_mod.PARAM_NAMES  = orig_names
            cfg_mod.PARAM_BOUNDS = orig_bounds
            dg_mod.PARAM_BOUNDS  = orig_bounds

    # ── train ────────────────────────────────────────────────
    if mode in ("train", "all"):
        print("\n=== Option 2 Stage 1: training 4-param surrogate ===")

        # Patch model to use 4 input params
        import config as cfg_mod
        orig_names = cfg_mod.PARAM_NAMES[:]
        cfg_mod.PARAM_NAMES = OPT2_PARAM_NAMES

        # Override PhysicsLSTM param input size to 4
        import model as model_mod
        orig_ParamMLP = model_mod.ParamMLP

        class ParamMLP4(model_mod.ParamMLP):
            def __init__(self, n_params=4, hidden=64, out_dim=128):
                super().__init__(n_params=4, hidden=hidden, out_dim=out_dim)
        model_mod.ParamMLP = ParamMLP4

        try:
            train(data_csv=data_csv, out_dir=ckpt_dir,
                  cfg_base=cfg_base, train_cfg=train_cfg)
        finally:
            cfg_mod.PARAM_NAMES = orig_names
            model_mod.ParamMLP  = orig_ParamMLP

    # ── optimise ─────────────────────────────────────────────
    if mode in ("optimise", "all"):
        print("\n=== Option 2 Stage 1: Bayesian optimisation (4 params) ===")

        ns_path   = Path(ckpt_dir) / "norm_stats.npz"
        ckpt_path = Path(ckpt_dir) / "best.pt"
        ns        = NormStats.load(str(ns_path))

        # 4-param model
        model = PhysicsLSTM(
            road_dim=train_cfg["road_dim"],
            param_dim=train_cfg["param_dim"],
            lstm_hidden=train_cfg["lstm_hidden"],
            lstm_layers=train_cfg["lstm_layers"],
            dropout=0.0,
        )
        # Replace param MLP with 4-input version
        from model import ParamMLP
        model.param_mlp = ParamMLP(n_params=4, hidden=64,
                                    out_dim=train_cfg["param_dim"])
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model"])

        road_arr  = precompute_road_array(cfg_base, t_eval)
        road_norm = (road_arr - ns.r_mean) / ns.r_std

        class _NS4:
            p_lo   = np.array([OPT2_PARAM_BOUNDS[k][0] for k in OPT2_PARAM_NAMES], np.float32)
            p_hi   = np.array([OPT2_PARAM_BOUNDS[k][1] for k in OPT2_PARAM_NAMES], np.float32)
            a_std  = ns.a_std
            a_mean = ns.a_mean

        emulator = RMSEmulator4(model, road_norm, _NS4(), cfg_base, device)

        nominal4 = {k: float(cfg_base[k]) for k in OPT2_PARAM_NAMES}
        print(f"Baseline RMS (nominal): {emulator(nominal4):.5f} m/s²")

        result = run_opt2_botorch(emulator,
                                   n_init=BAYES_CFG["n_init"],
                                   n_iter=BAYES_CFG["n_iter"],
                                   seed=BAYES_CFG["seed"])

        out_path = Path(ckpt_dir) / "stage1_result.json"
        with open(str(out_path), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nStage 1 optimal params saved → {out_path}")
        print(f"  Surrogate RMS = {result['rms_surrogate']:.5f} m/s²")
        for k, v in result["params"].items():
            nom = float(cfg_base[k])
            print(f"  {k:<12} {nom:>10.2f}  →  {v:>10.2f}  "
                  f"({(v-nom)/nom*100:+.1f}%)")
        return result

    return None


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",     default="all",
                    choices=["generate", "train", "optimise", "all"])
    ap.add_argument("--n_cases",  type=int, default=80)
    ap.add_argument("--n_jobs",   type=int, default=1)
    ap.add_argument("--data",     default="data/opt2_train.csv")
    ap.add_argument("--ckpt_dir", default="checkpoints/opt2")
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()

    cfg = dict(BASE_CFG)
    if args.data_dir:
        for side in ("front_left", "front_right",
                     "rear1_left", "rear1_right",
                     "rear2_left", "rear2_right"):
            ax = "front" if "front" in side else ("rear1" if "1" in side else "rear2")
            sd = "left" if "left" in side else "right"
            cfg[f"axle{ax}_{sd}_csv"] = f"{args.data_dir}/{side}.csv"

    run_stage1(mode=args.mode, n_cases=args.n_cases, n_jobs=args.n_jobs,
               data_csv=args.data, ckpt_dir=args.ckpt_dir, cfg_base=cfg)
