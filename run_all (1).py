"""
run_all.py  –  single entry point that runs the full pipeline end-to-end.

Usage
-----
# Full pipeline (generate → train → evaluate → optimise):
python run_all.py --data_dir data/ --mode all --n_cases 80 --n_jobs 8

# Individual stages:
python run_all.py --data_dir data/ --mode generate --n_cases 80 --n_jobs 8
python run_all.py --data_dir data/ --mode train
python run_all.py --data_dir data/ --mode evaluate
python run_all.py --data_dir data/ --mode optimise
python run_all.py --data_dir data/ --mode optimise --verify_ode

# Quick smoke-test (2 ODE cases, 5 training epochs):
python run_all.py --data_dir data/ --mode all --n_cases 2 --n_jobs 1 --smoke_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import BASE_CFG, BAYES_CFG, TRAIN_CFG


def make_cfg(data_dir: str) -> dict:
    cfg = dict(BASE_CFG)
    cfg.update({
        "axlefront_left_csv":  f"{data_dir}/front_left.csv",
        "axlefront_right_csv": f"{data_dir}/front_right.csv",
        "axlerear1_left_csv":  f"{data_dir}/rear1_left.csv",
        "axlerear1_right_csv": f"{data_dir}/rear1_right.csv",
        "axlerear2_left_csv":  f"{data_dir}/rear2_left.csv",
        "axlerear2_right_csv": f"{data_dir}/rear2_right.csv",
    })
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Cabin seat RMS surrogate pipeline")
    parser.add_argument("--data_dir",   required=True,
                        help="Directory containing the 6 road CSV files")
    parser.add_argument("--mode",       default="all",
                        choices=["generate", "train", "evaluate",
                                 "optimise", "all"],
                        help="Pipeline stage to run (default: all)")
    parser.add_argument("--n_cases",    type=int,   default=80,
                        help="LHS cases to generate (default 80)")
    parser.add_argument("--n_jobs",     type=int,   default=1,
                        help="Parallel ODE workers; -1=all cores (default 1)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="2 ODE cases + 5 training epochs for quick check")
    parser.add_argument("--verify_ode", action="store_true",
                        help="Verify optimal params with full ODE solve")
    parser.add_argument("--n_init",     type=int,   default=None)
    parser.add_argument("--n_iter",     type=int,   default=None)
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Checkpoint to resume training from")
    args = parser.parse_args()

    cfg      = make_cfg(args.data_dir)
    train_cfg = dict(TRAIN_CFG)
    bayes_cfg = dict(BAYES_CFG)

    if args.smoke_test:
        args.n_cases        = 2
        train_cfg["epochs"] = 5
        train_cfg["patience"] = 5
        bayes_cfg["n_init"] = 3
        bayes_cfg["n_iter"] = 5
        print("=== SMOKE TEST MODE ===")

    if args.n_init: bayes_cfg["n_init"] = args.n_init
    if args.n_iter: bayes_cfg["n_iter"] = args.n_iter

    # ── paths ─────────────────────────────────────────────────
    data_csv  = "data/physics_train.csv"
    ckpt_dir  = "checkpoints"
    ckpt_path = f"{ckpt_dir}/best.pt"
    norms_path = f"{ckpt_dir}/norm_stats.npz"
    eval_dir  = "outputs/eval"
    opt_dir   = "outputs/opt"

    # ─────────────────────────────────────────────────────────
    # Stage 1: generate
    # ─────────────────────────────────────────────────────────
    if args.mode in ("generate", "all"):
        print("\n" + "="*60)
        print(" STAGE 1 — ODE data generation")
        print("="*60)
        from data_gen import run_lhs_grid
        run_lhs_grid(
            n_cases=args.n_cases,
            cfg_base=cfg,
            out_csv=data_csv,
            seed=42,
            downsample=train_cfg.get("downsample", 4),
            n_jobs=args.n_jobs,
            test_frac=0.20,
        )

    # ─────────────────────────────────────────────────────────
    # Stage 2: train
    # ─────────────────────────────────────────────────────────
    if args.mode in ("train", "all"):
        print("\n" + "="*60)
        print(" STAGE 2 — Surrogate training")
        print("="*60)
        from train import train
        train(
            data_csv=data_csv,
            out_dir=ckpt_dir,
            cfg_base=cfg,
            train_cfg=train_cfg,
            resume=args.resume,
        )

    # ─────────────────────────────────────────────────────────
    # Stage 3: evaluate
    # ─────────────────────────────────────────────────────────
    if args.mode in ("evaluate", "all"):
        print("\n" + "="*60)
        print(" STAGE 3 — Test set evaluation")
        print("="*60)
        from evaluate import evaluate
        evaluate(
            data_csv=data_csv,
            ckpt_path=ckpt_path,
            norms_path=norms_path,
            out_dir=eval_dir,
            cfg_base=cfg,
            train_cfg=train_cfg,
            n_plot=10,
            split="test",
        )

    # ─────────────────────────────────────────────────────────
    # Stage 4: optimise
    # ─────────────────────────────────────────────────────────
    if args.mode in ("optimise", "all"):
        print("\n" + "="*60)
        print(" STAGE 4 — Bayesian optimisation")
        print("="*60)
        from optimise import optimise
        result = optimise(
            ckpt_path=ckpt_path,
            norms_path=norms_path,
            out_dir=opt_dir,
            cfg_base=cfg,
            train_cfg=train_cfg,
            bayes_cfg=bayes_cfg,
            verify_ode=args.verify_ode,
        )

    print("\n" + "="*60)
    print(" Pipeline complete.")
    print("="*60)


if __name__ == "__main__":
    main()
