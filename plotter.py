"""
plotter.py  (generic)
=====================
All plots auto-generated after every run.
Adapts to any parameter set and any number of Pareto solutions.

Public API
----------
generate_all_plots(result, model, t_eval, out_dir) -> List[str]
generate_time_history(params, model, signals, t_eval, label, out_dir) -> str
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List

from user_code_runner import UserModel
from solver import run_one_case, compute_cabin_rms


def _savefig(fig, path: str, dpi: int = 150) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Pareto plots (NSGA-II)
# ---------------------------------------------------------------------------
def plot_pareto_2d(df: pd.DataFrame, out_dir: str) -> List[str]:
    pairs  = [("rms_z","rms_x"), ("rms_z","rms_y"), ("rms_x","rms_y")]
    labels = {"rms_z":"RMS_z vertical [m/s²]",
               "rms_x":"RMS_x longitudinal [m/s²]",
               "rms_y":"RMS_y lateral [m/s²]"}
    paths  = []
    for a, b in pairs:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df[a], df[b], s=60, c="steelblue",
                   edgecolors="k", linewidths=0.5, zorder=5)
        for _, row in df.iterrows():
            ax.annotate(row["label"], (row[a], row[b]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(labels[a]); ax.set_ylabel(labels[b])
        ax.set_title(f"Pareto — {a} vs {b}"); ax.grid(True, alpha=0.35)
        plt.tight_layout()
        paths.append(_savefig(fig, os.path.join(out_dir, "plots", f"pareto_{a}_{b}.png")))
    return paths


def plot_pareto_3d(df: pd.DataFrame, out_dir: str) -> str:
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(df["rms_z"], df["rms_x"], df["rms_y"],
                     c=df["rms_total"], cmap="plasma", s=60,
                     edgecolors="k", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="RMS_total", shrink=0.6)
    for _, row in df.iterrows():
        ax.text(row["rms_z"], row["rms_x"], row["rms_y"], row["label"], fontsize=7)
    ax.set_xlabel("RMS_z"); ax.set_ylabel("RMS_x"); ax.set_zlabel("RMS_y")
    ax.set_title("3-D Pareto Front")
    plt.tight_layout()
    return _savefig(fig, os.path.join(out_dir, "plots", "pareto_3d.png"))


def plot_hypervolume(hv_history: List[float], out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(hv_history)+1), hv_history,
            marker="o", linewidth=2, color="mediumslateblue")
    ax.set_xlabel("Generation"); ax.set_ylabel("Hypervolume")
    ax.set_title("NSGA-II Hypervolume Convergence"); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    return _savefig(fig, os.path.join(out_dir, "plots", "convergence_hv.png"))


def plot_generation_scatter(eval_log: List[Dict], df_pareto: pd.DataFrame,
                             out_dir: str) -> str:
    valid = [e for e in eval_log if e.get("rms_z", 99) < 90]
    if not valid:
        return ""
    zv = np.array([e["rms_z"] for e in valid])
    xv = np.array([e["rms_x"] for e in valid])
    gv = np.array([e.get("gen", 0) for e in valid])
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(zv, xv, c=gv, cmap="viridis", s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Generation")
    ax.scatter(df_pareto["rms_z"], df_pareto["rms_x"],
               s=120, marker="*", color="red", zorder=5, label="Pareto")
    ax.set_xlabel("RMS_z"); ax.set_ylabel("RMS_x")
    ax.set_title("Population Evolution"); ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    return _savefig(fig, os.path.join(out_dir, "plots", "generation_scatter.png"))


# ---------------------------------------------------------------------------
# Shared plots
# ---------------------------------------------------------------------------
def plot_parallel_coordinates(df: pd.DataFrame, bounds: Dict, param_keys: List[str],
                               out_dir: str) -> str:
    keys = [k for k in param_keys if k in df.columns]
    if not keys:
        return ""
    norm = df.copy()
    for k in keys:
        lo, hi = float(bounds[k][0]), float(bounds[k][1])
        norm[k] = (df[k] - lo) / max(hi - lo, 1e-12)

    n    = len(df)
    cmap = cm.get_cmap("plasma", max(n, 1))
    xs   = list(range(len(keys)))
    fig, ax = plt.subplots(figsize=(max(10, len(keys)*1.5), 5))
    for i in range(n):
        vals = [norm[k].iloc[i] for k in keys]
        ax.plot(xs, vals, color=cmap(i), linewidth=1.4, alpha=0.85,
                label=df["label"].iloc[i])
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Normalised value in bounds [0–1]")
    ax.set_title("Parallel Coordinates — Pareto Parameter Sets")
    ax.axhline(0.05, color="red",    linestyle=":", linewidth=0.8)
    ax.axhline(0.95, color="orange", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return _savefig(fig, os.path.join(out_dir, "plots", "param_parallel.png"))


def plot_rms_bars(df: pd.DataFrame, out_dir: str) -> str:
    labels = df["label"].tolist()
    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.5), 5))
    ax.bar(x-w, df["rms_z"], w, label="RMS_z vertical",      color="steelblue",  edgecolor="k")
    ax.bar(x,   df["rms_x"], w, label="RMS_x longitudinal",  color="darkorange", edgecolor="k")
    ax.bar(x+w, df["rms_y"], w, label="RMS_y lateral",       color="seagreen",   edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("RMS acceleration [m/s²]")
    ax.set_title("Per-Axis RMS — Pareto Solutions")
    ax.legend(); ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    return _savefig(fig, os.path.join(out_dir, "plots", "pareto_rms_bars.png"))


def plot_bayesian_convergence(convergence: List[float], out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(convergence)+1), convergence,
            marker=".", linewidth=1.8, color="coral")
    ax.set_xlabel("Evaluation #"); ax.set_ylabel("Best RMS_total [m/s²]")
    ax.set_title("Bayesian Optimisation Convergence"); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    return _savefig(fig, os.path.join(out_dir, "plots", "convergence_bayesian.png"))


# ---------------------------------------------------------------------------
# Seat acceleration time-history
# ---------------------------------------------------------------------------
def generate_time_history(params: Dict, model: UserModel, signals: Dict,
                           t_eval: np.ndarray, label: str, out_dir: str) -> str:
    try:
        df  = run_one_case(params, model, signals, t_eval)
        rms = compute_cabin_rms(df, model)
        t   = df["t"].values
        az  = df["cabin_az"].values
        ax_ = df["cabin_ax"].values
        ay  = df["cabin_ay"].values

        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for axi, sig, col, yl in zip(
            axes, [az, ax_, ay],
            ["steelblue", "darkorange", "seagreen"],
            ["z̈ seat [m/s²]", "ẍ seat [m/s²]", "ÿ seat [m/s²]"],
        ):
            axi.plot(t, sig, linewidth=0.6, color=col)
            axi.set_ylabel(yl); axi.grid(True, alpha=0.35)
        axes[2].set_xlabel("Time [s]")
        fig.suptitle(
            f"{label}  |  RMS: z={rms['rms_z']:.4f}  "
            f"x={rms['rms_x']:.4f}  y={rms['rms_y']:.4f}  "
            f"total={rms['rms_total']:.4f} m/s²",
            fontsize=10,
        )
        plt.tight_layout()
        safe = label.replace(" ", "_").replace("/", "-")
        return _savefig(fig, os.path.join(out_dir, "plots", f"seat_accel_{safe}.png"))
    except Exception as e:
        print(f"[WARN] time-history failed for {label}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Master call
# ---------------------------------------------------------------------------
def generate_all_plots(result, model: UserModel, signals: Dict,
                        t_eval: np.ndarray, out_dir: str) -> List[str]:
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    paths: List[str] = []
    df         = result.df_pareto
    param_keys = result.param_keys
    bounds     = model.bounds

    paths.append(plot_rms_bars(df, out_dir))
    paths.append(plot_parallel_coordinates(df, bounds, param_keys, out_dir))

    if result.mode == "nsga2":
        paths += plot_pareto_2d(df, out_dir)
        paths.append(plot_pareto_3d(df, out_dir))
        if result.hv_history:
            paths.append(plot_hypervolume(result.hv_history, out_dir))
        if result.eval_log:
            paths.append(plot_generation_scatter(result.eval_log, df, out_dir))
    elif result.mode == "bayesian":
        if result.convergence:
            paths.append(plot_bayesian_convergence(result.convergence, out_dir))

    # Time histories for key solutions
    done = set()
    key_labels = {"Best_total", "Best_vertical", "Best_longitudinal", "Best_lateral"}
    for _, row in df.iterrows():
        lbl = row["label"]
        if lbl in key_labels and lbl not in done:
            params = {k: row[k] for k in param_keys if k in row}
            p = generate_time_history(params, model, signals, t_eval, lbl, out_dir)
            if p:
                paths.append(p)
            done.add(lbl)

    return [p for p in paths if p]
