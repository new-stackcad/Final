"""
config.py  -  all physical constants, parameter bounds, and global settings.
Edit this file to change vehicle parameters, bounds, or training hyperparameters.
"""

from __future__ import annotations
from typing import Dict, Tuple

# ─────────────────────────────────────────────────────────────
# Simulation constants
# ─────────────────────────────────────────────────────────────
DT       = 0.001          # timestep  [s]  -> 1000 Hz
FS       = 1000           # sample rate [Hz]
T_END    = 466.945        # simulation end  [s]
T_IGNORE = 0.5            # skip first 0.5 s when computing RMS (transient)

# State indices
STATE_NAMES = ["z_c", "th_c", "ph_c", "z_s", "th_s", "ph_s"]
ZC, THC, PHC, ZS, THS, PHS = range(6)

# Optimisation parameter names (must match BASE_CFG keys)
PARAM_NAMES = [
    "K_f", "C_f", "K_2", "K_3",
    "cs_minus", "asym_ratio", "gamma_c", "gamma_r",
]

# ─────────────────────────────────────────────────────────────
# Vehicle parameters
# ─────────────────────────────────────────────────────────────
BASE_CFG: Dict = {
    "axlefront_left_csv":  "data/front_left.csv",
    "axlefront_right_csv": "data/front_right.csv",
    "axlerear1_left_csv":  "data/rear1_left.csv",
    "axlerear1_right_csv": "data/rear1_right.csv",
    "axlerear2_left_csv":  "data/rear2_left.csv",
    "axlerear2_right_csv": "data/rear2_right.csv",

    "sim_duration_s": T_END,

    "s1": 0.6277, "s2": 0.6305,
    "WT1": 0.814, "WT2": 1.047, "WT3": 1.047,
    "a": 0.9, "b": 1.080,
    "m_s": 22485.0, "I_syy": 103787.0, "I_sxx": 8598.0, "I_sxy": 763.0,
    "M_1f": 600.0, "M_2": 1075.0, "M_3": 840.0,
    "I_xx1": 650.0, "I_xx2": 1200.0, "I_xx3": 1100.0,
    "lf": 5.05, "L12": 0.54, "L23": 1.96,
    "l_cf": 6.458, "l_cr": 4.5, "l_cfcg": 0.871, "l_crcg": 1.087,
    "m_c": 862.0, "I_xxc": 516.6, "I_yyc": 1045.0,
    "hs": 0.68, "g": 9.81, "hcp": 0.1,
    "h_seat": 0.1,    # same as hcp for ISO 2631 lever arm

    "L_DL2": 0.6211, "L_DR2": 0.6211,
    "L_DL3": 0.6251, "L_DR3": 0.6251,
    "beta_L2": 0.1693, "beta_R2": 0.1693,
    "beta_L3": 0.17453, "beta_R3": 0.17453,
    "S_tf2": 1.043, "S_tf3": 1.043, "S_f": 0.814,

    "C_cfl": 5035.0, "C_cfr": 5035.0, "C_crl": 3400.0, "C_crr": 3400.0,
    "K_cfl": 49050.0, "K_cfr": 49050.0, "K_crl": 24525.0, "K_crr": 24525.0,

    "K_f": 474257, "C_f": 15000,
    "K_2": 1077620, "C_2": 2000,
    "K_3": 1077620, "C_3": 2000,

    "cs_minus":   0.3,
    "asym_ratio": 3.0,
    "gamma_c":    0.12,
    "gamma_r":    0.09,

    "baum_omega": 10.0, "baum_zeta": 1.0,
}

# ─────────────────────────────────────────────────────────────
# Optimisation bounds  (matching your working ODE BayesOpt code)
# ─────────────────────────────────────────────────────────────
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "K_f":        (0.879 * 474257,  1.126 * 474257),
    "C_f":        (0.44  * 15000,   1.40  * 15000),
    "K_2":        (0.892 * 1077620, 1.116 * 1077620),
    "K_3":        (0.892 * 1077620, 1.116 * 1077620),
    "cs_minus":   (0.20, 0.40),
    "asym_ratio": (2.30, 4.00),
    "gamma_c":    (0.08, 0.16),
    "gamma_r":    (0.08, 0.10),
}

# ─────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────
TRAIN_CFG: Dict = {
    "epochs":       150,
    "lr":           3e-4,
    "weight_decay": 1e-5,
    "patience":     25,
    "batch_size":   4,
    "split_frac":   0.80,
    "downsample":   4,           # 1000 Hz / 4 = 250 Hz stored

    "lambda_state": 0.5,
    "lambda_accel": 5.0,
    "lambda_rms":   20.0,
    "lambda_phys":  0.01,

    # TCN + LSTM dims
    "road_dim":      64,    # increased from 32 for TCN (more expressive)
    "param_dim":     128,
    "lstm_hidden":   256,
    "lstm_layers":   2,
    "dropout":       0.10,
}

# ─────────────────────────────────────────────────────────────
# Bayesian optimisation settings
# ─────────────────────────────────────────────────────────────
BAYES_CFG: Dict = {
    "n_init": 12,
    "n_iter": 60,
    "seed":   42,
}
