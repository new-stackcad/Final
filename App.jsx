import React, { useState, useEffect, useRef, useCallback } from "react";
import Editor from "@monaco-editor/react";
import axios from "axios";

const API = "/api";

// ═══════════════════════════════════════════════════════════════════════════
// DEFAULT CODE TEMPLATES
// ═══════════════════════════════════════════════════════════════════════════

const DEFAULT_ODE_CODE = `# ─────────────────────────────────────────────────────────────────────────
# ODE Panel — define your system here
# Rules:
#   1. Define CONFIG = {...}   all vehicle/system parameters
#   2. Define BOUNDS = {...}   optimisation search ranges [lo, hi]
#   3. Define OUTPUTS = {...}  maps cabin acceleration indices (see below)
#   4. Define ode_rhs(t, x, params, inputs)  your equations of motion
#   5. Optionally define STATE_NAMES = [...]  for plot labels
#
# inputs dict keys (from road CSV signals):
#   z1f, ph_f, z2, ph2, z3, ph3      axle displacements / roll angles
#   dz1f, dph_f, dz2, dph2, dz3, dph3  their rates
# ─────────────────────────────────────────────────────────────────────────

import numpy as np
from dataclasses import dataclass

# ── State vector layout for the 6-DOF cabin + sprung mass system ──────────
# x = [z_c, th_c, ph_c, z_s, th_s, ph_s,   <- positions  (indices 0-5)
#      dz_c,dth_c,dph_c,dz_s,dth_s,dph_s]  <- velocities (indices 6-11)
#
STATE_NAMES = ["z_c","th_c","ph_c","z_s","th_s","ph_s",
               "dz_c","dth_c","dph_c","dz_s","dth_s","dph_s"]

CONFIG = {
    # ── Sprung mass ───────────────────────────────────────────────────────
    "m_s":   22485.0,   "I_syy": 103787.0,
    "I_sxx": 8598.0,    "I_sxy": 763.0,
    "hs":    0.68,

    # ── Cabin ─────────────────────────────────────────────────────────────
    "m_c":   862.0,     "I_xxc": 516.6,    "I_yyc": 1045.0,
    "hcp":   0.1,

    # ── Geometry ──────────────────────────────────────────────────────────
    "lf":    5.05,      "L12":   0.54,     "L23":   1.96,
    "l_cf":  6.458,     "l_cr":  4.5,
    "l_cfcg":0.871,     "l_crcg":1.087,
    "a":     0.9,       "b":     1.080,
    "WT1":   0.814,     "WT2":   1.047,    "WT3":   1.047,
    "S_f":   0.814,     "S_tf2": 1.043,    "S_tf3": 1.043,
    "s1":    0.6277,    "s2":    0.6305,

    # ── Leaf spring geometry ───────────────────────────────────────────────
    "beta_L2":0.1693,   "beta_R2":0.1693,
    "beta_L3":0.17453,  "beta_R3":0.17453,
    "L_DL2":  0.6211,   "L_DR2":  0.6211,
    "L_DL3":  0.6251,   "L_DR3":  0.6251,

    # ── Suspension (optimised parameters — baseline values) ───────────────
    "K_f":        474257,   "C_f":  15000,
    "K_2":        1077620,  "K_3":  1077620,
    "C_2":        2000,     "C_3":  2000,
    "cs_minus":   0.3,      "asym_ratio": 3.0,
    "gamma_c":    0.12,     "gamma_r":    0.09,

    # ── Cabin mounts ───────────────────────────────────────────────────────
    "K_cfl":49050.0, "K_cfr":49050.0, "K_crl":24525.0, "K_crr":24525.0,
    "C_cfl":5035.0,  "C_cfr":5035.0,  "C_crl":3400.0,  "C_crr":3400.0,

    # ── Physics / solver ───────────────────────────────────────────────────
    "g": 9.81,
    "baum_omega": 10.0,   "baum_zeta": 1.0,
}

# ── OUTPUTS: tell the framework which state indices carry cabin accelerations
# These come from the ACCELERATION part of x (second half of state vector).
# For a 12-state system: positions are 0-5, velocities 6-11.
# The RHS returns [velocities, accelerations] — accelerations are at indices 6-11.
# cabin_z_accel_idx=6 means xdot[6] = z̈_c (vertical cabin accel)
OUTPUTS = {
    "cabin_z_accel_idx":  6,    # index in xdot (not x) for z̈_c
    "cabin_th_accel_idx": 7,    # index in xdot for θ̈_c (pitch → longitudinal)
    "cabin_ph_accel_idx": 8,    # index in xdot for φ̈_c (roll  → lateral)
    "hcp": 0.1,                 # seat height offset [m]
}

BOUNDS = {
    "K_f":        [0.879 * 474257,  1.126 * 474257],
    "C_f":        [0.44  * 15000,   1.4   * 15000],
    "K_2":        [0.892 * 1077620, 1.116 * 1077620],
    "K_3":        [0.892 * 1077620, 1.116 * 1077620],
    "cs_minus":   [0.20,  0.40],
    "asym_ratio": [2.30,  4.00],
    "gamma_c":    [0.08,  0.16],
    "gamma_r":    [0.08,  0.10],
}

# ── Two-stage asymmetric damper ────────────────────────────────────────────
def damper_force(v_rel, cs_minus, asym_ratio, gamma_c, gamma_r,
                 alpha_c=-0.05, alpha_r=0.13):
    c_plus = asym_ratio * cs_minus
    if v_rel < 0.0:
        if v_rel >= alpha_c:
            return cs_minus * v_rel
        return cs_minus * (alpha_c + gamma_c * (v_rel - alpha_c))
    else:
        if v_rel <= alpha_r:
            return c_plus * v_rel
        return c_plus * (alpha_r + gamma_r * (v_rel - alpha_r))

# ── ODE RHS ────────────────────────────────────────────────────────────────
def ode_rhs(t, x, params, inputs):
    """
    6-DOF cabin + sprung mass ride model.
    x = [z_c, th_c, ph_c, z_s, th_s, ph_s, dz_c, dth_c, dph_c, dz_s, dth_s, dph_s]
    Returns xdot of same length.
    """
    p = params

    # ── Unpack positions and velocities ───────────────────────────────────
    z_c, th_c, ph_c, z_s, th_s, ph_s         = x[0], x[1], x[2], x[3], x[4], x[5]
    dz_c,dth_c,dph_c,dz_s,dth_s,dph_s        = x[6], x[7], x[8], x[9], x[10],x[11]

    # ── Road inputs ────────────────────────────────────────────────────────
    z1f  = inputs.get("z1f",  0.0);  ph_f  = inputs.get("ph_f",  0.0)
    z2   = inputs.get("z2",   0.0);  ph2   = inputs.get("ph2",   0.0)
    z3   = inputs.get("z3",   0.0);  ph3   = inputs.get("ph3",   0.0)
    dz2  = inputs.get("dz2",  0.0);  dph2  = inputs.get("dph2",  0.0)
    dz3  = inputs.get("dz3",  0.0);  dph3  = inputs.get("dph3",  0.0)
    dph_f= inputs.get("dph_f",0.0)

    # ── Short aliases ──────────────────────────────────────────────────────
    K_f = p["K_f"]; C_f = p["C_f"]
    K_2 = p["K_2"]; C_2 = p["C_2"]
    K_3 = p["K_3"]; C_3 = p["C_3"]
    Kcfl=p["K_cfl"]; Kcfr=p["K_cfr"]; Kcrl=p["K_crl"]; Kcrr=p["K_crr"]
    Ccfl=p["C_cfl"]; Ccfr=p["C_cfr"]; Ccrl=p["C_crl"]; Ccrr=p["C_crr"]
    m_c=p["m_c"]; I_xxc=p["I_xxc"]; I_yyc=p["I_yyc"]
    m_s=p["m_s"]; I_sxx=p["I_sxx"]; I_syy=p["I_syy"]; I_sxy=p["I_sxy"]
    a=p["a"]; b=p["b"]; hs=p["hs"]; hcp=p["hcp"]; g=p["g"]
    l_cfcg=p["l_cfcg"]; l_crcg=p["l_crcg"]
    l_cf=p["l_cf"]; l_cr=p["l_cr"]; lf=p["lf"]
    L12=p["L12"]; L23=p["L23"]
    l2=L12; l3=L12+L23
    S1=p["S_f"]; S2=p["S_tf2"]; S3=p["S_tf3"]
    bL2=p["beta_L2"]; bR2=p["beta_R2"]; bL3=p["beta_L3"]; bR3=p["beta_R3"]
    L_DL2=p["L_DL2"]; L_DR2=p["L_DR2"]; L_DL3=p["L_DL3"]; L_DR3=p["L_DR3"]

    # ── Damper force ───────────────────────────────────────────────────────
    v_f  = dz_s - lf*dth_s - inputs.get("dz1f", 0.0)
    F_df = C_f * damper_force(v_f, p["cs_minus"], p["asym_ratio"],
                               p["gamma_c"], p["gamma_r"])

    phi_NRS2 = (bL2*L_DL2 - bR2*L_DR2) / max(S2, 1e-6)
    phi_NRS3 = (bL3*L_DL3 - bR3*L_DR3) / max(S3, 1e-6)

    # ── Cabin (z_c, th_c, ph_c) ───────────────────────────────────────────
    Kc = Kcfl+Kcfr+Kcrl+Kcrr; Cc = Ccfl+Ccfr+Ccrl+Ccrr
    dz_rel   = dz_c - dz_s;   z_rel = z_c - z_s
    dth_rel  = dth_c - dth_s;
    dph_rel  = dph_c - dph_s; ph_rel = ph_c - ph_s

    Fz_c  = -(Cc*dz_rel + Kc*z_rel
               + (Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dth_c
               + (-Ccfl*l_cf-Ccfr*l_cf-Ccrl*l_cr-Ccrr*l_cr)*dth_s
               + (-Ccfl*b+Ccfr*a-Ccrl*b+Ccrr*a)*dph_c
               + (Ccfl*b-Ccfr*a+Ccrl*b-Ccrr*a)*dph_s
               + (Kcfl*l_cfcg+Kcfr*l_cfcg-Kcrl*l_crcg-Kcrr*l_crcg)*th_c
               + (-Kcfl*l_cf-Kcfr*l_cf-Kcrl*l_cr-Kcrr*l_cr)*th_s
               + (-Kcfl*b+Kcfr*a-Kcrl*b+Kcrr*a)*ph_c
               + (Kcfl*b-Kcfr*a+Kcrl*b-Kcrr*a)*ph_s)

    Fth_c = -((-Ccfl*l_cfcg-Ccfr*l_cfcg+Ccrl*l_crcg+Ccrr*l_crcg)*dz_c
              + (-Ccfl*l_cfcg-Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dz_s
              + (-Ccfl*l_cfcg**2-Ccfr*l_cfcg**2-Ccrl*l_crcg**2-Ccrr*l_crcg**2)*dth_c
              + (Ccfl*l_cfcg*l_cf+Ccfr*l_cfcg*l_cf-Ccrl*l_crcg*l_cr-Ccrr*l_crcg*l_cr)*dth_s
              + (-Ccfl*l_cfcg*b+Ccfr*l_cfcg*a-Ccrl*l_crcg*b+Ccrr*l_crcg*a)*dph_c
              + (Ccfl*l_cfcg*b-Ccfr*l_cfcg*a+Ccrl*l_crcg*b-Ccrr*l_crcg*a)*dph_s
              + (-Kcfl*l_cfcg**2-Kcfr*l_cfcg**2-Kcrl*l_crcg**2-Kcrr*l_crcg**2+m_c*g*hcp)*th_c
              + (Kcfl*l_cfcg*l_cf+Kcfr*l_cfcg*l_cf-Kcrl*l_crcg*l_cr-Kcrr*l_crcg*l_cr)*th_s
              + (-Kcfl*l_cfcg*b+Kcfr*l_cfcg*a-Kcrl*l_crcg*b+Kcrr*l_crcg*a)*ph_c
              + (Kcfl*l_cfcg*b-Kcfr*l_cfcg*a+Kcrl*l_crcg*b-Kcrr*l_crcg*a)*ph_s)

    Fph_c = -((-Ccfl*b+Ccfr*a-Ccrl*b+Ccrr*a)*dz_c
              + (Ccfl*b-Ccfr*a+Ccrl*b-Ccrr*a)*dz_s
              + (-Ccfl*l_cfcg*b-Ccfr*l_cfcg*a+Ccrl*l_crcg*b+Ccrr*l_crcg*a)*dth_c
              + (Ccfl*l_cfcg*b+Ccfr*l_cfcg*a-Ccrl*l_crcg*b-Kcrr*l_crcg*a)*dth_s
              + (-Ccfl*b**2+Ccfr*a**2-Ccrl*b**2+Ccrr*a**2)*dph_c
              + (Ccfl*b**2-Ccfr*a**2+Ccrl*b**2-Ccrr*a**2)*dph_s
              + (-Kcfl*l_cfcg*b-Kcfr*l_cfcg*a+Kcrl*l_crcg*b+Kcrr*l_crcg*a)*th_c
              + (Kcfl*l_cfcg*b+Kcfr*l_cfcg*a-Kcrl*l_crcg*b-Kcrr*l_crcg*a)*th_s
              + (-Kcfl*b**2+Kcfr*a**2-Kcrl*b**2+Kcrr*a**2)*ph_c
              + (Kcfl*b**2-Kcfr*a**2+Kcrl*b**2-Kcrr*a**2)*ph_s)

    ddz_c  = Fz_c  / m_c
    ddth_c = Fth_c / I_yyc
    ddph_c = Fph_c / I_xxc

    # ── Sprung mass (z_s, th_s, ph_s) — unconstrained (constraints in geom panel) ─
    Fz_s = (-(Ccfl+Ccfr+Ccrl+Ccrr)*dz_c
            -(-Ccfl*l_cfcg-Ccfr*l_cfcg+Ccrl*l_crcg+Ccrr*l_crcg)*dth_c
            -(-Ccfl-Ccfr-Ccrl-Ccrr)*dz_s
            -(Ccfl*l_cf+Ccfr*l_cf+Ccrl*l_cr+Ccrr*l_cr)*dth_s
            -(Kcfl+Kcfr+Kcrl+Kcrr)*z_c
            -(-Kcfl*l_cfcg-Kcfr*l_cfcg+Kcrl*l_crcg+Kcrr*l_crcg)*th_c
            -(-Kcfl-Kcfr-Kcrl-Kcrr)*z_s
            -(Kcfl*l_cf+Kcfr*l_cf+Kcrl*l_cr+Kcrr*l_cr)*th_s
            + K_f*(z_s - lf*th_s - z1f) + F_df
            + K_2*(z_s - z2 - bL2*L_DL2 - bR2*L_DR2 + l2*th_s) + C_2*(dz_s - dz2 + l2*dth_s)
            + K_3*(z_s - z3 - bL3*L_DL3 - bR3*L_DR3 + l3*th_s) + C_3*(dz_s - dz3 + l3*dth_s))

    Fth_s = (-(Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dz_c
             -(-Ccfl*l_cfcg**2-Ccfr*l_cfcg**2-Ccrl*l_crcg**2-Ccrr*l_crcg**2)*dth_c
             -(-Ccfl*l_cf-Ccfr*l_cf-Ccrl*l_cr-Ccrr*l_cr)*dz_s
             -(Ccfl*l_cfcg*l_cf+Ccfr*l_cfcg*l_cf-Ccrl*l_crcg*l_cr-Ccrr*l_crcg*l_cr)*dth_s
             -(Kcfl*l_cf+Kcfr*l_cf+Kcrl*l_cr+Kcrr*l_cr)*z_c
             -(-Kcfl*l_cfcg*l_cf-Kcfr*l_cfcg*l_cf+Kcrl*l_crcg*l_cr+Kcrr*l_crcg*l_cr)*th_c
             -(-Kcfl*l_cf-Kcfr*l_cf-Kcrl*l_cr-Kcrr*l_cr)*z_s
             -(Kcfl*l_cf**2+Kcfr*l_cf**2+Kcrl*l_cr**2+Kcrr*l_cr**2)*th_s
             - lf*(K_f*(z_s - lf*th_s - z1f) + F_df)
             + l2*(K_2*(z_s-z2-bL2*L_DL2-bR2*L_DR2+l2*th_s)+C_2*(dz_s-dz2+l2*dth_s))
             + l3*(K_3*(z_s-z3-bL3*L_DL3-bR3*L_DR3+l3*th_s)+C_3*(dz_s-dz3+l3*dth_s)))

    k_tf=0.5*K_f*S1**2; C_tf=0.5*C_f*S1**2
    K_r1=0.5*K_2*S2**2; C_r1=0.5*C_2*S2**2
    K_r2=0.5*K_3*S3**2; C_r2=0.5*C_3*S3**2
    Fph_s = -(m_s*g*hs*ph_s
              - k_tf*(ph_s-ph_f) - C_tf*(dph_s-dph_f)
              - K_r1*(ph_s-ph2-phi_NRS2) - C_r1*(dph_s-dph2)
              - K_r2*(ph_s-ph3-phi_NRS3) - C_r2*(dph_s-dph3))

    ddz_s  = Fz_s  / m_s
    # Cross-coupled inertia for sprung pitch/roll
    det = I_syy*I_sxx - I_sxy**2
    ddth_s = ( I_sxx*Fth_s - I_sxy*Fph_s) / det
    ddph_s = (-I_sxy*Fth_s + I_syy*Fph_s) / det

    return [dz_c, dth_c, dph_c, dz_s, dth_s, dph_s,
            ddz_c, ddth_c, ddph_c, ddz_s, ddth_s, ddph_s]
`;

const DEFAULT_GEOM_CODE = `# ─────────────────────────────────────────────────────────────────────────
# Geometry / Constraints Panel
# ─────────────────────────────────────────────────────────────────────────
# Define:  constraints(t, q, params, inputs) -> list of residuals
#
# q      = position sub-vector (first half of state x)
# params = CONFIG dict
# inputs = road signals at time t
#
# Return a LIST of scalar residuals — one per constraint.
# The solver enforces g(t,q) = 0 via Baumgarte stabilisation.
# If no constraints are needed, just:  return []
# ─────────────────────────────────────────────────────────────────────────

import numpy as np

def constraints(t, q, params, inputs):
    """
    6-DOF leaf spring geometric compatibility constraints.
    q = [z_c, th_c, ph_c, z_s, th_s, ph_s]
    """
    p   = params
    z_s = q[3]; th_s = q[4]; ph_s = q[5]

    z2  = inputs.get("z2",  0.0)
    ph2 = inputs.get("ph2", 0.0)
    z3  = inputs.get("z3",  0.0)
    ph3 = inputs.get("ph3", 0.0)

    l2 = p["L12"]
    l3 = p["L12"] + p["L23"]
    S2 = p["S_tf2"]; S3 = p["S_tf3"]

    g1 = (z_s + l2*th_s + S2*ph_s
          - p["s1"]*np.sin(p["beta_L2"] - th_s)
          - (z2 + 0.5*p["WT2"]*ph2))

    g2 = (z_s + l3*th_s + S3*ph_s
          - p["s2"]*np.sin(p["beta_L3"] - th_s)
          - (z3 + 0.5*p["WT3"]*ph3))

    return [g1, g2]
`;

// ═══════════════════════════════════════════════════════════════════════════
// CSV key definitions
// ═══════════════════════════════════════════════════════════════════════════
const CSV_KEYS = [
  { key:"fa_lh",  label:"Front Axle — Left"   },
  { key:"fa_rh",  label:"Front Axle — Right"  },
  { key:"ra1_lh", label:"Rear Axle 1 — Left"  },
  { key:"ra1_rh", label:"Rear Axle 1 — Right" },
  { key:"ra2_lh", label:"Rear Axle 2 — Left"  },
  { key:"ra2_rh", label:"Rear Axle 2 — Right" },
];

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════
function fmtTime(secs) {
  if (!secs || secs <= 0) return "—";
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function LogLine({ line }) {
  const cls = line.includes("[ERROR]") ? "log-err"
            : line.includes("[WARN]")  ? "log-warn"
            : line.includes("[GEN")    ? "log-info"
            : line.includes("Est.")    ? "log-info"
            : "";
  return <div className={cls}>{line}</div>;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main App
// ═══════════════════════════════════════════════════════════════════════════
export default function App() {
  // ── Tab state ────────────────────────────────────────────────────────────
  const [tab, setTab] = useState("ode");  // "ode" | "results" | "surrogate"

  // ── Editor sub-tab (inside ODE tab) ──────────────────────────────────────
  const [editorTab, setEditorTab] = useState("ode");  // "ode" | "geom"

  // ── Code states ──────────────────────────────────────────────────────────
  const [odeCode,  setOdeCode]  = useState(DEFAULT_ODE_CODE);
  const [geomCode, setGeomCode] = useState(DEFAULT_GEOM_CODE);
  const [geomEnabled, setGeomEnabled] = useState(false);
  const [codeErr, setCodeErr]   = useState(null);

  // ── CSV uploads ───────────────────────────────────────────────────────────
  const [csvPaths,  setCsvPaths]  = useState({});
  const [csvStatus, setCsvStatus] = useState({});

  // ── Optimiser settings ────────────────────────────────────────────────────
  const [mode,     setMode]     = useState("nsga2");
  const [popSize,  setPopSize]  = useState(10);
  const [nGen,     setNGen]     = useState(5);
  const [nCalls,   setNCalls]   = useState(50);
  const [nInitial, setNInitial] = useState(10);
  const [tEnd,     setTEnd]     = useState(466.945);
  const [tIgnore,  setTIgnore]  = useState(0.5);

  // ── Job state ─────────────────────────────────────────────────────────────
  const [jobId,     setJobId]     = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [logs,      setLogs]      = useState([]);
  const [running,   setRunning]   = useState(false);
  const [results,   setResults]   = useState(null);
  const [plots,     setPlots]     = useState([]);
  const [selPlot,   setSelPlot]   = useState(null);
  const [progress,  setProgress]  = useState(0);

  // ── Time estimate ─────────────────────────────────────────────────────────
  const [estTotal,  setEstTotal]  = useState(null);
  const [estRemain, setEstRemain] = useState(null);
  const [runStart,  setRunStart]  = useState(null);
  const logRef = useRef(null);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  // Remaining time ticker
  useEffect(() => {
    if (!running || !estTotal || !runStart) { setEstRemain(null); return; }
    const interval = setInterval(() => {
      const elapsed  = (Date.now() - runStart) / 1000;
      const remain   = Math.max(0, estTotal - elapsed);
      setEstRemain(remain);
    }, 1000);
    return () => clearInterval(interval);
  }, [running, estTotal, runStart]);

  // ── Basic code validation (just check functions/dicts are there) ──────────
  useEffect(() => {
    const hasConfig  = odeCode.includes("CONFIG");
    const hasBounds  = odeCode.includes("BOUNDS");
    const hasOutputs = odeCode.includes("OUTPUTS");
    const hasRhs     = odeCode.includes("def ode_rhs");
    if (!hasConfig)  { setCodeErr("CONFIG dict not found"); return; }
    if (!hasBounds)  { setCodeErr("BOUNDS dict not found"); return; }
    if (!hasOutputs) { setCodeErr("OUTPUTS dict not found"); return; }
    if (!hasRhs)     { setCodeErr("ode_rhs function not found"); return; }
    if (geomEnabled && !geomCode.includes("def constraints")) {
      setCodeErr("Geometry enabled but constraints() function not found");
      return;
    }
    setCodeErr(null);
  }, [odeCode, geomCode, geomEnabled]);

  // ── CSV upload ────────────────────────────────────────────────────────────
  const uploadCsv = useCallback(async (key, file) => {
    setCsvStatus(s => ({ ...s, [key]: "uploading" }));
    const fd = new FormData();
    fd.append("file", file); fd.append("key", key);
    try {
      const res = await axios.post(`${API}/upload-csv`, fd);
      setCsvPaths(p => ({ ...p, [key]: res.data.server_path }));
      setCsvStatus(s => ({ ...s, [key]: "done" }));
    } catch {
      setCsvStatus(s => ({ ...s, [key]: "error" }));
    }
  }, []);

  // ── Start run ─────────────────────────────────────────────────────────────
  const startRun = useCallback(async () => {
    if (codeErr) { alert(`Code error: ${codeErr}`); return; }
    const csvCount = Object.keys(csvPaths).length;
    if (csvCount < 6) {
      alert(`Upload all 6 axle CSV files (${csvCount}/6 done).`); return;
    }

    setLogs([]); setResults(null); setPlots([]);
    setRunning(true); setProgress(5);
    setEstTotal(null); setEstRemain(null);
    setRunStart(Date.now());

    const payload = {
      ode_code:  odeCode,
      geom_code: geomEnabled ? geomCode : "",
      mode, pop_size: +popSize, n_gen: +nGen,
      n_calls: +nCalls, n_initial: +nInitial,
      T_END: +tEnd, T_IGNORE: +tIgnore,
      csv_paths: csvPaths,
    };

    let jid;
    try {
      const res = await axios.post(`${API}/run`, payload);
      jid = res.data.job_id;
      setJobId(jid); setJobStatus("running");
    } catch (e) {
      alert("Failed to start: " + e.message);
      setRunning(false); return;
    }

    // SSE stream
    const evtSrc = new EventSource(`${API}/stream/${jid}`);
    let evalCount = 0;
    const totalEvals = mode === "nsga2"
      ? +popSize * (+nGen + 1) : +nCalls;

    evtSrc.onmessage = (e) => {
      if (e.data === "__DONE__") {
        evtSrc.close(); setProgress(100);
        fetchResults(jid); return;
      }
      setLogs(l => [...l, e.data]);

      // Extract time estimate from log
      const estMatch = e.data.match(/Est\. total time ≈ (.+)/);
      if (estMatch) {
        const parts = estMatch[1].match(/(\d+)m (\d+)s/);
        if (parts) setEstTotal(parseInt(parts[1])*60 + parseInt(parts[2]));
        else {
          const secs = parseFloat(estMatch[1]);
          if (!isNaN(secs)) setEstTotal(secs);
        }
      }
      if (e.data.includes("eval ")) {
        evalCount++;
        setProgress(Math.min(95, Math.round((evalCount/totalEvals)*90)+5));
      }
    };
    evtSrc.onerror = () => { evtSrc.close(); fetchResults(jid); };
  }, [odeCode, geomCode, geomEnabled, csvPaths, mode,
      popSize, nGen, nCalls, nInitial, tEnd, tIgnore, codeErr]);

  const fetchResults = async (jid) => {
    try {
      const [rRes, pRes] = await Promise.all([
        axios.get(`${API}/results/${jid}`),
        axios.get(`${API}/plots/${jid}`),
      ]);
      setResults(rRes.data.result);
      const pl = pRes.data.plots || [];
      setPlots(pl); if (pl.length) setSelPlot(pl[0]);
      setJobStatus(rRes.data.status);
    } catch {}
    finally { setRunning(false); setTab("results"); }
  };

  const totalEvals = mode === "nsga2" ? +popSize*(+nGen+1) : +nCalls;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100vh", overflow:"hidden" }}>

      {/* ── Header ── */}
      <header style={{
        background:"var(--surface)", borderBottom:"1px solid var(--border)",
        padding:"10px 24px", display:"flex", alignItems:"center", gap:14, flexShrink:0,
      }}>
        <span style={{ fontSize:"1.3rem" }}>🚛</span>
        <h1 style={{ fontSize:"1.15rem" }}>Ride Optimisation</h1>

        <div style={{ marginLeft:"auto", display:"flex", gap:10, alignItems:"center" }}>
          {running && estRemain != null && (
            <span className="badge badge-warn">
              ⏱ Est. remaining: {fmtTime(estRemain)}
            </span>
          )}
          {running && (
            <span className="badge badge-warn">⚙ Running…</span>
          )}
          {jobStatus === "done" && !running && (
            <span className="badge badge-ok">✓ Complete</span>
          )}
          {jobStatus === "failed" && !running && (
            <span className="badge badge-err">✗ Failed</span>
          )}
        </div>
      </header>

      {/* ── Main tabs ── */}
      <div style={{
        padding:"0 24px", background:"var(--surface)",
        borderBottom:"1px solid var(--border)", flexShrink:0,
      }}>
        <div className="tabs" style={{ marginBottom:0 }}>
          <div className={`tab ${tab==="ode"?"active":""}`}
               onClick={() => setTab("ode")}>ODE Setup</div>
          <div className={`tab ${tab==="results"?"active":""} ${!results?"disabled":""}`}
               onClick={() => results && setTab("results")}>
            Results {plots.length > 0 && `(${plots.length})`}
          </div>
          <div className={`tab ${tab==="surrogate"?"active":""}`}
               onClick={() => setTab("surrogate")}>
            Surrogate
            <span style={{ fontSize:10, color:"var(--text3)", marginLeft:4 }}>(disabled)</span>
          </div>
        </div>
      </div>

      {/* ── Body ── */}
      <div style={{
        flex:1, overflow:"hidden", display:"flex", flexDirection:"column",
        padding:"14px 24px", gap:12,
      }}>

        {/* ══ ODE TAB ══════════════════════════════════════════════════════ */}
        {tab === "ode" && (
          <div style={{
            flex:1, overflow:"hidden",
            display:"grid", gridTemplateColumns:"1fr 320px", gap:12,
          }}>
            {/* Left: editor area */}
            <div style={{ display:"flex", flexDirection:"column", gap:10,
                          overflow:"hidden", minHeight:0 }}>

              {/* Editor sub-tabs */}
              <div style={{
                display:"flex", alignItems:"center", gap:8,
                background:"var(--surface2)", borderRadius:"var(--radius)",
                padding:"6px 12px", flexShrink:0,
              }}>
                <SubTabBtn active={editorTab==="ode"}
                           onClick={() => setEditorTab("ode")}>
                  📄 ODE + Config
                </SubTabBtn>
                <div style={{ display:"flex", alignItems:"center", gap:6, marginLeft:"auto" }}>
                  <span style={{ fontSize:12, color:"var(--text2)" }}>
                    Geometry constraints
                  </span>
                  <Toggle checked={geomEnabled} onChange={setGeomEnabled} />
                </div>
                {geomEnabled && (
                  <SubTabBtn active={editorTab==="geom"}
                             onClick={() => setEditorTab("geom")}>
                    📐 Geometry
                  </SubTabBtn>
                )}
                <div style={{ marginLeft: geomEnabled ? 0 : "auto" }}>
                  {codeErr
                    ? <span className="badge badge-err">{codeErr}</span>
                    : <span className="badge badge-ok">✓ code valid</span>
                  }
                </div>
              </div>

              {/* Monaco editor */}
              <div className="card" style={{
                flex:1, padding:0, overflow:"hidden", minHeight:0,
              }}>
                <div style={{
                  background:"var(--surface2)", padding:"6px 14px",
                  display:"flex", alignItems:"center", gap:10,
                  borderBottom:"1px solid var(--border)",
                }}>
                  <span style={{ fontFamily:"var(--font-mono)", fontSize:12,
                                 color:"var(--text2)" }}>
                    {editorTab === "ode" ? "ode_config.py" : "geometry.py"}
                  </span>
                  {editorTab === "geom" && (
                    <span className="badge badge-warn" style={{ fontSize:10 }}>
                      Baumgarte stabilised
                    </span>
                  )}
                </div>
                <Editor
                  height="100%"
                  defaultLanguage="python"
                  theme="vs-dark"
                  value={editorTab === "ode" ? odeCode : geomCode}
                  onChange={v => {
                    if (editorTab === "ode") setOdeCode(v||"");
                    else setGeomCode(v||"");
                  }}
                  options={{
                    fontSize:13,
                    fontFamily:"JetBrains Mono, Fira Code, monospace",
                    minimap:{ enabled:false },
                    scrollBeyondLastLine:false,
                    lineNumbers:"on",
                    wordWrap:"on",
                    padding:{ top:10 },
                  }}
                />
              </div>

              {/* CSV uploads */}
              <div className="card" style={{ flexShrink:0 }}>
                <h3 style={{ marginBottom:8 }}>Axle Displacement CSV Files</h3>
                <div style={{
                  display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:8,
                }}>
                  {CSV_KEYS.map(({ key, label }) => (
                    <CsvDropzone key={key} csvKey={key} label={label}
                                 status={csvStatus[key]}
                                 onFile={f => uploadCsv(key, f)} />
                  ))}
                </div>
              </div>
            </div>

            {/* Right: settings + log */}
            <div style={{ display:"flex", flexDirection:"column", gap:10, overflow:"auto" }}>

              {/* Optimiser card */}
              <div className="card">
                <h3 style={{ marginBottom:10 }}>Optimiser</h3>

                <label style={{ marginBottom:6 }}>Mode</label>
                <div style={{ display:"flex", gap:6, marginBottom:12 }}>
                  <ModeBtn active={mode==="nsga2"} onClick={() => setMode("nsga2")}>
                    <div style={{ fontWeight:600, fontSize:12 }}>Multi-objective</div>
                    <div style={{ fontSize:10, color:"var(--text3)" }}>NSGA-II · Pareto front</div>
                  </ModeBtn>
                  <ModeBtn active={mode==="bayesian"} onClick={() => setMode("bayesian")}>
                    <div style={{ fontWeight:600, fontSize:12 }}>Single objective</div>
                    <div style={{ fontSize:10, color:"var(--text3)" }}>Bayesian · best RMS</div>
                  </ModeBtn>
                </div>

                {mode === "nsga2" ? (
                  <div className="grid-2">
                    <NF label="Population" value={popSize} onChange={setPopSize} min={4} />
                    <NF label="Generations" value={nGen} onChange={setNGen} min={1} />
                  </div>
                ) : (
                  <div className="grid-2">
                    <NF label="Total calls" value={nCalls} onChange={setNCalls} min={5} />
                    <NF label="Initial pts"  value={nInitial} onChange={setNInitial} min={3} />
                  </div>
                )}

                <div style={{ height:1, background:"var(--border)", margin:"10px 0" }} />
                <h3 style={{ marginBottom:8 }}>Simulation time</h3>
                <div className="grid-2">
                  <NF label="T_END [s]"    value={tEnd}   onChange={setTEnd}   step={0.001} />
                  <NF label="T_IGNORE [s]" value={tIgnore} onChange={setTIgnore} step={0.1} />
                </div>

                <div style={{
                  marginTop:10, padding:"8px 10px",
                  background:"var(--surface2)", borderRadius:"var(--radius)",
                  fontSize:12,
                }}>
                  <div style={{ color:"var(--text3)", marginBottom:2 }}>ODE evaluations</div>
                  <span className="text-accent text-mono" style={{ fontSize:15 }}>
                    {totalEvals}
                  </span>
                  {estTotal && (
                    <span style={{ color:"var(--text3)", marginLeft:8 }}>
                      ≈ {fmtTime(estTotal)} est.
                    </span>
                  )}
                </div>

                <button
                  className="btn btn-primary w-full"
                  style={{ marginTop:10 }}
                  disabled={running || !!codeErr || Object.keys(csvPaths).length < 6}
                  onClick={startRun}
                >
                  {running ? "⚙ Running…" : "▶ Run Optimisation"}
                </button>

                {Object.keys(csvPaths).length < 6 && !running && (
                  <div style={{ fontSize:11, color:"var(--warn)", marginTop:6, textAlign:"center" }}>
                    Upload all 6 CSV files to enable ({Object.keys(csvPaths).length}/6)
                  </div>
                )}
              </div>

              {/* Progress */}
              {(running || jobStatus) && (
                <div className="card">
                  <div style={{ display:"flex", justifyContent:"space-between",
                                marginBottom:5, fontSize:12 }}>
                    <span>Progress</span>
                    <span className="text-mono text-accent">{progress}%</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width:`${progress}%` }} />
                  </div>
                  {running && estRemain != null && (
                    <div style={{ fontSize:11, color:"var(--text3)", marginTop:5 }}>
                      Est. remaining: <span className="text-accent">{fmtTime(estRemain)}</span>
                    </div>
                  )}
                </div>
              )}

              {/* Log */}
              <div className="card" style={{ flex:1 }}>
                <div style={{ display:"flex", justifyContent:"space-between",
                              marginBottom:6 }}>
                  <h3>Live Log</h3>
                  <button className="btn btn-ghost btn-sm"
                          onClick={() => setLogs([])}>Clear</button>
                </div>
                <div className="log-console" ref={logRef}>
                  {logs.length === 0
                    ? <span className="text-muted">Waiting for run…</span>
                    : logs.map((l,i) => <LogLine key={i} line={l} />)
                  }
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══ RESULTS TAB ══════════════════════════════════════════════════ */}
        {tab === "results" && results && (
          <div style={{
            flex:1, overflow:"hidden",
            display:"grid", gridTemplateColumns:"1fr 1fr", gap:12,
          }}>
            {/* Left */}
            <div style={{ display:"flex", flexDirection:"column", gap:10, overflow:"auto" }}>
              <div className="card">
                <div style={{ display:"flex", justifyContent:"space-between",
                              alignItems:"center", marginBottom:8 }}>
                  <h3>
                    {results.mode==="nsga2" ? "Pareto Front" : "Best Solution"}
                    &nbsp;<span className="badge badge-info">{results.mode}</span>
                  </h3>
                  <div style={{ display:"flex", gap:6 }}>
                    <a href={`${API}/download/${jobId}`}
                       className="btn btn-ghost btn-sm">⬇ CSV</a>
                    <a href={`${API}/download-all/${jobId}`}
                       className="btn btn-ghost btn-sm">⬇ ZIP</a>
                  </div>
                </div>
                <div style={{ overflowX:"auto" }}>
                  <table className="data-table">
                    <thead><tr>
                      <th>Label</th><th>RMS_z</th>
                      <th>RMS_x</th><th>RMS_y</th><th>RMS_total</th>
                    </tr></thead>
                    <tbody>
                      {(results.pareto||[]).map((row,i) => (
                        <tr key={i}>
                          <td><span className="text-mono text-accent">{row.label||`P${i+1}`}</span></td>
                          <td className="text-mono">{(+row.rms_z).toFixed(4)}</td>
                          <td className="text-mono">{(+row.rms_x).toFixed(4)}</td>
                          <td className="text-mono">{(+row.rms_y).toFixed(4)}</td>
                          <td className="text-mono text-green">{(+row.rms_total).toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="card">
                <h3 style={{ marginBottom:8 }}>Run Summary</h3>
                <div className="grid-2" style={{ gap:6 }}>
                  <StatBox label="Total evals"     value={results.n_evals} />
                  <StatBox label="Wall time"        value={`${results.wall_seconds?.toFixed(1)}s`} />
                  <StatBox label="Pareto solutions" value={(results.pareto||[]).length} />
                  <StatBox label="Mode"             value={results.mode} />
                </div>
              </div>

              {results.pareto?.[0] && results.param_keys && (
                <div className="card">
                  <h3 style={{ marginBottom:8 }}>Best — Optimised Parameters</h3>
                  <table className="data-table">
                    <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
                    <tbody>
                      {results.param_keys.map(k => (
                        <tr key={k}>
                          <td className="text-mono" style={{ fontSize:12 }}>{k}</td>
                          <td className="text-mono text-accent">
                            {Number(results.pareto[0][k]).toExponential(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Right — plots */}
            <div style={{ display:"flex", flexDirection:"column", gap:10,
                          overflow:"hidden" }}>
              <div className="card" style={{ flex:1, display:"flex",
                                             flexDirection:"column", overflow:"hidden" }}>
                <h3 style={{ marginBottom:8 }}>Plots</h3>
                <div style={{ display:"flex", flexWrap:"wrap", gap:5, marginBottom:8 }}>
                  {plots.map(p => (
                    <button key={p}
                      className={`btn btn-sm ${selPlot===p?"btn-primary":"btn-ghost"}`}
                      onClick={() => setSelPlot(p)}
                      style={{ fontSize:10 }}>
                      {p.replace(".png","").replace(/_/g," ")}
                    </button>
                  ))}
                </div>
                {selPlot ? (
                  <div style={{ flex:1, overflow:"hidden",
                                borderRadius:"var(--radius)", background:"#111" }}>
                    <img src={`${API}/plot/${jobId}/${selPlot}`} alt={selPlot}
                         style={{ width:"100%", height:"100%", objectFit:"contain" }} />
                  </div>
                ) : (
                  <div style={{ flex:1, display:"flex", alignItems:"center",
                                justifyContent:"center" }}>
                    <span className="text-muted">No plots yet</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ══ SURROGATE TAB ════════════════════════════════════════════════ */}
        {tab === "surrogate" && (
          <div style={{ maxWidth:580, margin:"0 auto", width:"100%" }}>
            <div className="card" style={{ borderColor:"var(--warn)" }}>
              <div style={{ display:"flex", gap:12, alignItems:"flex-start" }}>
                <span style={{ fontSize:"2rem" }}>⚠</span>
                <div>
                  <h2 style={{ color:"var(--warn)", marginBottom:4 }}>
                    Surrogate — Disabled
                  </h2>
                  <p style={{ color:"var(--text2)", fontSize:13, lineHeight:1.6 }}>
                    ODE is the active solver. Surrogate support is reserved for future
                    development. Upload your model files here to pre-validate them.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Small components
// ═══════════════════════════════════════════════════════════════════════════
function SubTabBtn({ active, onClick, children }) {
  return (
    <button onClick={onClick} style={{
      padding:"4px 12px", borderRadius:"var(--radius)", border:"none", cursor:"pointer",
      background: active ? "var(--accent)" : "transparent",
      color: active ? "#fff" : "var(--text2)",
      fontSize:12, fontWeight:600, fontFamily:"var(--font-ui)",
      transition:"background 0.15s",
    }}>{children}</button>
  );
}

function Toggle({ checked, onChange }) {
  return (
    <div onClick={() => onChange(!checked)} style={{
      width:38, height:20, borderRadius:10, cursor:"pointer",
      background: checked ? "var(--accent)" : "var(--border)",
      position:"relative", transition:"background 0.2s", flexShrink:0,
    }}>
      <div style={{
        position:"absolute", top:3, left: checked ? 20 : 3,
        width:14, height:14, borderRadius:7,
        background:"#fff", transition:"left 0.2s",
      }} />
    </div>
  );
}

function CsvDropzone({ csvKey, label, status, onFile }) {
  const ref = useRef();
  return (
    <div className={`upload-zone ${status==="done"?"done":""}`}
         onDragOver={e => e.preventDefault()}
         onDrop={e => { e.preventDefault(); if(e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); }}
         onClick={() => ref.current.click()}>
      <input type="file" accept=".csv" ref={ref} style={{ display:"none" }}
             onChange={e => { if(e.target.files[0]) onFile(e.target.files[0]); }} />
      <div style={{ fontWeight:600, fontSize:11 }}>
        {status==="done"?"✓ ":status==="uploading"?"⟳ ":status==="error"?"✗ ":""}
        {label}
      </div>
      <div style={{ fontSize:10, marginTop:2 }}>
        {status==="done"?"Uploaded":status==="uploading"?"Uploading…":
         status==="error"?"Failed":"Click or drop CSV"}
      </div>
    </div>
  );
}

function ModeBtn({ active, onClick, children }) {
  return (
    <button className={`btn ${active?"btn-primary":"btn-ghost"}`}
            onClick={onClick}
            style={{ flex:1, flexDirection:"column", alignItems:"flex-start",
                     padding:"8px 10px", textAlign:"left", lineHeight:1.4 }}>
      {children}
    </button>
  );
}

function NF({ label, value, onChange, min, step }) {
  return (
    <div>
      <label>{label}</label>
      <input type="number" value={value} min={min} step={step||1}
             onChange={e => onChange(e.target.value)} />
    </div>
  );
}

function StatBox({ label, value }) {
  return (
    <div style={{ background:"var(--surface2)", borderRadius:"var(--radius)",
                  padding:"8px 12px" }}>
      <div style={{ fontSize:10, color:"var(--text3)", marginBottom:2 }}>{label}</div>
      <div style={{ fontFamily:"var(--font-mono)", fontSize:14,
                    color:"var(--accent)" }}>{value ?? "—"}</div>
    </div>
  );
}
