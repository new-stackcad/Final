"""
Multi-Objective Vehicle Dynamics Optimisation — NO pymoo
=========================================================
IDENTICAL to multiobjective_nsga2_pymoo.py in every way EXCEPT:
  • pymoo is NOT imported
  • NSGA-II / HV / NonDominatedSorting come from  nsga2_engine.py
    (pure NumPy — place both files in the same folder)

Objectives (all minimised):
    f1 = RMS_z  – vertical seat acceleration       [m/s²]
    f2 = RMS_x  – longitudinal seat accel (pitch)  [m/s²]
    f3 = RMS_y  – lateral seat accel (roll)         [m/s²]

Parameters optimised:
    K_f, C_f, K_2, K_3, cs_minus, asym_ratio, gamma_c, gamma_r

Budget:  POP_SIZE × N_GEN  total ODE evaluations
Default: 10 × 5 = 50  (same as pymoo version)

Dependencies (all standard):
    numpy  scipy  pandas  matplotlib
    + nsga2_engine.py  (in the same directory as this file)
"""

# ── standard imports ──────────────────────────────────────────────────────────
import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from dataclasses  import dataclass
from typing       import Dict, Callable, Tuple
from numpy.linalg import solve as lin_solve
from scipy.integrate import solve_ivp
from scipy.optimize  import least_squares

# ── local NSGA-II engine (no pymoo) ──────────────────────────────────────────
from nsga2_engine import NSGA2, HV, NonDominatedSorting

# ═══════════════════════════════════════════════════════════════════════════════
# BUDGET
# ═══════════════════════════════════════════════════════════════════════════════
POP_SIZE = 10
N_GEN    = 5

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
DT       = 0.001
T_IGNORE = 0.5
T_END    = 466.945
t_eval_full = np.arange(0.0, T_END + DT, DT)

RESULTS_DIR = "Res_Laden_MOBO_NSGA2_nopymoo"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

STATE_NAMES = ["z_c", "th_c", "ph_c", "z_s", "th_s", "ph_s"]
(ZC, THC, PHC, ZS, THS, PHS) = range(6)

HV_REF = np.array([5.0, 5.0, 5.0])

# ═══════════════════════════════════════════════════════════════════════════════
# CFG
# ═══════════════════════════════════════════════════════════════════════════════
CFG: Dict = {
    "axlefront_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_1_FA_LH.csv",
    "axlefront_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_2_FA_RH.csv",
    "axlerear1_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_3_RA1_LH.csv",
    "axlerear1_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_4_RA1_RH.csv",
    "axlerear2_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_5_RA2_LH.csv",
    "axlerear2_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_6_RA2_RH.csv",

    "s1": 0.6277, "s2": 0.6305,
    "WT1": 0.814,  "WT2": 1.047,  "WT3": 1.047,
    "m_c": 862.0,  "I_xxc": 516.6,"I_yyc": 1045.0,
    "M_1f": 600.0, "M_2": 1075.0, "M_3": 840.0,
    "I_xx1": 650.0,"I_xx2": 1200.0,"I_xx3": 1100.0,
    "S_tf2": 1.043,"S_tf3": 1.043,"S_f": 0.814,
    "C_cfl": 5035.0,"C_cfr": 5035.0,"C_crl": 3400.0,"C_crr": 3400.0,
    "K_cfl": 49050.0,"K_cfr": 49050.0,"K_crl": 24525.0,"K_crr": 24525.0,
    "C_2": 2000,"C_3": 2000,
    "L_DL2": 0.6211,"L_DR2": 0.6211,
    "L_DL3": 0.6251,"L_DR3": 0.6251,
    "beta_L2": 0.1693,"beta_R2": 0.1693,
    "beta_L3": 0.17453,"beta_R3": 0.17453,
    "a": 0.9,"b": 1.080,
    "l_cfcg": 0.871,"l_crcg": 1.087,"hcp": 0.1,
    "lf": 5.05,"L12": 0.54,"L23": 1.96,
    "l_cf": 6.458,"l_cr": 4.5,
    "m_s": 22485.0,"I_syy": 103787.0,"I_sxx": 8598.0,"I_sxy": 763.0,
    "hs": 0.68,
    "K_f": 474257,"C_f": 15000,"K_2": 1077620,"K_3": 1077620,
    "g": 9.81,
    "cs_minus": 0.3,"asym_ratio": 3.0,"gamma_c": 0.12,"gamma_r": 0.09,
    "baum_omega": 10.0,"baum_zeta": 1.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDS
# ═══════════════════════════════════════════════════════════════════════════════
PARAM_KEYS = ["K_f","C_f","K_2","K_3","cs_minus","asym_ratio","gamma_c","gamma_r"]
BOUNDS_RAW = {
    "K_f":       (0.879*CFG["K_f"],  1.126*CFG["K_f"]),
    "C_f":       (0.44 *CFG["C_f"],  1.4  *CFG["C_f"]),
    "K_2":       (0.892*CFG["K_2"],  1.116*CFG["K_2"]),
    "K_3":       (0.892*CFG["K_3"],  1.116*CFG["K_3"]),
    "cs_minus":  (0.2,  0.4),
    "asym_ratio":(2.3,  4.0),
    "gamma_c":   (0.08, 0.16),
    "gamma_r":   (0.08, 0.10),
}
XL = np.array([BOUNDS_RAW[k][0] for k in PARAM_KEYS])
XU = np.array([BOUNDS_RAW[k][1] for k in PARAM_KEYS])

_eval_log: list = []

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS  (identical to pymoo version — no changes)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TwoStageAsymmetricDamper:
    cs_minus: float; asym_ratio: float; gamma_c: float; gamma_r: float
    alpha_c: float = -0.05; alpha_r: float = 0.13

    def force(self, v_rel: float) -> float:
        c_plus = self.asym_ratio * self.cs_minus
        if v_rel < 0.0:
            return (self.cs_minus * v_rel if v_rel >= self.alpha_c
                    else self.cs_minus*(self.alpha_c+self.gamma_c*(v_rel-self.alpha_c)))
        else:
            return (c_plus * v_rel if v_rel <= self.alpha_r
                    else c_plus*(self.alpha_r+self.gamma_r*(v_rel-self.alpha_r)))


def load_track(csv_path):
    df   = pd.read_csv(csv_path, skiprows=2, header=None)
    t    = pd.to_numeric(df.iloc[:,0], errors="coerce").values
    z    = pd.to_numeric(df.iloc[:,1], errors="coerce").values
    mask = np.isfinite(t) & np.isfinite(z)
    return t[mask].astype(float), z[mask].astype(float)


def make_linear_interp(x, y):
    x = np.asarray(x); y = np.asarray(y)
    def f(xq):
        xq   = np.asarray(xq)
        xq_c = np.clip(xq, x[0], x[-1])
        idx  = np.clip(np.searchsorted(x, xq_c)-1, 0, len(x)-2)
        w    = (xq_c-x[idx])/np.maximum(x[idx+1]-x[idx],1e-12)
        return y[idx]*(1-w)+y[idx+1]*w
    return f


@dataclass
class RoadSignals:
    f1L: Callable; f1R: Callable
    f2L: Callable; f2R: Callable
    f3L: Callable; f3R: Callable

    def axle_inputs(self, t, cfg):
        zr1L,zr1R=self.f1L(t),self.f1R(t)
        zr2L,zr2R=self.f2L(t),self.f2R(t)
        zr3L,zr3R=self.f3L(t),self.f3R(t)
        z1f=0.5*(zr1L+zr1R); z2=0.5*(zr2L+zr2R); z3=0.5*(zr3L+zr3R)
        ph_f=(zr1L-zr1R)/cfg["WT1"]; ph2=(zr2L-zr2R)/cfg["WT2"]; ph3=(zr3L-zr3R)/cfg["WT3"]
        return float(z1f),float(ph_f),float(z2),float(ph2),float(z3),float(ph3)

    def axle_input_rates(self, t, cfg, dt=DT):
        p=self.axle_inputs(t+dt,cfg); m=self.axle_inputs(t-dt,cfg)
        return tuple((a-b)/(2.0*dt) for a,b in zip(p,m))


def build_road_signals(cfg):
    t1L,z1L=load_track(cfg["axlefront_left_csv"]); t1R,z1R=load_track(cfg["axlefront_right_csv"])
    t2L,z2L=load_track(cfg["axlerear1_left_csv"]); t2R,z2R=load_track(cfg["axlerear1_right_csv"])
    t3L,z3L=load_track(cfg["axlerear2_left_csv"]); t3R,z3R=load_track(cfg["axlerear2_right_csv"])
    return RoadSignals(
        make_linear_interp(t1L,z1L),make_linear_interp(t1R,z1R),
        make_linear_interp(t2L,z2L),make_linear_interp(t2R,z2R),
        make_linear_interp(t3L,z3L),make_linear_interp(t3R,z3R),
    )


def geom_constraints(q, t, cfg, road):
    z_s,th_s,ph_s=q[ZS],q[THS],q[PHS]
    _,_,z2,ph2,z3,ph3=road.axle_inputs(t,cfg)
    l2=cfg["L12"]; l3=cfg["L12"]+cfg["L23"]
    S2,S3=cfg["S_tf2"],cfg["S_tf3"]; sl2,sl3=cfg["s1"],cfg["s2"]
    bL2,bL3=cfg["beta_L2"],cfg["beta_L3"]
    g2=z_s+l2*th_s+S2*ph_s-sl2*np.sin(bL2-th_s)-(z2+0.5*cfg["WT2"]*ph2)
    g3=z_s+l3*th_s+S3*ph_s-sl3*np.sin(bL3-th_s)-(z3+0.5*cfg["WT3"]*ph3)
    gq=np.array([g2,g3],dtype=float)
    G=np.zeros((2,6),dtype=float)
    G[0,ZS]=1.; G[0,THS]=l2+sl2*np.cos(bL2-th_s); G[0,PHS]=S2
    G[1,ZS]=1.; G[1,THS]=l3+sl3*np.cos(bL3-th_s); G[1,PHS]=S3
    return gq, G


def build_M_R(q, v, t, cfg, road):
    z_c,th_c,ph_c,z_s,th_s,ph_s=q
    dz_c,dth_c,dph_c,dz_s,dth_s,dph_s=v
    z1f,ph_f,z2,ph2,z3,ph3=road.axle_inputs(t,cfg)
    dz1f,dph_f,dz2,dph2,dz3,dph3=road.axle_input_rates(t,cfg)
    phi_NRS2=(cfg["beta_L2"]*cfg["L_DL2"]-cfg["beta_R2"]*cfg["L_DR2"])/max(cfg["S_tf2"],1e-6)
    phi_NRS3=(cfg["beta_L3"]*cfg["L_DL3"]-cfg["beta_R3"]*cfg["L_DR3"])/max(cfg["S_tf3"],1e-6)
    m_c,I_xxc,I_yyc=cfg["m_c"],cfg["I_xxc"],cfg["I_yyc"]
    m_s,I_sxx,I_syy,I_sxy=cfg["m_s"],cfg["I_sxx"],cfg["I_syy"],cfg["I_sxy"]
    S1,S2,S3=cfg["S_f"],cfg["S_tf2"],cfg["S_tf3"]
    a,b=cfg["a"],cfg["b"]; hs,g=cfg["hs"],cfg["g"]
    l_cfcg,l_crcg,l_cf,l_cr=cfg["l_cfcg"],cfg["l_crcg"],cfg["l_cf"],cfg["l_cr"]
    lf,hcp=cfg["lf"],cfg["hcp"]; l2=cfg["L12"]; l3=cfg["L12"]+cfg["L23"]
    bL2,bR2=cfg["beta_L2"],cfg["beta_R2"]; bL3,bR3=cfg["beta_L3"],cfg["beta_R3"]
    L_DL2,L_DR2,L_DL3,L_DR3=cfg["L_DL2"],cfg["L_DR2"],cfg["L_DL3"],cfg["L_DR3"]
    Kcfl,Kcfr,Kcrl,Kcrr=cfg["K_cfl"],cfg["K_cfr"],cfg["K_crl"],cfg["K_crr"]
    Ccfl,Ccfr,Ccrl,Ccrr=cfg["C_cfl"],cfg["C_cfr"],cfg["C_crl"],cfg["C_crr"]
    K_f,C_f=cfg["K_f"],cfg["C_f"]; K_2,C_2=cfg["K_2"],cfg["C_2"]; K_3,C_3=cfg["K_3"],cfg["C_3"]

    damp=TwoStageAsymmetricDamper(cfg["cs_minus"],cfg["asym_ratio"],cfg["gamma_c"],cfg["gamma_r"])
    v_f=dz_s-lf*dth_s-dz1f; F_df=C_f*damp.force(v_f)
    Csum=Ccfl+Ccfr+Ccrl+Ccrr; Ksum=Kcfl+Kcfr+Kcrl+Kcrr

    M=np.zeros((6,6),dtype=float)
    M[ZC,ZC]=m_c; M[THC,THC]=I_yyc; M[PHC,PHC]=I_xxc; M[ZS,ZS]=m_s
    M[THS,THS]=I_syy; M[THS,PHS]=I_sxy; M[PHS,THS]=I_sxy; M[PHS,PHS]=I_sxx+m_s*hs**2

    R=np.zeros(6,dtype=float)
    R[ZC]=(Csum*(dz_c-dz_s)+Ksum*(z_c-z_s)
        -(Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dth_c
        -(-Ccfl*l_cf-Ccfr*l_cf-Ccrl*l_cr-Ccrr*l_cr)*dth_s
        -(-Ccfl*b+Ccfr*a-Ccrl*b+Ccrr*a)*dph_c-(Ccfl*b-Ccfr*a+Ccrl*b-Ccrr*a)*dph_s
        -(Kcfl*l_cfcg+Kcfr*l_cfcg-Kcrl*l_crcg-Kcrr*l_crcg)*th_c
        -(-Kcfl*l_cf-Kcfr*l_cf-Kcrl*l_cr-Kcrr*l_cr)*th_s
        -(-Kcfl*b+Kcfr*a-Kcrl*b+Kcrr*a)*ph_c-(Kcfl*b-Kcfr*a+Kcrl*b-Kcrr*a)*ph_s)
    R[THC]=(-(Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dz_c
        -(-Ccfl*l_cfcg-Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dz_s
        -(Kcfl*l_cfcg+Kcfr*l_cfcg-Kcrl*l_crcg-Kcrr*l_crcg)*z_c
        -(-Kcfl*l_cfcg-Kcfr*l_cfcg-Kcrl*l_crcg-Kcrr*l_crcg)*z_s
        -(-Ccfl*l_cfcg**2-Ccfr*l_cfcg**2-Ccrl*l_crcg**2-Ccrr*l_crcg**2)*dth_c
        -(Ccfl*l_cfcg*l_cf+Ccfr*l_cfcg*l_cf-Ccrl*l_crcg*l_cr-Ccrr*l_crcg*l_cr)*dth_s
        -(-Ccfl*l_cfcg*b+Ccfr*l_cfcg*a-Ccrl*l_crcg*b+Ccrr*l_crcg*a)*dph_c
        -(Ccfl*l_cfcg*b-Ccfr*l_cfcg*a+Ccrl*l_crcg*b-Ccrr*l_crcg*a)*dph_s
        -(-Kcfl*l_cfcg**2-Kcfr*l_cfcg**2-Kcrl*l_crcg**2-Kcrr*l_crcg**2+m_c*g*hcp)*th_c
        -(Kcfl*l_cfcg*l_cf+Kcfr*l_cfcg*l_cf-Kcrl*l_crcg*l_cr-Kcrr*l_crcg*l_cr)*th_s
        -(-Kcfl*l_cfcg*b+Kcfr*l_cfcg*a-Kcrl*l_crcg*b+Kcrr*l_crcg*a)*ph_c
        -(Kcfl*l_cfcg*b-Kcfr*l_cfcg*a+Kcrl*l_crcg*b-Kcrr*l_crcg*a)*ph_s)
    R[PHC]=(-(-Ccfl*b+Ccfr*a-Ccrl*b+Ccrr*a)*dz_c-(Ccfl*b-Ccfr*a+Ccrl*b-Ccrr*a)*dz_s
        -(-Kcfl*b+Kcfr*a-Kcrl*b+Kcrr*a)*z_c-(Kcfl*b-Kcfr*a+Kcrl*b-Kcrr*a)*z_s
        -(-Ccfl*l_cfcg*b-Ccfr*l_cfcg*a+Ccrl*l_crcg*b+Ccrr*l_crcg*a)*dth_c
        -(Ccfl*l_cfcg*b+Ccfr*l_cfcg*a-Ccrl*l_crcg*b-Ccrr*l_crcg*a)*dth_s
        -(-Ccfl*b**2+Ccfr*a**2-Ccrl*b**2+Ccrr*a**2)*dph_c
        -(Ccfl*b**2-Ccfr*a**2+Ccrl*b**2-Ccrr*a**2)*dph_s
        -(-Kcfl*l_cfcg*b-Kcfr*l_cfcg*a+Kcrl*l_crcg*b+Kcrr*l_crcg*a)*th_c
        -(Kcfl*l_cfcg*b+Kcfr*l_cfcg*a-Kcrl*l_crcg*b-Kcrr*l_crcg*a)*th_s
        -(-Kcfl*b**2+Kcfr*a**2-Kcrl*b**2+Kcrr*a**2)*ph_c
        -(Kcfl*b**2-Kcfr*a**2+Kcrl*b**2-Kcrr*a**2)*ph_s)
    R[ZS]=(-(Ccfl+Ccfr+Ccrl+Ccrr)*dz_c
        -(-Ccfl*l_cfcg-Ccfr*l_cfcg+Ccrl*l_crcg+Ccrr*l_crcg)*dth_c
        -(-Ccfl-Ccfr-Ccrl-Ccrr)*dz_s-(Ccfl*l_cf+Ccfr*l_cf+Ccrl*l_cr+Ccrr*l_cr)*dth_s
        -(Kcfl+Kcfr+Kcrl+Kcrr)*z_c
        -(-Kcfl*l_cfcg-Kcfr*l_cfcg+Kcrl*l_crcg+Kcrr*l_crcg)*th_c
        -(-Kcfl-Kcfr-Kcrl-Kcrr)*z_s-(Kcfl*l_cf+Kcfr*l_cf+Kcrl*l_cr+Kcrr*l_cr)*th_s
        +K_f*(z_s-lf*th_s-z1f)+F_df
        +K_2*(z_s-z2-bL2*L_DL2-bR2*L_DR2+l2*th_s)+C_2*(dz_s-dz2+l2*dth_s)
        +K_3*(z_s-z3-bL3*L_DL3-bR3*L_DR3+l3*th_s)+C_3*(dz_s-dz3+l3*dth_s))
    R[THS]=(-(Ccfl*l_cfcg+Ccfr*l_cfcg-Ccrl*l_crcg-Ccrr*l_crcg)*dz_c
        -(-Ccfl*l_cfcg**2-Ccfr*l_cfcg**2-Ccrl*l_crcg**2-Ccrr*l_crcg**2)*dth_c
        -(-Ccfl*l_cf-Ccfr*l_cf-Ccrl*l_cr-Ccrr*l_cr)*dz_s
        -(Ccfl*l_cfcg*l_cf+Ccfr*l_cfcg*l_cf-Ccrl*l_crcg*l_cr-Ccrr*l_crcg*l_cr)*dth_s
        -(Kcfl*l_cf+Kcfr*l_cf+Kcrl*l_cr+Kcrr*l_cr)*z_c
        -(-Kcfl*l_cfcg*l_cf-Kcfr*l_cfcg*l_cf+Kcrl*l_crcg*l_cr+Kcrr*l_crcg*l_cr)*th_c
        -(-Kcfl*l_cf-Kcfr*l_cf-Kcrl*l_cr-Kcrr*l_cr)*z_s
        -(Kcfl*l_cf**2+Kcfr*l_cf**2+Kcrl*l_cr**2+Kcrr*l_cr**2)*th_s
        -lf*(K_f*(z_s-lf*th_s-z1f)+F_df)
        +l2*(K_2*(z_s-z2-bL2*L_DL2-bR2*L_DR2+l2*th_s)+C_2*(dz_s-dz2+l2*dth_s))
        +l3*(K_3*(z_s-z3-bL3*L_DL3-bR3*L_DR3+l3*th_s)+C_3*(dz_s-dz3+l3*dth_s)))
    k_tf=0.5*K_f*S1**2; K_r1=0.5*K_2*S2**2; K_r2=0.5*K_3*S3**2
    C_tf=0.5*C_f*S1**2; C_r1=0.5*C_2*S2**2; C_r2=0.5*C_3*S3**2
    R[PHS]=-(m_s*g*hs*ph_s
        -k_tf*(ph_s-ph_f)-C_tf*(dph_s-dph_f)
        -K_r1*(ph_s-ph2-phi_NRS2)-C_r1*(dph_s-dph2)
        -K_r2*(ph_s-ph3-phi_NRS3)-C_r2*(dph_s-dph3))
    return M, R


def rhs_first_order(t, x, cfg, road):
    q,v=x[:6],x[6:]; M,R=build_M_R(q,v,t,cfg,road); gq,G=geom_constraints(q,t,cfg,road)
    w,z=cfg["baum_omega"],cfg["baum_zeta"]; gamma=w**2*gq+2*z*w*(G@v)
    nc=G.shape[0]; A=np.zeros((6+nc,6+nc)); b=np.zeros(6+nc)
    A[:6,:6]=M; A[:6,6:]=G.T; A[6:,:6]=G; b[:6]=-R; b[6:]=-gamma
    xdot=np.zeros_like(x); xdot[:6]=v; xdot[6:]=lin_solve(A,b)[:6]
    return xdot


def static_equilibrium_state(cfg, road):
    t0=0.0
    def F(y):
        q,lam=y[:6],y[6:]; M,R=build_M_R(q,np.zeros(6),t0,cfg,road)
        gq,G=geom_constraints(q,t0,cfg,road)
        return np.hstack([R+G.T@lam, 1e3*gq])
    lsq=least_squares(F,np.zeros(8),method="trf",loss="soft_l1",
                      xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=800)
    if lsq.success:
        q0=lsq.x[:6]
        print(f"    [EQ OK] ||g||={np.linalg.norm(geom_constraints(q0,t0,cfg,road)[0]):.2e}")
        return np.hstack([q0,np.zeros(6)])
    cfg_r={**cfg,**{k:cfg[k]*20 for k in ["C_2","C_3","C_cfl","C_cfr","C_crl","C_crr"]}}
    sol=solve_ivp(lambda t,x:rhs_first_order(t,x,cfg_r,road),(0.,3.),
                  np.zeros(12),method="Radau",rtol=1e-7,atol=1e-9)
    q0=sol.y[:6,-1]
    print(f"    [EQ fallback] ||g||={np.linalg.norm(geom_constraints(q0,t0,cfg,road)[0]):.2e}")
    return np.hstack([q0,np.zeros(6)])

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_case(params: Dict, t_eval: np.ndarray) -> pd.DataFrame:
    cfg=({**CFG,**params}); road=build_road_signals(cfg); x0=static_equilibrium_state(cfg,road)
    sol=solve_ivp(fun=lambda t,x:rhs_first_order(t,x,cfg,road),
                  t_span=(float(t_eval[0]),float(t_eval[-1])),
                  y0=x0,t_eval=t_eval,method="Radau",max_step=0.01,rtol=1e-6,atol=1e-8)
    if sol.status!=0 or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f"ODE failed: {sol.message}")
    rows=[]
    for i,t in enumerate(sol.t):
        x=sol.y[:,i]; qdd=rhs_first_order(t,x,cfg,road)[6:]
        row={"t":t}
        for j,name in enumerate(STATE_NAMES):
            row[name]=x[j]; row[f"qd_{name}"]=x[j+6]; row[f"qdd_{name}"]=qdd[j]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_per_axis_rms(df: pd.DataFrame) -> Tuple[float,float,float]:
    mask=df["t"]>=T_IGNORE; h=CFG["hcp"]
    az=df.loc[mask,"qdd_z_c"].values
    ax=-h*df.loc[mask,"qdd_th_c"].values
    ay= h*df.loc[mask,"qdd_ph_c"].values
    return (float(np.sqrt(np.mean(az**2))),
            float(np.sqrt(np.mean(ax**2))),
            float(np.sqrt(np.mean(ay**2))))

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE FUNCTION  (called by NSGA2.run)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(x: np.ndarray) -> np.ndarray:
    """
    Takes individual x (8,) → returns objectives (3,) = [rms_z, rms_x, rms_y].
    Returns large penalty on ODE failure so the solution is dominated
    and naturally pushed out of the Pareto front.
    """
    params={k:float(x[i]) for i,k in enumerate(PARAM_KEYS)}
    t0=time.time()
    try:
        df=run_one_case(params,t_eval_full)
        rms_z,rms_x,rms_y=compute_per_axis_rms(df)
        print(f"      ODE OK ({time.time()-t0:.1f}s) | "
              f"z={rms_z:.4f}  x={rms_x:.4f}  y={rms_y:.4f}")
    except Exception as exc:
        print(f"      ODE FAILED: {exc} → penalty")
        rms_z=rms_x=rms_y=99.0

    entry={**params,"rms_z":rms_z,"rms_x":rms_x,"rms_y":rms_y,
           "gen":_current_gen}
    _eval_log.append(entry)
    return np.array([rms_z,rms_x,rms_y])

# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK  (tracks HV each generation)
# ═══════════════════════════════════════════════════════════════════════════════

_current_gen = 0
_hv_history  = []
_hv_indicator = HV(ref_point=HV_REF)
_nds          = NonDominatedSorting()

def hv_callback(algorithm: NSGA2):
    global _current_gen
    _current_gen = algorithm.n_gen

    F = algorithm.pop_F
    if F is not None and len(F) > 0:
        front = _nds.do(F, only_non_dominated_front=True)
        try:    hv = _hv_indicator(F[front])
        except: hv = 0.0
        n_nd = len(front)
    else:
        hv = 0.0; n_nd = 0

    _hv_history.append(hv)
    print(f"  → Generation {algorithm.n_gen}/{N_GEN}  "
          f"Pareto size={n_nd}  HV={hv:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PARETO EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pareto(result) -> pd.DataFrame:
    X,F=result.X,result.F
    rows=[]
    for i in range(len(X)):
        row={k:float(X[i,j]) for j,k in enumerate(PARAM_KEYS)}
        row["rms_z"]=float(F[i,0]); row["rms_x"]=float(F[i,1]); row["rms_y"]=float(F[i,2])
        row["rms_total"]=float(np.sqrt(F[i,0]**2+F[i,1]**2+F[i,2]**2))
        rows.append(row)
    df=pd.DataFrame(rows).sort_values("rms_total").reset_index(drop=True)
    labels=[]
    for i in range(len(df)):
        if i==0: labels.append("Best_total")
        elif df["rms_z"].iloc[i]==df["rms_z"].min(): labels.append("Best_vertical")
        elif df["rms_x"].iloc[i]==df["rms_x"].min(): labels.append("Best_longitudinal")
        elif df["rms_y"].iloc[i]==df["rms_y"].min(): labels.append("Best_lateral")
        else: labels.append(f"Pareto_{i+1}")
    df.insert(0,"label",labels)
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_pareto_csv(df_pareto):
    path=os.path.join(RESULTS_DIR,"pareto_front.csv")
    df_pareto.to_csv(path,index=False); print(f"[CSV] → {path}"); return path

def save_run_json(df_pareto):
    valid=[e for e in _eval_log if e.get("rms_z",99)<90]
    out={"description":"NSGA-II (pure NumPy) multi-objective run. No pymoo.",
         "config":{"pop_size":POP_SIZE,"n_gen":N_GEN},
         "bounds":{k:{"lo":float(v[0]),"hi":float(v[1])} for k,v in BOUNDS_RAW.items()},
         "pareto_front":df_pareto.to_dict(orient="records"),
         "all_evaluations":valid}
    path=os.path.join(RESULTS_DIR,"run_results.json")
    with open(path,"w") as fh: json.dump(out,fh,indent=2)
    print(f"[JSON] → {path}"); return path

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS  (identical to pymoo version)
# ═══════════════════════════════════════════════════════════════════════════════

def _all_F():
    return np.array([[e["rms_z"],e["rms_x"],e["rms_y"]]
                     for e in _eval_log if e.get("rms_z",99)<90])

def plot_pareto_2d(df_pareto):
    all_f=_all_F(); obj={"rms_z":0,"rms_x":1,"rms_y":2}
    lax={"rms_z":"RMS_z vertical [m/s²]","rms_x":"RMS_x longitudinal [m/s²]","rms_y":"RMS_y lateral [m/s²]"}
    for xa,ya in [("rms_z","rms_x"),("rms_z","rms_y"),("rms_x","rms_y")]:
        fig,ax=plt.subplots(figsize=(7,5))
        if len(all_f): ax.scatter(all_f[:,obj[xa]],all_f[:,obj[ya]],c="lightgrey",s=18,zorder=1,label="All evaluated")
        sc=ax.scatter(df_pareto[xa],df_pareto[ya],c=df_pareto["rms_total"],cmap="plasma",
                      s=90,zorder=3,edgecolors="k",linewidths=0.5,label="Pareto front")
        plt.colorbar(sc,ax=ax,label="RMS_total [m/s²]")
        for _,row in df_pareto.iterrows():
            ax.annotate(row["label"],(row[xa],row[ya]),fontsize=7,xytext=(4,4),textcoords="offset points")
        ax.set_xlabel(lax[xa]); ax.set_ylabel(lax[ya])
        ax.set_title(f"Pareto Front: {xa} vs {ya}  (pure NumPy NSGA-II)")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.4); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,f"pareto_{xa}_vs_{ya}.png"),dpi=150); plt.close()
        print(f"[PLOT] pareto_{xa}_vs_{ya}.png")

def plot_pareto_3d(df_pareto):
    all_f=_all_F()
    fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection="3d")
    if len(all_f): ax.scatter(all_f[:,0],all_f[:,1],all_f[:,2],c="lightgrey",s=12,alpha=0.4,label="All evaluated")
    sc=ax.scatter(df_pareto["rms_z"],df_pareto["rms_x"],df_pareto["rms_y"],
                  c=df_pareto["rms_total"],cmap="plasma",s=100,edgecolors="k",linewidths=0.5,label="Pareto")
    plt.colorbar(sc,ax=ax,label="RMS_total",pad=0.1,shrink=0.6)
    for _,row in df_pareto.iterrows(): ax.text(row["rms_z"],row["rms_x"],row["rms_y"],row["label"],fontsize=6)
    ax.set_xlabel("RMS_z [m/s²]"); ax.set_ylabel("RMS_x [m/s²]"); ax.set_zlabel("RMS_y [m/s²]")
    ax.set_title("3-D Pareto Front  (pure NumPy NSGA-II)"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,"pareto_3d.png"),dpi=150); plt.close()
    print("[PLOT] pareto_3d.png")

def plot_hypervolume():
    plt.figure(figsize=(7,4))
    plt.plot(range(1,len(_hv_history)+1),_hv_history,marker="o",ms=4,color="steelblue")
    plt.xlabel("Generation"); plt.ylabel("Hypervolume indicator")
    plt.title("Hypervolume per Generation  (pure NumPy NSGA-II)")
    plt.grid(True,alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"hypervolume.png"),dpi=150); plt.close()
    print("[PLOT] hypervolume.png")

def plot_parallel_coordinates(df_pareto):
    norm_df=df_pareto.copy()
    for k in PARAM_KEYS:
        lo,hi=BOUNDS_RAW[k]; norm_df[k]=(df_pareto[k]-lo)/(hi-lo)
    n=len(df_pareto); cmap=cm.get_cmap("plasma",n); xs=list(range(len(PARAM_KEYS)))
    fig,ax=plt.subplots(figsize=(13,5))
    for i in range(n):
        ax.plot(xs,norm_df[PARAM_KEYS].iloc[i].values,color=cmap(i),lw=1.4,alpha=0.85,
                label=df_pareto["label"].iloc[i])
    ax.set_xticks(xs); ax.set_xticklabels(PARAM_KEYS,rotation=25,ha="right",fontsize=9)
    ax.set_ylabel("Normalised value in search range [0–1]")
    ax.set_title("Parallel Coordinates – Pareto Parameter Sets  (pure NumPy NSGA-II)")
    ax.axhline(0.05,color="red",linestyle=":",lw=0.8,label="Near lo bound")
    ax.axhline(0.95,color="orange",linestyle=":",lw=0.8,label="Near hi bound")
    ax.legend(fontsize=7,loc="upper right",ncol=2); ax.grid(True,axis="y",alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,"param_parallel.png"),dpi=150); plt.close()
    print("[PLOT] param_parallel.png")

def plot_rms_bars(df_pareto):
    labels=df_pareto["label"].tolist(); x=np.arange(len(labels)); w=0.25
    fig,ax=plt.subplots(figsize=(max(9,len(labels)*1.5),5))
    ax.bar(x-w, df_pareto["rms_z"],w,label="RMS_z (vertical)",    color="steelblue", edgecolor="k")
    ax.bar(x,   df_pareto["rms_x"],w,label="RMS_x (longitudinal)",color="darkorange",edgecolor="k")
    ax.bar(x+w, df_pareto["rms_y"],w,label="RMS_y (lateral)",     color="seagreen",  edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right")
    ax.set_ylabel("RMS acceleration [m/s²]"); ax.set_title("Per-Axis RMS – Pareto Solutions")
    ax.legend(); ax.grid(True,axis="y",alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,"pareto_rms_bars.png"),dpi=150); plt.close()
    print("[PLOT] pareto_rms_bars.png")

def plot_generation_scatter(df_pareto):
    all_f=_all_F()
    if not len(all_f): return
    gens=np.array([e["gen"] for e in _eval_log if e.get("rms_z",99)<90])
    fig,ax=plt.subplots(figsize=(7,5))
    sc=ax.scatter(all_f[:,0],all_f[:,1],c=gens,cmap=cm.get_cmap("viridis",N_GEN+1),s=25,alpha=0.7,edgecolors="none")
    plt.colorbar(sc,ax=ax,label="Generation")
    ax.scatter(df_pareto["rms_z"],df_pareto["rms_x"],s=120,marker="*",color="red",zorder=5,label="Final Pareto")
    ax.set_xlabel("RMS_z vertical [m/s²]"); ax.set_ylabel("RMS_x longitudinal [m/s²]")
    ax.set_title("Population Evolution – all generations (z vs x)")
    ax.legend(); ax.grid(True,alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,"generation_scatter.png"),dpi=150); plt.close()
    print("[PLOT] generation_scatter.png")

def plot_seat_time_history(label, params):
    try:
        df=run_one_case(params,t_eval_full); t=df["t"]; h=CFG["hcp"]
        az=df["qdd_z_c"]; ax_=-h*df["qdd_th_c"]; ay=h*df["qdd_ph_c"]
        rz,rx,ry=compute_per_axis_rms(df)
        fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
        axes[0].plot(t,az,lw=0.6,color="steelblue"); axes[0].set_ylabel("z̈ seat [m/s²]"); axes[0].grid(True,alpha=0.35)
        axes[1].plot(t,ax_,lw=0.6,color="darkorange"); axes[1].set_ylabel("ẍ seat [m/s²]"); axes[1].grid(True,alpha=0.35)
        axes[2].plot(t,ay,lw=0.6,color="seagreen"); axes[2].set_ylabel("ÿ seat [m/s²]")
        axes[2].set_xlabel("Time [s]"); axes[2].grid(True,alpha=0.35)
        fig.suptitle(f"{label}  |  z={rz:.4f}  x={rx:.4f}  y={ry:.4f}  [m/s²]")
        plt.tight_layout()
        fname=f"seat_accel_{label.replace(' ','_')}.png"
        plt.savefig(os.path.join(PLOTS_DIR,fname),dpi=150); plt.close()
        print(f"[PLOT] {fname}")
    except Exception as exc:
        print(f"[WARN] Time-history ODE failed for {label}: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n"+"="*65)
    print("  Multi-Objective Vehicle Dynamics  (pure NumPy NSGA-II)")
    print(f"  Population : {POP_SIZE}   Generations : {N_GEN}")
    print(f"  Total ODE evaluations ≈ {POP_SIZE*(N_GEN+1)}")
    print(f"  Objectives : RMS_z  RMS_x  RMS_y  (all minimised)")
    print(f"  Engine     : nsga2_engine.py  (no pymoo)")
    print("="*65+"\n")

    algorithm = NSGA2(
        pop_size = POP_SIZE,
        xl       = XL,
        xu       = XU,
        eta_c    = 15.0,
        eta_m    = 20.0,
        prob_c   = 0.9,
        seed     = 42,
    )

    result = algorithm.run(
        evaluate_fn = evaluate,
        n_gen       = N_GEN,
        callback    = hv_callback,
    )

    # ── Pareto extraction ──────────────────────────────────────────────────
    df_pareto = extract_pareto(result)

    print("\n"+"="*65)
    print("  PARETO FRONT  (non-dominated solutions from final population)")
    print("="*65)
    print(df_pareto[["label","rms_z","rms_x","rms_y","rms_total"]].to_string(index=False))

    csv_path  = save_pareto_csv(df_pareto)
    json_path = save_run_json(df_pareto)

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_pareto_2d(df_pareto)
    plot_pareto_3d(df_pareto)
    plot_hypervolume()
    plot_parallel_coordinates(df_pareto)
    plot_rms_bars(df_pareto)
    plot_generation_scatter(df_pareto)

    key_labels={"Best_total","Best_vertical","Best_longitudinal","Best_lateral"}
    done=set()
    for _,row in df_pareto.iterrows():
        if row["label"] in key_labels and row["label"] not in done:
            plot_seat_time_history(row["label"],{k:row[k] for k in PARAM_KEYS})
            done.add(row["label"])

    # ── Summary ────────────────────────────────────────────────────────────
    best=df_pareto.iloc[0]
    print("\n"+"="*65); print("  SUMMARY"); print("="*65)
    print(f"  Total ODE evaluations : {len(_eval_log)}")
    print(f"  Pareto front size     : {len(df_pareto)}")
    print(f"\n  Best-total solution  ({best['label']}):")
    for k in PARAM_KEYS:
        lo,hi=BOUNDS_RAW[k]; nv=(best[k]-lo)/(hi-lo)
        flag="  ← near boundary" if nv<0.05 or nv>0.95 else ""
        print(f"    {k:15s}: {best[k]:.6g}  (norm={nv:.3f}){flag}")
    print(f"\n  RMS_z     = {best['rms_z']:.4f} m/s²  (vertical)")
    print(f"  RMS_x     = {best['rms_x']:.4f} m/s²  (longitudinal)")
    print(f"  RMS_y     = {best['rms_y']:.4f} m/s²  (lateral)")
    print(f"  RMS_total = {best['rms_total']:.4f} m/s²")
    print(f"\n  Outputs : {RESULTS_DIR}/")
    print(f"  CSV     : {csv_path}")
    print(f"  JSON    : {json_path}")
    print(f"\n  Tip: Increase POP_SIZE=20, N_GEN=15 for richer Pareto front.")
    print(f"  When pymoo is available, swap the import line at the top of this file:")
    print(f"    from nsga2_engine import NSGA2, HV, NonDominatedSorting")
    print(f"  → will be replaced with pymoo imports (interface is identical).")
