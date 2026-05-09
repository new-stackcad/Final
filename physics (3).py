"""
physics.py  -  road signals, asymmetric damper, EOM, static equilibrium.

KEY CHANGE:
  run_one_case stores q - q_static (dynamic deviation, zero-mean oscillation).
  This removes the DC offset (~0.1 m for z_c) that caused the surrogate to
  predict flat z_c after normalisation.
  Accelerations qdd_* are UNCHANGED because d²(q-q0)/dt² = d²q/dt².
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import solve as lin_solve
from scipy.optimize import least_squares

from config import DT, STATE_NAMES, T_IGNORE, ZC, THC, PHC, ZS, THS, PHS

NDOF = 6
NC   = 2


# ── Asymmetric damper ─────────────────────────────────────────
@dataclass
class TwoStageAsymmetricDamper:
    cs_minus: float; asym_ratio: float
    gamma_c:  float; gamma_r:    float
    alpha_c:  float = -0.05
    alpha_r:  float =  0.13

    def force(self, v: float) -> float:
        cp = self.asym_ratio * self.cs_minus
        if v < 0.0:
            return (self.cs_minus * v if v >= self.alpha_c
                    else self.cs_minus*(self.alpha_c + self.gamma_c*(v - self.alpha_c)))
        else:
            return (cp * v if v <= self.alpha_r
                    else cp*(self.alpha_r + self.gamma_r*(v - self.alpha_r)))


# ── Road signals ──────────────────────────────────────────────
def load_track(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df   = pd.read_csv(csv_path, skiprows=2, header=None)
    t    = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
    z    = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
    mask = np.isfinite(t) & np.isfinite(z)
    return t[mask].astype(float), z[mask].astype(float)


def make_linear_interp(x: np.ndarray, y: np.ndarray) -> Callable:
    x, y = np.asarray(x), np.asarray(y)
    def f(xq):
        xq  = np.asarray(xq)
        xqc = np.clip(xq, x[0], x[-1])
        idx = np.clip(np.searchsorted(x, xqc) - 1, 0, len(x)-2)
        w   = (xqc - x[idx]) / np.maximum(x[idx+1]-x[idx], 1e-12)
        return y[idx]*(1-w) + y[idx+1]*w
    return f


@dataclass
class RoadSignals:
    f1L: Callable; f1R: Callable
    f2L: Callable; f2R: Callable
    f3L: Callable; f3R: Callable

    def axle_inputs(self, t: float, cfg: Dict) -> Tuple:
        r1L,r1R = float(self.f1L(t)), float(self.f1R(t))
        r2L,r2R = float(self.f2L(t)), float(self.f2R(t))
        r3L,r3R = float(self.f3L(t)), float(self.f3R(t))
        return ((r1L+r1R)/2, (r1L-r1R)/cfg["WT1"],
                (r2L+r2R)/2, (r2L-r2R)/cfg["WT2"],
                (r3L+r3R)/2, (r3L-r3R)/cfg["WT3"])

    def axle_input_rates(self, t: float, cfg: Dict, dt: float = DT) -> Tuple:
        p = self.axle_inputs(t+dt, cfg)
        m = self.axle_inputs(t-dt, cfg)
        return tuple((a-b)/(2.*dt) for a,b in zip(p,m))


def build_road_signals(cfg: Dict) -> RoadSignals:
    def lt(k): return load_track(cfg[k])
    t1L,z1L=lt("axlefront_left_csv");  t1R,z1R=lt("axlefront_right_csv")
    t2L,z2L=lt("axlerear1_left_csv");  t2R,z2R=lt("axlerear1_right_csv")
    t3L,z3L=lt("axlerear2_left_csv");  t3R,z3R=lt("axlerear2_right_csv")
    return RoadSignals(
        make_linear_interp(t1L,z1L), make_linear_interp(t1R,z1R),
        make_linear_interp(t2L,z2L), make_linear_interp(t2R,z2R),
        make_linear_interp(t3L,z3L), make_linear_interp(t3R,z3R))


def precompute_road_array(cfg: Dict, t_eval: np.ndarray) -> np.ndarray:
    road = build_road_signals(cfg)
    out  = np.empty((len(t_eval), 6), dtype=np.float32)
    for i, t in enumerate(t_eval):
        out[i] = road.axle_inputs(t, cfg)
    return out


# ── Constraints ───────────────────────────────────────────────
def geom_constraints(q, t, cfg, road):
    z_s,th_s,ph_s = q[ZS],q[THS],q[PHS]
    _,_,z2,ph2,z3,ph3 = road.axle_inputs(t, cfg)
    l2=cfg["L12"]; l3=l2+cfg["L23"]
    S2=cfg["S_tf2"]; S3=cfg["S_tf3"]
    sl2=cfg["s1"];   sl3=cfg["s2"]
    bL2=cfg["beta_L2"]; bL3=cfg["beta_L3"]
    g2 = z_s+l2*th_s+S2*ph_s - sl2*np.sin(bL2-th_s) - (z2+0.5*cfg["WT2"]*ph2)
    g3 = z_s+l3*th_s+S3*ph_s - sl3*np.sin(bL3-th_s) - (z3+0.5*cfg["WT3"]*ph3)
    G  = np.zeros((NC, NDOF))
    G[0,ZS]=1.; G[0,THS]=l2+sl2*np.cos(bL2-th_s); G[0,PHS]=S2
    G[1,ZS]=1.; G[1,THS]=l3+sl3*np.cos(bL3-th_s); G[1,PHS]=S3
    return np.array([g2,g3], dtype=float), G


# ── EOM ───────────────────────────────────────────────────────
def build_M_R(q, v, t, cfg, road):
    z_c,th_c,ph_c,z_s,th_s,ph_s       = q
    dz_c,dth_c,dph_c,dz_s,dth_s,dph_s = v
    z1f,ph_f,z2,ph2,z3,ph3             = road.axle_inputs(t, cfg)
    dz1f,dph_f,dz2,dph2,dz3,dph3       = road.axle_input_rates(t, cfg)

    pNRS2=(cfg["beta_L2"]*cfg["L_DL2"]-cfg["beta_R2"]*cfg["L_DR2"])/max(cfg["S_tf2"],1e-6)
    pNRS3=(cfg["beta_L3"]*cfg["L_DL3"]-cfg["beta_R3"]*cfg["L_DR3"])/max(cfg["S_tf3"],1e-6)

    mc=cfg["m_c"]; Ixxc=cfg["I_xxc"]; Iyyc=cfg["I_yyc"]
    ms=cfg["m_s"]; Isxx=cfg["I_sxx"]; Isyy=cfg["I_syy"]; Isxy=cfg["I_sxy"]
    S1=cfg["S_f"]; S2=cfg["S_tf2"]; S3=cfg["S_tf3"]
    a=cfg["a"]; b=cfg["b"]; hs=cfg["hs"]; g=cfg["g"]
    lcfcg=cfg["l_cfcg"]; lcrcg=cfg["l_crcg"]
    lcf=cfg["l_cf"]; lcr=cfg["l_cr"]
    lf=cfg["lf"]; hcp=cfg["hcp"]
    l2=cfg["L12"]; l3=l2+cfg["L23"]
    bL2=cfg["beta_L2"]; bR2=cfg["beta_R2"]
    bL3=cfg["beta_L3"]; bR3=cfg["beta_R3"]
    LDL2=cfg["L_DL2"]; LDR2=cfg["L_DR2"]
    LDL3=cfg["L_DL3"]; LDR3=cfg["L_DR3"]
    Kcfl=cfg["K_cfl"]; Kcfr=cfg["K_cfr"]; Kcrl=cfg["K_crl"]; Kcrr=cfg["K_crr"]
    Ccfl=cfg["C_cfl"]; Ccfr=cfg["C_cfr"]; Ccrl=cfg["C_crl"]; Ccrr=cfg["C_crr"]
    Kf=cfg["K_f"]; Cf=cfg["C_f"]
    K2=cfg["K_2"]; C2=cfg["C_2"]
    K3=cfg["K_3"]; C3=cfg["C_3"]

    M = np.zeros((NDOF, NDOF))
    M[ZC,ZC]=mc; M[THC,THC]=Iyyc; M[PHC,PHC]=Ixxc
    M[ZS,ZS]=ms; M[THS,THS]=Isyy
    M[THS,PHS]=Isxy; M[PHS,THS]=Isxy; M[PHS,PHS]=Isxx+ms*hs**2

    dm  = TwoStageAsymmetricDamper(cfg["cs_minus"],cfg["asym_ratio"],
                                    cfg["gamma_c"],cfg["gamma_r"])
    Fdf = Cf * dm.force(dz_s-lf*dth_s-dz1f)

    Cs=Ccfl+Ccfr+Ccrl+Ccrr; Ks=Kcfl+Kcfr+Kcrl+Kcrr
    R = np.zeros(NDOF)

    R[ZC]  = (Cs*(dz_c-dz_s)+Ks*(z_c-z_s)
              -(Ccfl*lcfcg+Ccfr*lcfcg-Ccrl*lcrcg-Ccrr*lcrcg)*dth_c
              -(-Ccfl*lcf-Ccfr*lcf-Ccrl*lcr-Ccrr*lcr)*dth_s
              -(-Ccfl*b+Ccfr*a-Ccrl*b+Ccrr*a)*dph_c
              -(Ccfl*b-Ccfr*a+Ccrl*b-Ccrr*a)*dph_s
              -(Kcfl*lcfcg+Kcfr*lcfcg-Kcrl*lcrcg-Kcrr*lcrcg)*th_c
              -(-Kcfl*lcf-Kcfr*lcf-Kcrl*lcr-Kcrr*lcr)*th_s
              -(-Kcfl*b+Kcfr*a-Kcrl*b+Kcrr*a)*ph_c
              -(Kcfl*b-Kcfr*a+Kcrl*b-Kcrr*a)*ph_s)

    R[THC] = (-(Ccfl*lcfcg+Ccfr*lcfcg-Ccrl*lcrcg-Ccrr*lcrcg)*dz_c
              -(-Ccfl*lcfcg-Ccfr*lcfcg-Ccrl*lcrcg-Ccrr*lcrcg)*dz_s
              -(Kcfl*lcfcg+Kcfr*lcfcg-Kcrl*lcrcg-Kcrr*lcrcg)*z_c
              -(-Kcfl*lcfcg-Kcfr*lcfcg-Kcrl*lcrcg-Kcrr*lcrcg)*z_s
              -(-Ccfl*lcfcg**2-Ccfr*lcfcg**2-Ccrl*lcrcg**2-Ccrr*lcrcg**2)*dth_c
              -(Ccfl*lcfcg*lcf+Ccfr*lcfcg*lcf-Ccrl*lcrcg*lcr-Ccrr*lcrcg*lcr)*dth_s
              -(-Ccfl*lcfcg*b+Ccfr*lcfcg*a-Ccrl*lcrcg*b+Ccrr*lcrcg*a)*dph_c
              -(Ccfl*lcfcg*b-Ccfr*lcfcg*a+Ccrl*lcrcg*b-Ccrr*lcrcg*a)*dph_s
              -(-Kcfl*lcfcg**2-Kcfr*lcfcg**2-Kcrl*lcrcg**2-Kcrr*lcrcg**2+mc*g*hcp)*th_c
              -(Kcfl*lcfcg*lcf+Kcfr*lcfcg*lcf-Kcrl*lcrcg*lcr-Kcrr*lcrcg*lcr)*th_s
              -(-Kcfl*lcfcg*b+Kcfr*lcfcg*a-Kcrl*lcrcg*b+Kcrr*lcrcg*a)*ph_c
              -(Kcfl*lcfcg*b-Kcfr*lcfcg*a+Kcrl*lcrcg*b-Kcrr*lcrcg*a)*ph_s)

    R[PHC] = (-(-Ccfl*b+Ccfr*a-Ccrl*b+Ccrr*a)*dz_c
              -(Ccfl*b-Ccfr*a+Ccrl*b-Ccrr*a)*dz_s
              -(-Kcfl*b+Kcfr*a-Kcrl*b+Kcrr*a)*z_c
              -(Kcfl*b-Kcfr*a+Kcrl*b-Kcrr*a)*z_s
              -(-Ccfl*lcfcg*b-Ccfr*lcfcg*a+Ccrl*lcrcg*b+Ccrr*lcrcg*a)*dth_c
              -(Ccfl*lcfcg*b+Ccfr*lcfcg*a-Ccrl*lcrcg*b-Ccrr*lcrcg*a)*dth_s
              -(-Ccfl*b**2+Ccfr*a**2-Ccrl*b**2+Ccrr*a**2)*dph_c
              -(Ccfl*b**2-Ccfr*a**2+Ccrl*b**2-Ccrr*a**2)*dph_s
              -(-Kcfl*lcfcg*b-Kcfr*lcfcg*a+Kcrl*lcrcg*b+Kcrr*lcrcg*a)*th_c
              -(Kcfl*lcfcg*b+Kcfr*lcfcg*a-Kcrl*lcrcg*b-Kcrr*lcrcg*a)*th_s
              -(-Kcfl*b**2+Kcfr*a**2-Kcrl*b**2+Kcrr*a**2)*ph_c
              -(Kcfl*b**2-Kcfr*a**2+Kcrl*b**2-Kcrr*a**2)*ph_s)

    R[ZS]  = (-(Ccfl+Ccfr+Ccrl+Ccrr)*dz_c
              -(-Ccfl*lcfcg-Ccfr*lcfcg+Ccrl*lcrcg+Ccrr*lcrcg)*dth_c
              -(-Ccfl-Ccfr-Ccrl-Ccrr)*dz_s
              -(Ccfl*lcf+Ccfr*lcf+Ccrl*lcr+Ccrr*lcr)*dth_s
              -(Kcfl+Kcfr+Kcrl+Kcrr)*z_c
              -(-Kcfl*lcfcg-Kcfr*lcfcg+Kcrl*lcrcg+Kcrr*lcrcg)*th_c
              -(-Kcfl-Kcfr-Kcrl-Kcrr)*z_s
              -(Kcfl*lcf+Kcfr*lcf+Kcrl*lcr+Kcrr*lcr)*th_s
              +Kf*(z_s-lf*th_s-z1f)+Fdf
              +K2*(z_s-z2-bL2*LDL2-bR2*LDR2+l2*th_s)+C2*(dz_s-dz2+l2*dth_s)
              +K3*(z_s-z3-bL3*LDL3-bR3*LDR3+l3*th_s)+C3*(dz_s-dz3+l3*dth_s))

    R[THS] = (-(Ccfl*lcfcg+Ccfr*lcfcg-Ccrl*lcrcg-Ccrr*lcrcg)*dz_c
              -(-Ccfl*lcfcg**2-Ccfr*lcfcg**2-Ccrl*lcrcg**2-Ccrr*lcrcg**2)*dth_c
              -(-Ccfl*lcf-Ccfr*lcf-Ccrl*lcr-Ccrr*lcr)*dz_s
              -(Ccfl*lcfcg*lcf+Ccfr*lcfcg*lcf-Ccrl*lcrcg*lcr-Ccrr*lcrcg*lcr)*dth_s
              -(Kcfl*lcf+Kcfr*lcf+Kcrl*lcr+Kcrr*lcr)*z_c
              -(-Kcfl*lcfcg*lcf-Kcfr*lcfcg*lcf+Kcrl*lcrcg*lcr+Kcrr*lcrcg*lcr)*th_c
              -(-Kcfl*lcf-Kcfr*lcf-Kcrl*lcr-Kcrr*lcr)*z_s
              -(Kcfl*lcf**2+Kcfr*lcf**2+Kcrl*lcr**2+Kcrr*lcr**2)*th_s
              -lf*(Kf*(z_s-lf*th_s-z1f)+Fdf)
              +l2*(K2*(z_s-z2-bL2*LDL2-bR2*LDR2+l2*th_s)+C2*(dz_s-dz2+l2*dth_s))
              +l3*(K3*(z_s-z3-bL3*LDL3-bR3*LDR3+l3*th_s)+C3*(dz_s-dz3+l3*dth_s)))

    ktf=0.5*Kf*S1**2; Kr1=0.5*K2*S2**2; Kr2=0.5*K3*S3**2
    Ctf=0.5*Cf*S1**2; Cr1=0.5*C2*S2**2; Cr2=0.5*C3*S3**2
    R[PHS] = -(ms*g*hs*ph_s
               -ktf*(ph_s-ph_f)-Ctf*(dph_s-dph_f)
               -Kr1*(ph_s-ph2-pNRS2)-Cr1*(dph_s-dph2)
               -Kr2*(ph_s-ph3-pNRS3)-Cr2*(dph_s-dph3))
    return M, R


def kkt_solve(M, R, G, gamma):
    n=NDOF; nc=NC
    A=np.zeros((n+nc,n+nc)); b=np.zeros(n+nc)
    A[:n,:n]=M; A[:n,n:]=G.T; A[n:,:n]=G
    b[:n]=-R; b[n:]=-gamma
    return lin_solve(A, b)


def rhs(t, x, cfg, road):
    q=x[:NDOF]; v=x[NDOF:]
    M,R  = build_M_R(q,v,t,cfg,road)
    gq,G = geom_constraints(q,t,cfg,road)
    w,z  = cfg["baum_omega"],cfg["baum_zeta"]
    gam  = w**2*gq + 2.*z*w*(G@v)
    sol  = kkt_solve(M,R,G,gam)
    xd   = np.empty_like(x)
    xd[:NDOF]=v; xd[NDOF:]=sol[:NDOF]
    return xd


def static_equilibrium(cfg, road, verbose=True):
    y0 = np.zeros(NDOF+NC)
    def F(y):
        q=y[:NDOF]; lam=y[NDOF:]
        M,R  = build_M_R(q,np.zeros(NDOF),0.,cfg,road)
        gq,G = geom_constraints(q,0.,cfg,road)
        return np.hstack([R+G.T@lam, 1e3*gq])
    lsq=least_squares(F,y0,method="trf",loss="soft_l1",
                      xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=800)
    if lsq.success or np.linalg.norm(F(lsq.x))<1.0:
        q0=lsq.x[:NDOF]
        if verbose:
            gn=np.linalg.norm(geom_constraints(q0,0.,cfg,road)[0])
            print(f"  Static eq. (LSQ) ||g||={gn:.3e}")
        return np.hstack([q0,np.zeros(NDOF)])
    if verbose: print("  LSQ fallback -> dynamic relaxation ...")
    cfg_r=dict(cfg)
    for k in ("C_2","C_3","C_cfl","C_cfr","C_crl","C_crr"):
        cfg_r[k]=cfg[k]*20
    sol=solve_ivp(lambda t,x:rhs(t,x,cfg_r,road),(0.,5.),
                  np.zeros(2*NDOF),method="Radau",rtol=1e-7,atol=1e-9)
    q0=sol.y[:NDOF,-1]
    lsq2=least_squares(F,np.hstack([q0,np.zeros(NC)]),method="trf",loss="soft_l1",
                       xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
    q0=lsq2.x[:NDOF] if lsq2.success else q0
    if verbose:
        gn=np.linalg.norm(geom_constraints(q0,0.,cfg,road)[0])
        print(f"  Static eq. (relax) ||g||={gn:.3e}")
    return np.hstack([q0,np.zeros(NDOF)])


def run_one_case(params, cfg_base, t_eval, verbose=False):
    """
    Solve ODE. Stores DYNAMIC DEVIATION (q - q_static) for all states.
    z_c will now oscillate around 0 instead of 0.1m  ->  normalisation works.
    Accelerations qdd_* are unchanged.
    Static offset stored in z_c_static, th_c_static, ph_c_static columns.
    """
    cfg  = {**cfg_base, **params}
    road = build_road_signals(cfg)
    try:
        x0 = static_equilibrium(cfg, road, verbose=verbose)
    except Exception as e:
        print(f"    Static eq. failed: {e}"); return None

    q_static = x0[:NDOF].copy()

    try:
        sol = solve_ivp(
            fun=lambda t,x: rhs(t,x,cfg,road),
            t_span=(t_eval[0],t_eval[-1]),
            y0=x0, t_eval=t_eval,
            method="Radau", max_step=0.01, rtol=1e-6, atol=1e-8)
    except Exception as e:
        print(f"    solve_ivp failed: {e}"); return None

    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
        print(f"    solve_ivp: {sol.message}"); return None

    rows = []
    for k in range(len(sol.t)):
        qk   = sol.y[:NDOF, k]
        qdk  = sol.y[NDOF:, k]
        tt   = float(sol.t[k])
        M,R  = build_M_R(qk,qdk,tt,cfg,road)
        gq,G = geom_constraints(qk,tt,cfg,road)
        w,z  = cfg["baum_omega"],cfg["baum_zeta"]
        gam  = w**2*gq + 2.*z*w*(G@qdk)
        qddk = kkt_solve(M,R,G,gam)[:NDOF]
        row  = {"t": tt}
        row.update(params)
        for i,n in enumerate(STATE_NAMES):
            row[n]          = float(qk[i] - q_static[i])   # dynamic deviation
            row[f"qd_{n}"]  = float(qdk[i])
            row[f"qdd_{n}"] = float(qddk[i])
        # Store static values so they can be added back for physical plots
        row["z_c_static"]  = float(q_static[0])
        row["th_c_static"] = float(q_static[1])
        row["ph_c_static"] = float(q_static[2])
        rows.append(row)
    return pd.DataFrame(rows)


def compute_seat_rms(df, cfg, t_ignore=T_IGNORE):
    """ISO 2631 3-axis combined RMS. Works on absolute or dynamic z_c."""
    mask = df["t"].values >= t_ignore
    if not mask.any(): return float("nan")
    h  = cfg.get("hcp", 0.1)
    az = df["qdd_z_c"].values[mask]
    ax = -h * df["qdd_th_c"].values[mask]
    ay =  h * df["qdd_ph_c"].values[mask]
    return float(np.sqrt(np.mean(az**2)+np.mean(ax**2)+np.mean(ay**2)))
