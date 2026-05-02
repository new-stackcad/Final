import numpy as np

import pandas as pd

from bayes_opt import BayesianOptimization

import matplotlib.pyplot as plt

import os

import numpy as np

import pandas as pd

from dataclasses import dataclass

from typing import Dict, Callable, Tuple

from numpy.linalg import solve as lin_solve

from scipy.integrate import solve_ivp

from scipy.optimize import root

from scipy.optimize import least_squares

import time

import json                         # <<< NEW: for saving best params


DT = 0.001          

FS = 1000          

T_IGNORE = 0.5

T_END = 466.945   


t_eval_full = np.arange(0.0, T_END + DT, DT)


RESULTS_DIR = "Res_Laden_single_ode_bay_opt1"

PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


os.makedirs(PLOTS_DIR, exist_ok=True)


STATE_NAMES = [

    "z_c","th_c","ph_c",

    "z_s","th_s","ph_s"]

(ZC, THC, PHC, ZS, THS, PHS) = range(6)


CFG: Dict = {

    "axlefront_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_1_FA_LH.csv",

    "axlefront_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_2_FA_RH.csv",

    "axlerear1_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_3_RA1_LH.csv",

    "axlerear1_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_4_RA1_RH.csv",

    "axlerear2_left_csv":  r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_5_RA2_LH.csv",

    "axlerear2_right_csv": r"C:\Users\inp_madhupranavi\OneDrive - Ashok Leyland Ltd\Desktop\PINN\Finalset_codes\ODE\Final_ODE\Axle Disp Data_6X4\Laden_HST_40\Displacement_6_RA2_RH.csv",

    
#a-900mm,b-1080mm, stf1-814 mm Stf2- 1043 mm

    "s1" : 0.6277,"s2" : 0.6305,

    "WT1": 0.814,"WT2": 1.047, "WT3": 1.047,


    "m_c": 862.0,"I_xxc": 516.6,"I_yyc": 1045.0,

    "M_1f": 600.0,"M_2": 1075.0,"M_3": 840.0,

    "I_xx1": 650.0,"I_xx2": 1200.0,"I_xx3": 1100.0,

    "S_tf2": 1.043,"S_tf3": 1.043,

    "S_f" : 0.814,

    "C_cfl": 5035.0,"C_cfr": 5035.0,"C_crl": 3400.0,"C_crr": 3400.0,

    "K_cfl": 49050.0,"K_cfr": 49050.0,"K_crl": 24525.0,"K_crr": 24525.0,

    "C_2": 2000,"C_3": 2000,


    "L_DL2": 0.6211, "L_DR2": 0.6211,

    "L_DL3": 0.6251, "L_DR3": 0.6251,

    "beta_L2": 0.1693,"beta_R2": 0.1693,

    "beta_L3": 0.17453,"beta_R3": 0.17453,


    "a": 0.9, "b": 1.080,    

    "l_cfcg": 0.871,"l_crcg": 1.087,

    "hcp": 0.1,


#If Unladen to be changed 

    "lf": 5.05,"L12": 0.54,"L23": 1.96,

    "l_cf": 6.458,"l_cr": 4.5,


    "m_s": 22485.0,"I_syy": 103787.0,"I_sxx": 8598.0,"I_sxy": 763.0,

    "hs": 0.68,

#hs - for laden 680mm and unladen -75 mm


#To be Optimised 

    "K_f": 474257,"C_f": 15000,"K_2": 1077620,"K_3": 1077620,


#gen

    "g": 9.81,


#Damper shape

    "cs_minus" : 0.3,

    "asym_ratio": 3.0,

    "gamma_c": 0.12,

    "gamma_r": 0.09,


#Solver choice

    "baum_omega": 10.0,

    "baum_zeta": 1.0,


}


@dataclass

class TwoStageAsymmetricDamper:

    cs_minus: float         # Compression low-speed slope

    asym_ratio: float       # c_plus / cs_minus

    gamma_c: float          # High-speed compression slope multiplier

    gamma_r: float          # High-speed rebound slope multiplier

    alpha_c: float = -0.05  # Compression velocity breakpoint

    alpha_r: float = 0.13   # Rebound velocity breakpoint


    def force(self, v_rel: float) -> float:

        c_plus = self.asym_ratio * self.cs_minus


        if v_rel < 0.0:

            if v_rel >= self.alpha_c:

                # Low-speed compression: slope = c-

                return self.cs_minus * v_rel

            else:

                # High-speed compression: slope = gamma_c * c-

                return self.cs_minus * (

                    self.alpha_c

                    + self.gamma_c * (v_rel - self.alpha_c)

                )


        else:

            if v_rel <= self.alpha_r:

                # Low-speed rebound: slope = c+

                return c_plus * v_rel

            else:

                # High-speed rebound: slope = gamma_r * c+

                return c_plus * (

                    self.alpha_r

                    + self.gamma_r * (v_rel - self.alpha_r)

                )


def load_track(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:

    df = pd.read_csv(csv_path,skiprows=2,header=None)

    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").values

    z = pd.to_numeric(df.iloc[:, 1], errors="coerce").values


    mask = np.isfinite(t) & np.isfinite(z)

    t = t[mask]

    z = z[mask]


    return t.astype(float), z.astype(float)


def make_linear_interp(x: np.ndarray, y: np.ndarray):

    x = np.asarray(x); y = np.asarray(y)

    def f(xq):

        xq = np.asarray(xq)

        xq_c = np.clip(xq, x[0], x[-1])

        idx = np.searchsorted(x, xq_c) - 1

        idx = np.clip(idx, 0, len(x)-2)

        x0, x1 = x[idx], x[idx+1]

        y0, y1 = y[idx], y[idx+1]

        w = (xq_c - x0) / np.maximum((x1-x0), 1e-12)

        return y0*(1-w) + y1*w

    return f


@dataclass

class RoadSignals:

    f1L: Callable[[np.ndarray], np.ndarray]

    f1R: Callable[[np.ndarray], np.ndarray]

    f2L: Callable[[np.ndarray], np.ndarray]

    f2R: Callable[[np.ndarray], np.ndarray]

    f3L: Callable[[np.ndarray], np.ndarray]

    f3R: Callable[[np.ndarray], np.ndarray]


    def axle_inputs(self, t: float, cfg: Dict):

        zr1L = self.f1L(t)

        zr1R = self.f1R(t)

        zr2L = self.f2L(t)

        zr2R = self.f2R(t)

        zr3L = self.f3L(t)

        zr3R = self.f3R(t)


        z1f = 0.5 * (zr1L + zr1R)

        z2  = 0.5 * (zr2L + zr2R)

        z3  = 0.5 * (zr3L + zr3R)


        ph_f = (zr1L - zr1R) / cfg["WT1"]

        ph2  = (zr2L - zr2R) / cfg["WT2"]

        ph3  = (zr3L - zr3R) / cfg["WT3"]


        return float(z1f), float(ph_f), float(z2), float(ph2), float(z3), float(ph3)


    def axle_input_rates(

        self, t: float, cfg: Dict, dt: float = DT

    ) -> Tuple[float, ...]:

        if dt is None:

            dt = DT

        p = self.axle_inputs(t + dt, cfg)

        m = self.axle_inputs(t - dt, cfg)

        return tuple((a - b) / (2.0 * dt) for a, b in zip(p, m))


def build_road_signals(cfg: Dict) -> RoadSignals:

    t1L, z1L = load_track(cfg["axlefront_left_csv"])

    t1R, z1R = load_track(cfg["axlefront_right_csv"])

    t2L, z2L = load_track(cfg["axlerear1_left_csv"])

    t2R, z2R = load_track(cfg["axlerear1_right_csv"])

    t3L, z3L = load_track(cfg["axlerear2_left_csv"])

    t3R, z3R = load_track(cfg["axlerear2_right_csv"])


    return RoadSignals(

        make_linear_interp(t1L, z1L), make_linear_interp(t1R, z1R),

        make_linear_interp(t2L, z2L), make_linear_interp(t2R, z2R),

        make_linear_interp(t3L, z3L), make_linear_interp(t3R, z3R),

    )


#Physics

def geom_constraints(q: np.ndarray, t: float, cfg: Dict, road: RoadSignals):

    z_s, th_s, ph_s = q[ZS], q[THS], q[PHS]


    _, _, z2, ph2, z3, ph3 = road.axle_inputs(t, cfg)


    l2 = cfg["L12"]

    l3 = cfg["L12"] + cfg["L23"]


    S2, S3 = cfg["S_tf2"], cfg["S_tf3"]


    sl2 = cfg["s1"]

    sl3 = cfg["s2"]


    bL2, bL3 = cfg["beta_L2"], cfg["beta_L3"]


    g2 = (

        z_s + l2*th_s + S2*ph_s

        - (sl2)*np.sin(bL2 - th_s)

        - (z2 + 0.5*cfg["WT2"]*ph2)

    )


    g3 = (

        z_s + l3*th_s + S3*ph_s

        - (sl3)*np.sin(bL3 - th_s)

        - (z3 + 0.5*cfg["WT3"]*ph3)

    )


    g = np.array([g2, g3], dtype=float)

    # 2 constraints × 6 DOF

    G = np.zeros((2, 6), dtype=float)

    G[0, ZS]  = 1.0

    G[0, THS] = l2 + (sl2)*np.cos(bL2 - th_s)

    G[0, PHS] = S2


    G[1, ZS]  = 1.0

    G[1, THS] = l3 + (sl3)*np.cos(bL3 - th_s)

    G[1, PHS] = S3


    return g, G


def build_M_R(q: np.ndarray, v: np.ndarray, t: float, cfg: Dict, road: RoadSignals) -> Tuple[np.ndarray, np.ndarray]:

    z_c, th_c, ph_c, z_s, th_s, ph_s = q

    dz_c, dth_c, dph_c, dz_s, dth_s, dph_s = v


    z1f, ph_f, z2, ph2, z3, ph3 = road.axle_inputs(t, cfg)

    dz1f, dph_f, dz2, dph2, dz3, dph3 = road.axle_input_rates(t, cfg)


    phi_NRS2 = (cfg["beta_L2"]*cfg["L_DL2"] - cfg["beta_R2"]*cfg["L_DR2"]) / max(cfg["S_tf2"], 1e-6)

    phi_NRS3 = (cfg["beta_L3"]*cfg["L_DL3"] - cfg["beta_R3"]*cfg["L_DR3"]) / max(cfg["S_tf3"], 1e-6)


    m_c, I_xxc, I_yyc = cfg["m_c"], cfg["I_xxc"], cfg["I_yyc"]

    m_s, I_sxx, I_syy, I_sxy = cfg["m_s"], cfg["I_sxx"], cfg["I_syy"], cfg["I_sxy"]

    S1, S2, S3 = cfg["S_f"],cfg["S_tf2"], cfg["S_tf3"]

    a, b = cfg["a"], cfg["b"]; hs, g = cfg["hs"], cfg["g"]

    l_cfcg, l_crcg, l_cf, l_cr = cfg["l_cfcg"], cfg["l_crcg"], cfg["l_cf"], cfg["l_cr"]

    lf = cfg["lf"]

    hcp = cfg["hcp"]

    l2 = cfg["L12"]; l3 = cfg["L12"] + cfg["L23"]

    beta_L2, beta_R2 = cfg["beta_L2"], cfg["beta_R2"]

    beta_L3, beta_R3 = cfg["beta_L3"], cfg["beta_R3"]

    L_DL2, L_DR2, L_DL3, L_DR3 = cfg["L_DL2"], cfg["L_DR2"], cfg["L_DL3"], cfg["L_DR3"]

    Kcfl,Kcfr,Kcrl,Kcrr = cfg["K_cfl"], cfg["K_cfr"], cfg["K_crl"], cfg["K_crr"]

    Ccfl,Ccfr,Ccrl,Ccrr = cfg["C_cfl"], cfg["C_cfr"], cfg["C_crl"], cfg["C_crr"]

    K_f, C_f = cfg["K_f"], cfg["C_f"]; K_2, C_2 = cfg["K_2"], cfg["C_2"]; K_3, C_3 = cfg["K_3"], cfg["C_3"]

    m_c, m_s = cfg["m_c"], cfg["m_s"]

    I_xxc, I_yyc = cfg["I_xxc"], cfg["I_yyc"]

    I_sxx, I_syy, I_sxy = cfg["I_sxx"], cfg["I_syy"], cfg["I_sxy"]

    hs, g = cfg["hs"], cfg["g"]


    M = np.zeros((6, 6), dtype=float)

    M[ZC, ZC]   = m_c

    M[THC, THC] = I_yyc

    M[PHC, PHC] = I_xxc

    M[ZS, ZS]   = m_s

    M[THS, THS] = I_syy

    M[THS, PHS] = I_sxy

    M[PHS, THS] = I_sxy

    M[PHS, PHS] = I_sxx + m_s*hs**2


    front_damper = TwoStageAsymmetricDamper(

                        cs_minus=cfg["cs_minus"],

                        asym_ratio=cfg["asym_ratio"],

                        gamma_c=cfg["gamma_c"],

                        gamma_r=cfg["gamma_r"],

                        alpha_c=-0.05,

                        alpha_r=0.13,

                    )   

    v_f = dz_s - lf*dth_s - dz1f      

    F_df = C_f*front_damper.force(v_f)                


    Csum = Ccfl + Ccfr + Ccrl + Ccrr

    Ksum = Kcfl + Kcfr + Kcrl + Kcrr

    R = np.zeros(6, dtype=float)


    R[ZC]  = ( 

                + Csum*(dz_c - dz_s) + Ksum*(z_c - z_s)

                - (Ccfl*l_cfcg + Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dth_c

                - (-Ccfl*l_cf - Ccfr*l_cf - Ccrl*l_cr - Ccrr*l_cr)*dth_s

                - (-Ccfl*b + Ccfr*a - Ccrl*b + Ccrr*a)*dph_c

                - (Ccfl*b - Ccfr*a + Ccrl*b - Ccrr*a)*dph_s

                - (Kcfl*l_cfcg + Kcfr*l_cfcg - Kcrl*l_crcg - Kcrr*l_crcg)*th_c

                - (-Kcfl*l_cf - Kcfr*l_cf - Kcrl*l_cr - Kcrr*l_cr)*th_s

                - (-Kcfl*b + Kcfr*a - Kcrl*b + Kcrr*a)*ph_c

                - (Kcfl*b - Kcfr*a + Kcrl*b - Kcrr*a)*ph_s

            )


    R[THC] = ( 

                - (Ccfl*l_cfcg + Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dz_c 

                - (-Ccfl*l_cfcg - Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dz_s

                - (Kcfl*l_cfcg + Kcfr*l_cfcg - Kcrl*l_crcg - Kcrr*l_crcg)*z_c

                - (-Kcfl*l_cfcg - Kcfr*l_cfcg - Kcrl*l_crcg - Kcrr*l_crcg)*z_s

                - (-Ccfl*l_cfcg**2 - Ccfr*l_cfcg**2 - Ccrl*l_crcg**2 - Ccrr*l_crcg**2)*dth_c

                - (Ccfl*l_cfcg*l_cf + Ccfr*l_cfcg*l_cf - Ccrl*l_crcg*l_cr - Ccrr*l_crcg*l_cr)*dth_s

                - (-Ccfl*l_cfcg*b + Ccfr*l_cfcg*a - Ccrl*l_crcg*b + Ccrr*l_crcg*a)*dph_c

                - (Ccfl*l_cfcg*b - Ccfr*l_cfcg*a + Ccrl*l_crcg*b - Ccrr*l_crcg*a)*dph_s

                - (-Kcfl*l_cfcg**2 - Kcfr*l_cfcg**2 - Kcrl*l_crcg**2 - Kcrr*l_crcg**2 + m_c*g*hcp)*th_c

                - (Kcfl*l_cfcg*l_cf + Kcfr*l_cfcg*l_cf - Kcrl*l_crcg*l_cr - Kcrr*l_crcg*l_cr)*th_s

                - (-Kcfl*l_cfcg*b + Kcfr*l_cfcg*a - Kcrl*l_crcg*b + Kcrr*l_crcg*a)*ph_c

                - (Kcfl*l_cfcg*b - Kcfr*l_cfcg*a + Kcrl*l_crcg*b - Kcrr*l_crcg*a)*ph_s

            )


    R[PHC] = (

                - (-Ccfl*b + Ccfr*a - Ccrl*b + Ccrr*a)*dz_c

                - (Ccfl*b - Ccfr*a + Ccrl*b - Ccrr*a)*dz_s

                - (-Kcfl*b + Kcfr*a - Kcrl*b + Kcrr*a)*z_c

                - (Kcfl*b - Kcfr*a + Kcrl*b - Kcrr*a)*z_s

                - (-Ccfl*l_cfcg*b - Ccfr*l_cfcg*a + Ccrl*l_crcg*b + Ccrr*l_crcg*a)*dth_c 

                - (Ccfl*l_cfcg*b + Ccfr*l_cfcg*a - Ccrl*l_crcg*b - Ccrr*l_crcg*a)*dth_s

                - (-Ccfl*b**2 + Ccfr*a**2 - Ccrl*b**2 + Ccrr*a**2)*dph_c 

                - (Ccfl*b**2 - Ccfr*a**2 + Ccrl*b**2 - Ccrr*a**2)*dph_s

                - (-Kcfl*l_cfcg*b - Kcfr*l_cfcg*a + Kcrl*l_crcg*b + Kcrr*l_crcg*a)*th_c         

                - (Kcfl*l_cfcg*b + Kcfr*l_cfcg*a - Kcrl*l_crcg*b - Kcrr*l_crcg*a)*th_s

                - (-Kcfl*b**2 + Kcfr*a**2 - Kcrl*b**2 + Kcrr*a**2)*ph_c    

                - (Kcfl*b**2 - Kcfr*a**2 + Kcrl*b**2 - Kcrr*a**2)*ph_s

            )


    R[ZS]  = ( - (Ccfl + Ccfr+ Ccrl + Ccrr)*dz_c 

               - (-Ccfl*l_cfcg - Ccfr*l_cfcg + Ccrl*l_crcg + Ccrr*l_crcg)*dth_c

               - (-Ccfl - Ccfr - Ccrl - Ccrr)*dz_s

               - (Ccfl*l_cf + Ccfr*l_cf + Ccrl*l_cr + Ccrr*l_cr)*dth_s

               - (Kcfl + Kcfr + Kcrl + Kcrr)*z_c

               - ( - Kcfl*l_cfcg - Kcfr*l_cfcg + Kcrl*l_crcg + Kcrr*l_crcg)*th_c

               - ( - Kcfl - Kcfr - Kcrl - Kcrr)*z_s

               - (Kcfl*l_cf + Kcfr*l_cf + Kcrl*l_cr + Kcrr*l_cr)*th_s

               + K_f*(z_s - lf*th_s - z1f) + F_df 

               + K_2*(z_s - z2 - beta_L2*L_DL2 - beta_R2*L_DR2 + l2*th_s) + C_2*(dz_s - dz2 + l2*dth_s)

               + K_3*(z_s - z3 - beta_L3*L_DL3 - beta_R3*L_DR3 + l3*th_s) + C_3*(dz_s - dz3 + l3*dth_s) 

            )

    

    R[THS] = ( - (Ccfl*l_cfcg + Ccfr*l_cfcg - Ccrl*l_crcg - Ccrr*l_crcg)*dz_c 

               - (-Ccfl*l_cfcg**2 - Ccfr*l_cfcg**2 - Ccrl*l_crcg**2 - Ccrr*l_crcg**2)*dth_c

                - (-Ccfl*l_cf - Ccfr*l_cf - Ccrl*l_cr - Ccrr*l_cr)*dz_s

                - (Ccfl*l_cfcg*l_cf + Ccfr*l_cfcg*l_cf - Ccrl*l_crcg*l_cr - Ccrr*l_crcg*l_cr)*dth_s

                - (Kcfl*l_cf + Kcfr*l_cf + Kcrl*l_cr + Kcrr*l_cr)*z_c

                - (-Kcfl*l_cfcg*l_cf - Kcfr*l_cfcg*l_cf + Kcrl*l_crcg*l_cr + Kcrr*l_crcg*l_cr)*th_c

                - (-Kcfl*l_cf - Kcfr*l_cf - Kcrl*l_cr - Kcrr*l_cr)*z_s

                - (Kcfl*l_cf**2 + Kcfr*l_cf**2 + Kcrl*l_cr**2 + Kcrr*l_cr**2)*th_s

               - lf*(K_f*(z_s - lf*th_s - z1f) + F_df )

               + l2*(K_2*(z_s - z2 - beta_L2*L_DL2 - beta_R2*L_DR2 + l2*th_s) + C_2*(dz_s - dz2 + l2*dth_s))

               + l3*(K_3*(z_s - z3 - beta_L3*L_DL3 - beta_R3*L_DR3 + l3*th_s) + C_3*(dz_s - dz3 + l3*dth_s)) 

            )

    k_tf = 0.5*K_f*S1**2

    K_r1 = 0.5*K_2*S2**2

    K_r2 = 0.5*K_3*S3**2

    C_tf = 0.5*C_f*S1**2

    C_r1 = 0.5*C_2*S2**2

    C_r2 = 0.5*C_3*S3**2


    R[PHS] = ( + m_s*g*hs*ph_s

               - k_tf*(ph_s - ph_f) - C_tf*(dph_s - dph_f)

               - K_r1*(ph_s - ph2 - phi_NRS2) - C_r1*(dph_s - dph2)

               - K_r2*(ph_s - ph3 - phi_NRS3) - C_r2*(dph_s - dph3) )


    R[PHS] *= -1.0

    return M, R


#Solving

def rhs_first_order(t: float, x: np.ndarray, cfg: Dict, road: RoadSignals):

    q = x[:6]

    v = x[6:]


    M, R = build_M_R(q, v, t, cfg, road)

    gq, G = geom_constraints(q, t, cfg, road)


    w = cfg["baum_omega"]

    zeta = cfg["baum_zeta"]

    gamma = w**2*gq + 2*zeta*w*(G @ v)


    nc = G.shape[0]

    A = np.zeros((6 + nc, 6 + nc))

    b = np.zeros(6 + nc)


    A[:6, :6]   = M

    A[:6, 6:]   = G.T

    A[6:, :6]   = G


    b[:6] = -R

    b[6:] = -gamma


    sol = lin_solve(A, b)

    qdd = sol[:6]


    xdot = np.zeros_like(x)

    xdot[:6] = v

    xdot[6:] = qdd


    return xdot


def static_equilibrium_state(cfg: Dict, road: RoadSignals) -> np.ndarray:

    q_seed = np.zeros(6, dtype=float)

    lam_seed = np.zeros(2, dtype=float)

    y0 = np.hstack([q_seed, lam_seed])


    t0 = 0.0  


    def F(y: np.ndarray) -> np.ndarray:

        q = y[:6]

        lam = y[6:]

        v = np.zeros(6, dtype=float)


        M, R = build_M_R(q, v, t0, cfg, road)

        gq, G = geom_constraints(q, t0, cfg, road)


        eq_dyn = R + G.T @ lam          # size 6

        alpha = 1e3

        eq_con = alpha * gq             # size 2


        return np.hstack([eq_dyn, eq_con])  # size 8

    

    lsq = least_squares(

        F, y0,

        method="trf",

        loss="soft_l1",

        xtol=1e-12,

        ftol=1e-12,

        gtol=1e-12,

        max_nfev=800

    )


    if lsq.success:

        q0 = lsq.x[:6]

        v0 = np.zeros(6, dtype=float)


        g0, G0 = geom_constraints(q0, t0, cfg, road)

        M0, R0 = build_M_R(q0, v0, t0, cfg, road)

        eq_dyn0 = R0 + G0.T @ lsq.x[6:]


        print(

            "=== GEN: Static equilibrium (LSQ) OK. "

            "||g||=%.3e, ||R+G^Tλ||=%.3e"

            % (np.linalg.norm(g0), np.linalg.norm(eq_dyn0))

        )


        return np.hstack([q0, v0])


    print(

        "=== GEN: Static equilibrium LSQ failed ('%s'); trying dynamic relaxation..."

        % lsq.message

    )


    cfg_relax = dict(cfg)

    damp_mult = 20.0

    cfg_relax["C_f"]   = cfg["C_f"]   

    cfg_relax["C_2"]   = cfg["C_2"]   * damp_mult

    cfg_relax["C_3"]   = cfg["C_3"]   * damp_mult

    cfg_relax["C_cfl"] = cfg["C_cfl"] * damp_mult

    cfg_relax["C_cfr"] = cfg["C_cfr"] * damp_mult

    cfg_relax["C_crl"] = cfg["C_crl"] * damp_mult

    cfg_relax["C_crr"] = cfg["C_crr"] * damp_mult


    road_relax = road


    x_init = np.hstack([q_seed, np.zeros(6, dtype=float)])


    sol = solve_ivp(

        fun=lambda t, x: rhs_first_order(t, x, cfg_relax, road_relax),

        t_span=(0.0, 3.0),

        y0=x_init,

        method="Radau",

        rtol=1e-7,

        atol=1e-9

    )


    q_relax = sol.y[:6, -1]


    y_pol = np.hstack([q_relax, np.zeros(2, dtype=float)])

    lsq2 = least_squares(

        F, y_pol,

        method="trf",

        loss="soft_l1",

        xtol=1e-12,

        ftol=1e-12,

        gtol=1e-12,

        max_nfev=400

    )


    q0 = lsq2.x[:6] if lsq2.success else q_relax

    v0 = np.zeros(6, dtype=float)


    print(

        "=== GEN: Dynamic relaxation end. ||g||=%.3e"

        % np.linalg.norm(geom_constraints(q0, t0, cfg, road)[0])

    )


    return np.hstack([q0, v0])


def run_one_case(params: Dict, cfg_base: Dict, t_eval: np.ndarray) -> pd.DataFrame:


    cfg = dict(cfg_base)

    cfg.update(params)


    road = build_road_signals(cfg)


    x0 = static_equilibrium_state(cfg, road)  


    t0 = float(t_eval[0])

    tf = float(t_eval[-1])

    print(

        f"\n=== GEN: Integrating (Radau) | "

        f"T_end = {tf:.2f} s | dt = {t_eval[1] - t_eval[0]:.4f} s"

    )


    wall_start = time.time()


    sol = solve_ivp(

        fun=lambda t, x: rhs_first_order(t, x, cfg, road),

        t_span=(t0, tf),

        y0=x0,

        t_eval=t_eval,

        method="Radau",

        max_step = 0.01,

        rtol=1e-6,

        atol=1e-8

    )


    print(

        f"=== GEN: solve_ivp success={sol.success}, "

        f"message='{sol.message}', nfev={sol.nfev}"

    )


    if sol.status != 0 or not np.all(np.isfinite(sol.y)):

        raise RuntimeError("ODE integration failed or diverged")


    print(

        f"=== GEN: Integration OK | steps={len(sol.t)} | "

        f"wall={time.time() - wall_start:.1f} s\n"

    )


    rows = []


    for i, t in enumerate(sol.t):

        x = sol.y[:, i]

        q = x[:6]

        v = x[6:]


        # Recompute acceleration (consistent with your model)

        qdd = rhs_first_order(t, x, cfg, road)[6:]


        row = {"t": t}

        for j, name in enumerate(STATE_NAMES):

            row[name] = q[j]

            row[f"qd_{name}"] = v[j]

            row[f"qdd_{name}"] = qdd[j]


        rows.append(row)


    return pd.DataFrame(rows)


# ===========================================================================
# CHANGE 1: 3-axis combined seat RMS
#   z̈_seat = z̈_c           (vertical bounce of cabin)
#   ẍ_seat = -h * θ̈_c       (longitudinal due to pitch, h = hcp)
#   ÿ_seat =  h * φ̈_c       (lateral due to roll,  h = hcp)
#   RMS_total = sqrt( mean(z̈²) + mean(ẍ²) + mean(ÿ²) )
# ===========================================================================

def compute_seat_rms(df: pd.DataFrame, cfg: Dict) -> float:
    """
    Combined 3-axis seat-point acceleration RMS (ISO 2631-style scalar).

    z̈_seat = qdd_z_c
    ẍ_seat = -hcp * qdd_th_c   (pitch contribution at seat)
    ÿ_seat =  hcp * qdd_ph_c   (roll  contribution at seat)
    """
    mask = df["t"] >= T_IGNORE
    h = cfg["hcp"]

    az = df.loc[mask, "qdd_z_c"].values
    ax = -h * df.loc[mask, "qdd_th_c"].values   # longitudinal (pitch)
    ay =  h * df.loc[mask, "qdd_ph_c"].values   # lateral      (roll)

    # Combined RMS: sqrt( E[az²] + E[ax²] + E[ay²] )
    rms_total = np.sqrt(np.mean(az**2) + np.mean(ax**2) + np.mean(ay**2))
    return float(rms_total)


def compute_seat_rms_axes(df: pd.DataFrame, cfg: Dict) -> Dict[str, float]:
    """Returns per-axis RMS values for diagnostics / plotting."""
    mask = df["t"] >= T_IGNORE
    h = cfg["hcp"]

    az = df.loc[mask, "qdd_z_c"].values
    ax = -h * df.loc[mask, "qdd_th_c"].values
    ay =  h * df.loc[mask, "qdd_ph_c"].values

    return {
        "rms_z": float(np.sqrt(np.mean(az**2))),
        "rms_x": float(np.sqrt(np.mean(ax**2))),
        "rms_y": float(np.sqrt(np.mean(ay**2))),
        "rms_total": float(np.sqrt(np.mean(az**2) + np.mean(ax**2) + np.mean(ay**2))),
    }


def objective(K_f, C_f, K_2, K_3, cs_minus, asym_ratio, gamma_c, gamma_r):

    params = {

        "K_f": K_f,

        "C_f": C_f,

        "K_2": K_2,

        "K_3": K_3,

        "cs_minus": cs_minus,

        "asym_ratio": asym_ratio,

        "gamma_c": gamma_c,

        "gamma_r": gamma_r,

    }


    df = run_one_case(params, CFG, t_eval_full)

    rms = compute_seat_rms(df, CFG)        # <<< now uses 3-axis formula

    return -rms


bounds = {

    # -------------------------

    # Vehicle / suspension params (existing)

    # -------------------------

    "K_f": (0.879 * CFG["K_f"], 1.126 * CFG["K_f"]),

    "C_f": (0.44   * CFG["C_f"], 1.4   * CFG["C_f"]),

    "K_2": (0.8920 * CFG["K_2"], 1.116 * CFG["K_2"]),

    "K_3": (0.8920 * CFG["K_3"], 1.116 * CFG["K_3"]),


    # -------------------------

    # Damper SHAPE parameters (image-based)

    # -------------------------

    "cs_minus":   (0.2, 0.4),

    "asym_ratio": (2.3, 4.0),

    "gamma_c":    (0.08, 0.16),

    "gamma_r":    (0.08, 0.10),

}


# ===========================================================================
# CHANGE 2: Normalised best-parameter selection
#   Each parameter is expressed as a fraction of its search-range width so
#   that stiffnesses (O(10^5-10^6)) cannot dominate dimensionless damper
#   shape coefficients (O(0.1-4)).  The "best" iteration is simply the one
#   with the lowest (most negative) objective value – that part is unchanged.
#   What is added is (a) saving to JSON and (b) a "normalised value" column
#   in the parameter bar-plot so you can see where in [0, 1] each param sits.
# ===========================================================================

def select_best_params(optimizer: BayesianOptimization) -> Dict:
    """
    Returns the parameter set corresponding to the iteration with the lowest
    combined-RMS objective.  The selection is made on the raw objective value
    (as BayesianOptimization.max does), but the result dict is augmented with
    normalised values [0, 1] per parameter so you can check boundary proximity.
    """
    best_res = optimizer.max                      # dict: {"target": ..., "params": {...}}
    raw = best_res["params"].copy()

    normalised = {}
    for key, val in raw.items():
        lo, hi = bounds[key]
        normalised[key] = round((val - lo) / (hi - lo), 4)

    return {"params": raw, "normalised_in_range": normalised, "rms_combined": -best_res["target"]}


def save_best_params_json(best_info: Dict, save_dir: str) -> str:
    """Saves best params + metadata to JSON in save_dir."""
    out = {
        "description": (
            "Best Bayesian-optimised parameters. "
            "'params' are physical values; "
            "'normalised_in_range' shows position within the search bounds "
            "(0 = lower bound, 1 = upper bound). "
            "Values near 0 or 1 indicate the optimum may lie outside the current bounds."
        ),
        "bounds": {k: {"lo": float(v[0]), "hi": float(v[1])} for k, v in bounds.items()},
        "best": {
            "params": {k: float(v) for k, v in best_info["params"].items()},
            "normalised_in_range": best_info["normalised_in_range"],
            "rms_combined_m_s2": float(best_info["rms_combined"]),
        },
    }
    path = os.path.join(save_dir, "best_params.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[JSON] Best params saved → {path}")
    return path


def plot_road_inputs(road, cfg, t_eval, save_dir):

    z1, z2, z3 = [], [], []


    for t in t_eval:

        z1f, ph_f, z2i, ph2, z3i, ph3 = road.axle_inputs(t, cfg)

        z1.append(z1f)

        z2.append(z2i)

        z3.append(z3i)


    plt.figure()

    plt.plot(t_eval, z1, label="Front axle")

    plt.plot(t_eval, z2, label="Rear axle 1")

    plt.plot(t_eval, z3, label="Rear axle 2")

    plt.xlabel("Time [s]")

    plt.ylabel("Road displacement [m]")

    plt.title("Road / Axle Vertical Inputs")

    plt.legend()

    plt.grid(True)

    plt.savefig(os.path.join(save_dir, "road_inputs.png"))

    plt.close()


def plot_vehicle_response(df, label, save_dir):

    t = df["t"]


    # Sprung mass displacement

    plt.figure()

    plt.plot(t, df["z_s"], label=label)

    plt.xlabel("Time [s]")

    plt.ylabel("zₛ [m]")

    plt.title(f"Sprung Mass Vertical Displacement ({label})")

    plt.legend()

    plt.grid(True)

    plt.savefig(os.path.join(save_dir, f"z_s_displacement_{label.lower()}.png"))

    plt.close()


    # Sprung mass acceleration

    plt.figure()

    plt.plot(t, df["qdd_z_s"], label=label)

    plt.xlabel("Time [s]")

    plt.ylabel("z̈ₛ [m/s²]")

    plt.title(f"Sprung Mass Vertical Acceleration ({label})")

    plt.legend()

    plt.grid(True)

    plt.savefig(os.path.join(save_dir, f"z_s_acceleration_{label.lower()}.png"))

    plt.close()


def plot_seat_response(df, cfg, label, save_dir):

    t = df["t"]

    h = cfg["hcp"]

    az = df["qdd_z_c"]
    ax = -h * df["qdd_th_c"]
    ay =  h * df["qdd_ph_c"]


    # Time history – all three axes

    plt.figure(figsize=(9, 4))

    plt.plot(t, az, label="z̈_seat (vertical)", linewidth=0.8)

    plt.plot(t, ax, label="ẍ_seat (longitudinal)", linewidth=0.8)

    plt.plot(t, ay, label="ÿ_seat (lateral)", linewidth=0.8)

    plt.xlabel("Time [s]")

    plt.ylabel("Seat acceleration [m/s²]")

    plt.title(f"Seat (Cabin) 3-Axis Acceleration ({label})")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"seat_acceleration_time_{label.lower()}.png"))

    plt.close()


    # Histogram – all three axes

    plt.figure(figsize=(9, 4))

    plt.hist(az, bins=50, alpha=0.6, label="z̈_seat")

    plt.hist(ax, bins=50, alpha=0.6, label="ẍ_seat")

    plt.hist(ay, bins=50, alpha=0.6, label="ÿ_seat")

    plt.xlabel("Seat acceleration [m/s²]")

    plt.ylabel("Count")

    plt.title(f"Seat 3-Axis Acceleration Distribution ({label})")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"seat_acceleration_hist_{label.lower()}.png"))

    plt.close()


def plot_convergence(optimizer, save_dir):

    best = np.minimum.accumulate([-res["target"] for res in optimizer.res])


    plt.figure()

    plt.plot(best)

    plt.xlabel("Iteration")

    plt.ylabel("Best Combined Seat RMS [m/s²]")

    plt.title("Optimization Convergence (3-Axis Combined RMS)")

    plt.grid(True)

    plt.savefig(os.path.join(save_dir, "convergence.png"))

    plt.close()


def plot_parameter_evolution(optimizer, save_dir):

    it = range(len(optimizer.res))


    param_keys = [

        "K_f", "C_f", "K_2", "K_3",

        "cs_minus", "asym_ratio", "gamma_c", "gamma_r"

    ]


    for key in param_keys:

        plt.figure()

        plt.plot(it, [r["params"][key] for r in optimizer.res])

        plt.xlabel("Iteration")

        plt.ylabel(key)

        plt.title(f"{key} Evolution")

        plt.grid(True)

        plt.savefig(os.path.join(save_dir, f"{key}_evolution.png"))

        plt.close()


# <<< NEW: normalised bar-chart so all params appear on the same [0,1] scale

def plot_normalised_params(best_info: Dict, save_dir: str):
    """
    Bar chart of each parameter's normalised position in its search range.
    Red dashed lines at 0.05 / 0.95 flag near-boundary values that may
    indicate the true optimum lies outside the current bounds.
    """
    keys = list(best_info["normalised_in_range"].keys())
    vals = [best_info["normalised_in_range"][k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(keys, vals, color="steelblue", edgecolor="k")
    ax.axhline(0.05, color="red",   linestyle="--", linewidth=1.0, label="Near lower bound (0.05)")
    ax.axhline(0.95, color="orange", linestyle="--", linewidth=1.0, label="Near upper bound (0.95)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalised position in search range  [0 = lo, 1 = hi]")
    ax.set_title("Optimal Parameters – Normalised Position in Search Bounds")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.5)

    # Annotate physical values on bars
    for bar, key in zip(bars, keys):
        phys = best_info["params"][key]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{phys:.4g}",
            ha="center", va="bottom", fontsize=8, rotation=30
        )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_params_normalised.png"))
    plt.close()
    print(f"[PLOT] Normalised param chart saved → {save_dir}/best_params_normalised.png")


def plot_seat_comparison(df_base, df_opt, cfg, base_rms, opt_rms, save_dir):

    t = df_base["t"]

    h = cfg["hcp"]

    # --- vertical (z) ---
    az_base = df_base["qdd_z_c"]
    az_opt  = df_opt["qdd_z_c"]

    # --- longitudinal (x, from pitch) ---
    ax_base = -h * df_base["qdd_th_c"]
    ax_opt  = -h * df_opt["qdd_th_c"]

    # --- lateral (y, from roll) ---
    ay_base =  h * df_base["qdd_ph_c"]
    ay_opt  =  h * df_opt["qdd_ph_c"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t, az_base, label=f"Baseline (RMS_z={np.sqrt(np.mean(az_base[df_base['t']>=T_IGNORE]**2)):.3f})", alpha=0.8)
    axes[0].plot(t, az_opt,  label=f"Optimized (RMS_z={np.sqrt(np.mean(az_opt[df_opt['t']>=T_IGNORE]**2)):.3f})", alpha=0.8)
    axes[0].set_ylabel("z̈_seat [m/s²]")
    axes[0].set_title(f"Seat Vertical Accel  |  Combined: Baseline={base_rms:.3f}, Opt={opt_rms:.3f} m/s²")
    axes[0].legend(fontsize=8); axes[0].grid(True)

    axes[1].plot(t, ax_base, label="Baseline", alpha=0.8)
    axes[1].plot(t, ax_opt,  label="Optimized", alpha=0.8)
    axes[1].set_ylabel("ẍ_seat = -h·θ̈ [m/s²]")
    axes[1].set_title("Seat Longitudinal Accel (Pitch contribution)")
    axes[1].legend(fontsize=8); axes[1].grid(True)

    axes[2].plot(t, ay_base, label="Baseline", alpha=0.8)
    axes[2].plot(t, ay_opt,  label="Optimized", alpha=0.8)
    axes[2].set_ylabel("ÿ_seat = h·φ̈ [m/s²]")
    axes[2].set_title("Seat Lateral Accel (Roll contribution)")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(fontsize=8); axes[2].grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "seat_acceleration_comparison.png"))

    plt.close()


# <<< NEW: per-axis RMS bar chart (baseline vs optimised)

def plot_per_axis_rms_comparison(axes_base: Dict, axes_opt: Dict, save_dir: str):
    """Grouped bar chart: per-axis RMS for baseline vs optimised."""
    labels   = ["RMS_z (vertical)", "RMS_x (longitudinal)", "RMS_y (lateral)", "RMS_total"]
    keys     = ["rms_z", "rms_x", "rms_y", "rms_total"]
    base_vals = [axes_base[k] for k in keys]
    opt_vals  = [axes_opt[k]  for k in keys]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, base_vals, w, label="Baseline",  color="steelblue",  edgecolor="k")
    ax.bar(x + w/2, opt_vals,  w, label="Optimized", color="darkorange", edgecolor="k")

    for xi, bv, ov in zip(x, base_vals, opt_vals):
        ax.text(xi - w/2, bv + 0.0005, f"{bv:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w/2, ov + 0.0005, f"{ov:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("RMS acceleration [m/s²]")
    ax.set_title("Per-Axis Seat RMS: Baseline vs Optimized")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_axis_rms_comparison.png"))
    plt.close()
    print(f"[PLOT] Per-axis RMS chart saved → {save_dir}/per_axis_rms_comparison.png")


def plot_sprung_displacement_comparison(df_base, df_opt, save_dir):

    t = df_base["t"]


    plt.figure(figsize=(8, 4))

    plt.plot(t, df_base["z_s"], label="Baseline", alpha=0.8)

    plt.plot(t, df_opt["z_s"],  label="Optimized", alpha=0.8)


    plt.xlabel("Time [s]")

    plt.ylabel("Sprung mass displacement zₛ [m]")

    plt.title("Sprung Mass Vertical Displacement")

    plt.legend()

    plt.grid(True)


    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "sprung_mass_displacement_comparison.png"))

    plt.close()


def plot_sprung_accel_comparison(df_base, df_opt, save_dir):

    t = df_base["t"]


    plt.figure(figsize=(8, 4))

    plt.plot(t, df_base["qdd_z_s"], label="Baseline", alpha=0.8)

    plt.plot(t, df_opt["qdd_z_s"],  label="Optimized", alpha=0.8)


    plt.xlabel("Time [s]")

    plt.ylabel("Sprung mass acceleration z̈ₛ [m/s²]")

    plt.title("Sprung Mass Vertical Acceleration")

    plt.legend()

    plt.grid(True)


    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "sprung_mass_acceleration_comparison.png"))

    plt.close()


if __name__ == "__main__":


    optimizer = BayesianOptimization(

        f=objective,

        pbounds=bounds,

        random_state=123,

    )


    print("Running Bayesian Optimization...")

    optimizer.maximize(init_points=8, n_iter=30)


    print("\nBest result:")

    print(optimizer.max)


    # ===========================================================================
    # CHANGE 2: Normalised best-param selection + JSON save
    # ===========================================================================

    best_info = select_best_params(optimizer)

    print("\n===== BEST PARAMS (normalised position in search range) =====")

    for k, v in best_info["normalised_in_range"].items():

        flag = "  *** near boundary ***" if v < 0.05 or v > 0.95 else ""

        print(f"  {k:15s}: {best_info['params'][k]:.6g}  (norm={v:.4f}){flag}")

    json_path = save_best_params_json(best_info, RESULTS_DIR)


    print("\nRunning post-optimization analysis...")


    baseline_params = {

        "K_f": CFG["K_f"],

        "C_f": CFG["C_f"],

        "K_2": CFG["K_2"],

        "K_3": CFG["K_3"],

        "cs_minus": CFG["cs_minus"],

        "asym_ratio": CFG["asym_ratio"],

        "gamma_c": CFG["gamma_c"],

        "gamma_r": CFG["gamma_r"],

    }


    df_baseline = run_one_case(baseline_params, CFG, t_eval_full)

    baseline = compute_seat_rms(df_baseline, CFG)

    axes_base = compute_seat_rms_axes(df_baseline, CFG)


    best_params = best_info["params"]


    df_optimized = run_one_case(

        {

            "K_f": best_params["K_f"],

            "C_f": best_params["C_f"],

            "K_2": best_params["K_2"],

            "K_3": best_params["K_3"],

            "cs_minus": best_params["cs_minus"],

            "asym_ratio": best_params["asym_ratio"],

            "gamma_c": best_params["gamma_c"],

            "gamma_r": best_params["gamma_r"],

        },

        CFG,

        t_eval_full

    )


    optimized = compute_seat_rms(df_optimized, CFG)

    axes_opt = compute_seat_rms_axes(df_optimized, CFG)


    # ============================================================

    # SUMMARY

    # ============================================================

    print("\n===== FINAL RESULTS (3-AXIS COMBINED RMS) =====")

    print("Best Parameters (physical):", best_params)

    print(f"Baseline  RMS total : {baseline:.4f} m/s²")

    print(f"  per-axis → z:{axes_base['rms_z']:.4f}  x:{axes_base['rms_x']:.4f}  y:{axes_base['rms_y']:.4f}")

    print(f"Optimized RMS total : {optimized:.4f} m/s²")

    print(f"  per-axis → z:{axes_opt['rms_z']:.4f}  x:{axes_opt['rms_x']:.4f}  y:{axes_opt['rms_y']:.4f}")

    print(f"Improvement (%)     : {((baseline - optimized) / baseline) * 100:.2f}%")


    # ============================================================

    # PLOTS

    # ============================================================

    plot_vehicle_response(df_baseline, "Baseline", PLOTS_DIR)

    plot_seat_response(df_baseline, CFG, "Baseline", PLOTS_DIR)


    plot_vehicle_response(df_optimized, "Optimized", PLOTS_DIR)

    plot_seat_response(df_optimized, CFG, "Optimized", PLOTS_DIR)


    road = build_road_signals(CFG)

    plot_road_inputs(road, CFG, t_eval_full, PLOTS_DIR)


    plot_convergence(optimizer, PLOTS_DIR)

    plot_parameter_evolution(optimizer, PLOTS_DIR)


    plot_seat_comparison(

        df_baseline,

        df_optimized,

        CFG,

        baseline,

        optimized,

        PLOTS_DIR

    )


    plot_sprung_displacement_comparison(

        df_baseline,

        df_optimized,

        PLOTS_DIR

    )


    plot_sprung_accel_comparison(

        df_baseline,

        df_optimized,

        PLOTS_DIR

    )


    # <<< NEW plots

    plot_normalised_params(best_info, PLOTS_DIR)

    plot_per_axis_rms_comparison(axes_base, axes_opt, PLOTS_DIR)


    print(f"\nAll plots saved in: {PLOTS_DIR}")

    print(f"Best params JSON  : {json_path}")

    print(f"Final validated combined RMS: {optimized:.4f} m/s²")
