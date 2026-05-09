"""
optimiser.py  (generic)
=======================
Two modes:
  nsga2     -> NSGA-II multi-objective (RMS_z, RMS_x, RMS_y)
  bayesian  -> Bayesian single-objective (RMS_total)

Works with any UserModel — no assumptions about ODE structure.

Public API
----------
run_optimisation(model, signals, t_eval, mode, opt_settings, log_cb) -> OptResult
"""

from __future__ import annotations
import copy, time, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from user_code_runner import UserModel
from solver import run_one_case, compute_cabin_rms

warnings.filterwarnings("ignore")

PARAM_KEYS_FROM_BOUNDS = None   # derived per-run from model.bounds


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class OptResult:
    mode:         str
    df_pareto:    pd.DataFrame
    eval_log:     List[Dict]
    hv_history:   List[float]
    convergence:  List[float]
    wall_seconds: float
    n_evals:      int
    param_keys:   List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared eval wrapper
# ---------------------------------------------------------------------------
def _evaluate(params: Dict, model: UserModel, signals: Dict,
               t_eval: np.ndarray, log_cb: Callable, label: str) -> Dict:
    try:
        df  = run_one_case(params, model, signals, t_eval)
        rms = compute_cabin_rms(df, model)
    except Exception as e:
        log_cb(f"  [WARN] {label} failed: {e}")
        rms = {"rms_z": 99.0, "rms_x": 99.0, "rms_y": 99.0, "rms_total": 99.0}
    return rms


# ===========================================================================
# NSGA-II
# ===========================================================================
def _run_nsga2(model, signals, t_eval, pop_size, n_gen, log_cb) -> OptResult:
    from pymoo.core.problem          import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2  import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm   import PM
    from pymoo.operators.sampling.rnd  import FloatRandomSampling
    from pymoo.optimize                import minimize as pymoo_minimize
    from pymoo.indicators.hv           import HV
    from pymoo.core.callback           import Callback

    param_keys = list(model.bounds.keys())
    XL = np.array([float(model.bounds[k][0]) for k in param_keys])
    XU = np.array([float(model.bounds[k][1]) for k in param_keys])
    HV_REF = np.array([5.0, 5.0, 5.0])

    eval_log:   List[Dict]  = []
    hv_history: List[float] = []

    class VehicleProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(param_keys), n_obj=3, n_constr=0,
                             xl=XL, xu=XU)

        def _evaluate(self, x, out, *args, **kwargs):
            params = dict(zip(param_keys, x.tolist()))
            gen    = getattr(self, "_gen", 0)
            rms    = _evaluate(params, model, signals, t_eval, log_cb,
                               f"gen={gen} eval={len(eval_log)+1}")
            f      = [rms["rms_z"], rms["rms_x"], rms["rms_y"]]
            out["F"] = f
            eval_log.append({**params, **rms, "gen": gen})
            log_cb(
                f"  eval {len(eval_log):>4d} | gen={gen} "
                f"| z={f[0]:.4f} x={f[1]:.4f} y={f[2]:.4f}"
            )

    problem = VehicleProblem()

    class HVCallback(Callback):
        def notify(self, algorithm):
            gen = algorithm.n_gen
            for e in eval_log:
                if e.get("gen", 0) == 0:
                    e["gen"] = gen
            F     = algorithm.pop.get("F")
            valid = F[np.all(F < 90, axis=1)]
            hv    = HV(ref_point=HV_REF)
            hv_history.append(float(hv.do(valid)) if len(valid) else 0.0)
            log_cb(f"[GEN {gen:>3d}/{n_gen}]  HV={hv_history[-1]:.4f}")

    log_cb(f"NSGA-II | pop={pop_size} gen={n_gen} | params={param_keys}")
    t0 = time.time()

    result = pymoo_minimize(
        problem,
        NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        ),
        ("n_gen", n_gen),
        callback=HVCallback(),
        seed=42, verbose=False,
    )

    wall = time.time() - t0
    log_cb(f"NSGA-II done in {wall:.1f}s | evals={len(eval_log)}")

    rows = []
    X, F = result.opt.get("X"), result.opt.get("F")
    for i, (xi, fi) in enumerate(zip(X, F)):
        params = dict(zip(param_keys, xi.tolist()))
        rows.append({
            **params,
            "rms_z": float(fi[0]), "rms_x": float(fi[1]),
            "rms_y": float(fi[2]),
            "rms_total": float(np.sum(fi)),
            "label": f"P{i+1:02d}",
        })

    df = pd.DataFrame(rows).sort_values("rms_total").reset_index(drop=True)
    for col, lbl in [("rms_z","Best_vertical"), ("rms_x","Best_longitudinal"),
                     ("rms_y","Best_lateral"),   ("rms_total","Best_total")]:
        df.loc[df[col].idxmin(), "label"] = lbl

    return OptResult("nsga2", df, eval_log, hv_history, [], wall, len(eval_log), param_keys)


# ===========================================================================
# Bayesian
# ===========================================================================
def _run_bayesian(model, signals, t_eval, n_calls, n_initial, log_cb) -> OptResult:
    from skopt         import gp_minimize
    from skopt.space   import Real
    from skopt.utils   import use_named_args

    param_keys  = list(model.bounds.keys())
    space       = [Real(float(model.bounds[k][0]), float(model.bounds[k][1]),
                        name=k) for k in param_keys]
    eval_log:    List[Dict]  = []
    convergence: List[float] = []

    @use_named_args(space)
    def objective(**params):
        rms = _evaluate(dict(params), model, signals, t_eval, log_cb,
                        f"eval={len(eval_log)+1}")
        val  = rms["rms_total"]
        eval_log.append({**params, **rms})
        best = min((e["rms_total"] for e in eval_log), default=val)
        convergence.append(best)
        log_cb(f"  eval {len(eval_log):>4d} | total={val:.4f} | best={best:.4f}")
        return float(val)

    log_cb(f"Bayesian | n_calls={n_calls} n_initial={n_initial} | params={param_keys}")
    t0 = time.time()

    result = gp_minimize(
        objective, space,
        n_calls=n_calls, n_initial_points=n_initial,
        acq_func="EI", noise=1e-10, random_state=42, verbose=False,
    )

    wall = time.time() - t0
    log_cb(f"Bayesian done in {wall:.1f}s | evals={len(eval_log)}")

    best_params = dict(zip(param_keys, result.x))
    best_rms    = compute_cabin_rms(
        run_one_case(best_params, model, signals, t_eval), model
    )
    df = pd.DataFrame([{**best_params, **best_rms, "label": "Best_total"}])

    return OptResult("bayesian", df, eval_log, [], convergence, wall, len(eval_log), param_keys)


# ===========================================================================
# Public entry point
# ===========================================================================
def run_optimisation(
    model:        UserModel,
    signals:      Dict,
    t_eval:       np.ndarray,
    mode:         str = "nsga2",
    opt_settings: Optional[Dict] = None,
    log_cb:       Optional[Callable] = None,
) -> OptResult:
    if log_cb is None:
        log_cb = print
    if opt_settings is None:
        opt_settings = {}

    if mode == "nsga2":
        return _run_nsga2(
            model, signals, t_eval,
            pop_size=int(opt_settings.get("pop_size", 10)),
            n_gen=int(opt_settings.get("n_gen", 5)),
            log_cb=log_cb,
        )
    elif mode == "bayesian":
        return _run_bayesian(
            model, signals, t_eval,
            n_calls=int(opt_settings.get("n_calls", 50)),
            n_initial=int(opt_settings.get("n_initial", 10)),
            log_cb=log_cb,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'nsga2' or 'bayesian'.")
