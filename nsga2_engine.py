"""
nsga2_engine.py
===============
Self-contained NSGA-II implementation in pure NumPy.
No pymoo, no torch, no external dependencies beyond numpy.

Implements:
  - Fast non-dominated sorting  (Deb et al. 2002)
  - Crowding distance assignment
  - Tournament selection  (binary, crowded-comparison operator)
  - Simulated Binary Crossover  SBX  (eta_c = 15)
  - Polynomial Mutation  PM   (eta_m = 20)
  - Hypervolume indicator  (exact, 3-objective, WFG sweep)
  - NSGA2 class  – drop-in for the parts of pymoo used in the vehicle codes

Usage:
    from nsga2_engine import NSGA2, HV, NonDominatedSorting

All public symbols replicate the pymoo interface used in the two vehicle files
so that swapping back to pymoo later requires only changing the import line.
"""

import numpy as np
from typing import Callable, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# NON-DOMINATED SORTING
# ═══════════════════════════════════════════════════════════════════════════

class NonDominatedSorting:
    """
    Fast non-dominated sorting  (Deb 2002, Algorithm 1).
    Returns a list of fronts; each front is an array of indices into F.
    """

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """True if a dominates b: a ≤ b on all objectives, a < b on at least one."""
        return bool(np.all(a <= b) and np.any(a < b))

    def do(self, F: np.ndarray,
           only_non_dominated_front: bool = False) -> list:
        """
        Parameters
        ----------
        F : (N, M) array of objective values (minimisation assumed)
        only_non_dominated_front : if True, return only front-0 as a flat array

        Returns
        -------
        If only_non_dominated_front=True  →  1-D array of Pareto indices
        Else                              →  list of 1-D arrays (one per front)
        """
        N = len(F)
        dom_count  = np.zeros(N, dtype=int)    # how many solutions dominate i
        dom_set    = [[] for _ in range(N)]    # solutions that i dominates

        for i in range(N):
            for j in range(i + 1, N):
                if self._dominates(F[i], F[j]):
                    dom_set[i].append(j)
                    dom_count[j] += 1
                elif self._dominates(F[j], F[i]):
                    dom_set[j].append(i)
                    dom_count[i] += 1

        fronts  = []
        current = list(np.where(dom_count == 0)[0])

        while current:
            fronts.append(np.array(current, dtype=int))
            next_front = []
            for i in current:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        next_front.append(j)
            current = next_front

        if only_non_dominated_front:
            return fronts[0] if fronts else np.array([], dtype=int)
        return fronts


# ═══════════════════════════════════════════════════════════════════════════
# CROWDING DISTANCE
# ═══════════════════════════════════════════════════════════════════════════

def crowding_distance(F: np.ndarray) -> np.ndarray:
    """
    Crowding distance for a set of solutions F (already on the same front).
    F : (n, m) — n solutions, m objectives.
    Returns (n,) distance array; boundary solutions get inf.
    """
    n, m = F.shape
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf
        return dist

    for k in range(m):
        order = np.argsort(F[:, k])
        f_min = F[order[0],  k]
        f_max = F[order[-1], k]
        dist[order[0]]  = np.inf
        dist[order[-1]] = np.inf
        span = f_max - f_min
        if span < 1e-12:
            continue
        for idx in range(1, n - 1):
            dist[order[idx]] += (F[order[idx + 1], k] - F[order[idx - 1], k]) / span

    return dist


# ═══════════════════════════════════════════════════════════════════════════
# HYPERVOLUME  (exact, 3-objective, WFG algorithm)
# ═══════════════════════════════════════════════════════════════════════════

class HV:
    """
    Exact hypervolume indicator for 2- or 3-objective minimisation problems.
    Uses the WFG recursive algorithm (While, Lyndon & While 2006).

    Parameters
    ----------
    ref_point : array-like, shape (m,)
        Reference point (must be dominated by all Pareto solutions).
        Typically set slightly above the worst expected objective values.
    """

    def __init__(self, ref_point: np.ndarray):
        self.ref_point = np.asarray(ref_point, dtype=float)

    def compute(self, F: np.ndarray) -> float:
        """
        Compute hypervolume of point set F w.r.t. self.ref_point.
        F : (n, m) — minimisation; points must dominate the reference point.
        Returns scalar hypervolume value.
        """
        F   = np.asarray(F, dtype=float)
        ref = self.ref_point

        # Filter: keep only points that dominate the reference point
        mask = np.all(F < ref, axis=1)
        F    = F[mask]
        if len(F) == 0:
            return 0.0

        return self._wfg(F, ref)

    # alias used by HVCallback: hv_indicator(F_pareto)
    def __call__(self, F: np.ndarray) -> float:
        return self.compute(F)

    # ── WFG recursive implementation ─────────────────────────────────────

    def _wfg(self, F: np.ndarray, ref: np.ndarray) -> float:
        """Recursive WFG hypervolume computation."""
        n, m = F.shape
        if n == 0:
            return 0.0
        if m == 1:
            return float(ref[0] - np.min(F[:, 0]))
        if m == 2:
            return self._hv2d(F, ref)

        # Sort by last objective (descending so we process in sweep order)
        order = np.argsort(F[:, -1])
        F_sorted = F[order]

        hv = 0.0
        for i in range(n):
            # Slice in the last dimension: contribution between F_sorted[i,-1]
            # and either F_sorted[i-1,-1] or ref[-1]
            z_lo = F_sorted[i, -1]
            z_hi = ref[-1] if i == 0 else F_sorted[i - 1, -1]
            if z_hi <= z_lo:
                continue
            # Compute (m-1)-dimensional HV of the points up to and including i
            # in the remaining dimensions
            pts_sub = F_sorted[:i + 1, :-1].copy()
            ref_sub = ref[:-1].copy()
            hv += self._wfg(pts_sub, ref_sub) * (z_hi - z_lo)

        return hv

    @staticmethod
    def _hv2d(F: np.ndarray, ref: np.ndarray) -> float:
        """Exact 2-D hypervolume by sweep line."""
        order = np.argsort(F[:, 0])
        F_s   = F[order]
        hv    = 0.0
        prev_y = ref[1]
        for i in range(len(F_s)):
            width  = ref[0] - F_s[i, 0]
            height = prev_y - F_s[i, 1]
            if width > 0 and height > 0:
                hv    += width * height
            prev_y = min(prev_y, F_s[i, 1])
        return hv


# ═══════════════════════════════════════════════════════════════════════════
# SBX CROSSOVER
# ═══════════════════════════════════════════════════════════════════════════

def sbx_crossover(parent1: np.ndarray, parent2: np.ndarray,
                  xl: np.ndarray, xu: np.ndarray,
                  eta_c: float = 15.0,
                  prob_c: float = 0.9,
                  rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX) for real-valued variables.
    Produces two offspring from two parents.

    Parameters
    ----------
    parent1, parent2 : (n_var,) arrays
    xl, xu           : lower / upper bounds
    eta_c            : distribution index (larger = offspring closer to parents)
    prob_c           : crossover probability per variable
    """
    if rng is None:
        rng = np.random.default_rng()

    n   = len(parent1)
    c1  = parent1.copy()
    c2  = parent2.copy()

    for i in range(n):
        if rng.random() > prob_c:
            continue

        x1, x2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])

        if x2 - x1 < 1e-12:
            continue

        # Spread factor β
        u   = rng.random()
        bl  = 1.0 + 2.0 * (x1 - xl[i]) / (x2 - x1)
        bu  = 1.0 + 2.0 * (xu[i] - x2) / (x2 - x1)

        # β_l
        if u <= 0.5:
            beta_l = (2.0 * u + (1.0 - 2.0 * u) * (1.0 / bl) ** (eta_c + 1.0)) ** (1.0 / (eta_c + 1.0))
        else:
            beta_l = (1.0 / (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 / bl) ** (eta_c + 1.0))) ** (1.0 / (eta_c + 1.0))

        # β_u
        u2 = rng.random()
        if u2 <= 0.5:
            beta_u = (2.0 * u2 + (1.0 - 2.0 * u2) * (1.0 / bu) ** (eta_c + 1.0)) ** (1.0 / (eta_c + 1.0))
        else:
            beta_u = (1.0 / (2.0 * (1.0 - u2) + 2.0 * (u2 - 0.5) * (1.0 / bu) ** (eta_c + 1.0))) ** (1.0 / (eta_c + 1.0))

        c1[i] = 0.5 * ((x1 + x2) - beta_l * (x2 - x1))
        c2[i] = 0.5 * ((x1 + x2) + beta_u * (x2 - x1))

        # Swap randomly (SBX symmetry)
        if rng.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]

        # Clip to bounds
        c1[i] = float(np.clip(c1[i], xl[i], xu[i]))
        c2[i] = float(np.clip(c2[i], xl[i], xu[i]))

    return c1, c2


# ═══════════════════════════════════════════════════════════════════════════
# POLYNOMIAL MUTATION
# ═══════════════════════════════════════════════════════════════════════════

def polynomial_mutation(x: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                         eta_m: float = 20.0,
                         prob_m: Optional[float] = None,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Polynomial mutation (PM) for real-valued variables.

    Parameters
    ----------
    x      : (n_var,) individual to mutate
    xl, xu : bounds
    eta_m  : distribution index (larger = smaller perturbations)
    prob_m : per-variable mutation probability (default = 1/n_var)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    if prob_m is None:
        prob_m = 1.0 / n

    y = x.copy()
    for i in range(n):
        if rng.random() > prob_m:
            continue

        delta1 = (x[i] - xl[i]) / (xu[i] - xl[i] + 1e-12)
        delta2 = (xu[i] - x[i]) / (xu[i] - xl[i] + 1e-12)

        u  = rng.random()
        if u < 0.5:
            val      = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (eta_m + 1.0)
            delta_q  = val ** (1.0 / (eta_m + 1.0)) - 1.0
        else:
            val      = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2) ** (eta_m + 1.0)
            delta_q  = 1.0 - val ** (1.0 / (eta_m + 1.0))

        y[i] = float(np.clip(x[i] + delta_q * (xu[i] - xl[i]), xl[i], xu[i]))

    return y


# ═══════════════════════════════════════════════════════════════════════════
# TOURNAMENT SELECTION  (binary, crowded-comparison)
# ═══════════════════════════════════════════════════════════════════════════

def binary_tournament(rank: np.ndarray, crowd: np.ndarray,
                      rng: np.random.Generator) -> int:
    """
    Crowded-comparison tournament between two randomly chosen individuals.
    Lower rank wins; ties broken by larger crowding distance.
    Returns index of winner.
    """
    n   = len(rank)
    i   = rng.integers(0, n)
    j   = rng.integers(0, n)
    while j == i:
        j = rng.integers(0, n)

    if rank[i] < rank[j]:
        return i
    elif rank[j] < rank[i]:
        return j
    else:
        return i if crowd[i] >= crowd[j] else j


# ═══════════════════════════════════════════════════════════════════════════
# NSGA-II MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════

class NSGA2:
    """
    NSGA-II optimiser — pure NumPy implementation.

    Interface mirrors the pymoo usage in the vehicle files:

        algorithm = NSGA2(pop_size=10, xl=XL, xu=XU, seed=42)
        result    = algorithm.run(evaluate_fn, n_gen=5, callback=cb)

    Parameters
    ----------
    pop_size : int
        Population size (must be even for crossover pairing).
    xl, xu   : np.ndarray  (n_var,)
        Lower / upper bounds for each variable.
    eta_c    : float  SBX distribution index      (default 15)
    eta_m    : float  PM  distribution index       (default 20)
    prob_c   : float  crossover probability        (default 0.9)
    seed     : int    random seed for reproducibility
    """

    def __init__(self,
                 pop_size: int,
                 xl: np.ndarray,
                 xu: np.ndarray,
                 eta_c: float = 15.0,
                 eta_m: float = 20.0,
                 prob_c: float = 0.9,
                 seed: int = 42):

        if pop_size % 2 != 0:
            pop_size += 1   # SBX needs pairs

        self.pop_size = pop_size
        self.xl       = np.asarray(xl, dtype=float)
        self.xu       = np.asarray(xu, dtype=float)
        self.eta_c    = eta_c
        self.eta_m    = eta_m
        self.prob_c   = prob_c
        self.rng      = np.random.default_rng(seed)
        self.n_var    = len(xl)

        # Public attributes populated during run()
        self.n_gen      = 0
        self.pop_X      = None   # current population parameters  (pop_size, n_var)
        self.pop_F      = None   # current population objectives   (pop_size, n_obj)
        self._nds       = NonDominatedSorting()

    # ── public result container (mirrors pymoo Result) ────────────────────

    class _Result:
        def __init__(self, X, F):
            self.X = X   # (n_pareto, n_var)
            self.F = F   # (n_pareto, n_obj)

    # ── main entry point ──────────────────────────────────────────────────

    def run(self,
            evaluate_fn: Callable[[np.ndarray], np.ndarray],
            n_gen: int,
            callback=None) -> "_Result":
        """
        Run NSGA-II.

        Parameters
        ----------
        evaluate_fn : callable
            Takes a single individual x (n_var,) and returns objectives (n_obj,).
            Should handle failures gracefully (return penalty values).
        n_gen       : int
            Number of generations to run.
        callback    : optional callable
            Called at the end of each generation with this NSGA2 instance.
            Receives self so it can read self.n_gen, self.pop_F, etc.

        Returns
        -------
        _Result with .X (Pareto params) and .F (Pareto objectives)
        """

        # ── initialise population ─────────────────────────────────────────
        print("  [NSGA-II] Initialising population...")
        self.pop_X = self._init_population()
        self.pop_F = self._evaluate_population(self.pop_X, evaluate_fn)

        # ── generational loop ─────────────────────────────────────────────
        for gen in range(1, n_gen + 1):
            self.n_gen = gen
            print(f"\n  ── Generation {gen}/{n_gen} ──")

            # Rank + crowding on current population
            rank, crowd = self._rank_and_crowd(self.pop_F)

            # Generate offspring population (same size as parent)
            off_X = self._make_offspring(self.pop_X, rank, crowd)
            off_F = self._evaluate_population(off_X, evaluate_fn)

            # Merge parent + offspring → size 2*pop_size
            merged_X = np.vstack([self.pop_X, off_X])
            merged_F = np.vstack([self.pop_F, off_F])

            # Select next generation via crowded non-dominated sort
            self.pop_X, self.pop_F = self._environmental_selection(merged_X, merged_F)

            # Callback
            if callback is not None:
                callback(self)

        # ── extract final Pareto front ────────────────────────────────────
        fronts    = self._nds.do(self.pop_F)
        pareto_idx = fronts[0]
        return self._Result(self.pop_X[pareto_idx].copy(),
                            self.pop_F[pareto_idx].copy())

    # ── internal helpers ──────────────────────────────────────────────────

    def _init_population(self) -> np.ndarray:
        """Latin Hypercube Sampling for initial population (better coverage than pure random)."""
        n   = self.pop_size
        d   = self.n_var
        lhs = np.zeros((n, d))
        for j in range(d):
            perm       = self.rng.permutation(n)
            lhs[:, j]  = (perm + self.rng.random(n)) / n
        return self.xl + lhs * (self.xu - self.xl)

    def _evaluate_population(self, X: np.ndarray,
                              evaluate_fn: Callable) -> np.ndarray:
        """Evaluate all individuals; collect results into (N, n_obj) matrix."""
        results = []
        for i, x in enumerate(X):
            print(f"    Evaluating individual {i+1}/{len(X)}...")
            f = evaluate_fn(x)
            results.append(np.asarray(f, dtype=float))
        return np.array(results)

    def _rank_and_crowd(self, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assign Pareto rank and crowding distance to every individual."""
        n     = len(F)
        rank  = np.zeros(n, dtype=int)
        crowd = np.zeros(n, dtype=float)

        fronts = self._nds.do(F)
        for r, front in enumerate(fronts):
            rank[front]  = r
            crowd[front] = crowding_distance(F[front])

        return rank, crowd

    def _make_offspring(self, X: np.ndarray,
                        rank: np.ndarray,
                        crowd: np.ndarray) -> np.ndarray:
        """
        Create offspring population via tournament selection → SBX → PM.
        Produces exactly pop_size offspring.
        """
        offspring = []
        while len(offspring) < self.pop_size:
            p1_idx = binary_tournament(rank, crowd, self.rng)
            p2_idx = binary_tournament(rank, crowd, self.rng)
            while p2_idx == p1_idx:
                p2_idx = binary_tournament(rank, crowd, self.rng)

            c1, c2 = sbx_crossover(
                X[p1_idx], X[p2_idx],
                self.xl, self.xu,
                eta_c=self.eta_c,
                prob_c=self.prob_c,
                rng=self.rng,
            )
            c1 = polynomial_mutation(c1, self.xl, self.xu, eta_m=self.eta_m, rng=self.rng)
            c2 = polynomial_mutation(c2, self.xl, self.xu, eta_m=self.eta_m, rng=self.rng)

            offspring.append(c1)
            if len(offspring) < self.pop_size:
                offspring.append(c2)

        return np.array(offspring[:self.pop_size])

    def _environmental_selection(self, X: np.ndarray,
                                  F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select pop_size individuals from 2*pop_size merged population.
        Fill by front; last front trimmed by crowding distance.
        """
        fronts = self._nds.do(F)
        selected = []

        for front in fronts:
            if len(selected) + len(front) <= self.pop_size:
                selected.extend(front.tolist())
            else:
                # Partial front: sort by crowding distance (descending)
                needed   = self.pop_size - len(selected)
                cd       = crowding_distance(F[front])
                order    = np.argsort(cd)[::-1]
                selected.extend(front[order[:needed]].tolist())
                break

        idx = np.array(selected, dtype=int)
        return X[idx].copy(), F[idx].copy()
