"""
fno_core.py
-----------
Python port of the OFFICIAL MATLAB implementation of FNO:
    https://github.com/AhmadTaheri2021/FNO-Optimizer
    Files: FNO_final.m, Run_FNO.m  (fetched and read directly, 2026-09-10)

This supersedes an earlier version of this file that was based solely on the
printed equations/pseudocode in the paper PDF (Taheri et al. 2026, Artificial
Intelligence Review 59:79). Reading the actual source code revealed several
places where the *code* differs from what's printed in the paper -- the code
is authoritative here since it's what produced the paper's reported results.
Every deviation from the printed paper is called out below, matched line by
line against FNO_final.m.

DEVIATIONS FROM THE PRINTED PAPER (paper text/Eq. -> actual code behavior):

1. Farthest Better (FB) candidate pool is NOT "all Xj with f(Xj) < f(Xi)"
   (Eq. 1 as printed). The code first ranks the population by cost, splits
   into a "Betters" pool (better-ranked individuals) and "Worses" pool, and
   then RANDOMLY TRUNCATES the Betters pool to its top-ranked subset before
   picking the farthest one:
       shrink   = (1 - sqrt(FEs/MaxFEs))     # -> 1 early, -> 0 late
       keep_n   = ceil(len(Betters) * shrink * rand())
       Betters  = Betters[:keep_n]           # best-ranked keep_n candidates
       FB_i     = argmax distance(i, j) for j in Betters (or self if empty)
   No such truncation is applied to the Worses/NW pool.
   Practical effect: early in the search FB can be drawn from a wide pool of
   better solutions; late in the search FB is drawn almost exclusively from
   the very best few individuals (or falls back to self), sharply increasing
   exploitation pressure near convergence. This nuance is invisible in the
   printed Eq. (1).

2. phi / S (Eq. 6): the paper prints `min(0, sign(FB-X)*sign(NW-X))`, but the
   code computes `max(0, sign(FB-X)*sign(NW-X))`. The `max` version is the
   one consistent with the paper's own prose ("if Xi is between FBi and NWi,
   phi=0; otherwise phi=1"), so this looks like a transcription error in the
   published equation, not in the code. We use `max`, matching the code.

3. W (Eq. 10): paper prints `2 + U(-1,+1) x U([1,2])`, one vector value per
   dimension. Code computes a single SCALAR per individual per iteration:
       W = (2 + unifrnd(-1, 1)) * randi([1, 2])
   i.e., "2 + noise" is computed first, THEN multiplied by a random choice of
   1 or 2 -- different grouping and different shape (scalar, not per-dim).

4. r in Eq. (7): paper says "r <- U(0,1)" (implying a scalar). Code draws
   `rand(VarSize)`, i.e., an independent random value per dimension.

5. Branch condition for jump-over-NW (Eq. 8) vs. explore-FB (Eq. 9-11): paper
   says jump happens "with probability alpha" (alpha = sqrt(1-FEs/MaxFEs)).
   Code instead branches on:
       if rand() < (1 - FEs/MaxFEs): jump      # linear decay, not sqrt
       else: explore around FB with DFS
   alpha (sqrt form) is still used, but only inside the DFF (Eq. 5) term --
   not as the branch probability.

6. alpha/DFF/W/r are recomputed for EVERY individual using the *live*,
   continuously-incrementing FEs counter (updated after each single
   evaluation within the sweep), not a single value fixed at the start of
   the outer iteration. FB/NW (and the Betters-pool truncation) ARE computed
   once per outer sweep, using the FEs value at the start of that sweep --
   so there's an intentional asymmetry between when FB/NW get updated vs.
   when alpha/DFF/W/r get updated.

7. Bound handling ("Clipping"): the code does NOT clamp out-of-bound
   coordinates to the boundary; instead, each out-of-bound coordinate is
   REPLACED with a fresh uniform-random value in [LB, UB]:
       x[violating_dims] = LB + U(0,1) * (UB - LB)
   This is a materially different repair strategy from np.clip and is
   reproduced exactly here.

8. Evaluation budget bookkeeping: in Run_FNO.m / FNO_final.m, the initial
   population is evaluated OUTSIDE the FEs budget (FEs starts at 1, already
   pointing at the initial best), so total evaluations = nPop (init) +
   MaxFEs (search loop). This port instead counts the initial population
   evaluations AS PART OF `max_fes` (total evaluations == max_fes), matching
   the more common convention in the wrapper-FS literature and simplifying
   budget accounting for our (expensive, CV-based) SDP fitness. Documented
   here since it offsets FEs by nPop relative to the official code when
   comparing convergence curves at matched FEs values.

Convention: minimization, matching the paper's benchmark suite and case
studies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
from scipy.spatial.distance import cdist


@dataclass
class FNOResult:
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: np.ndarray  # best fitness recorded at every single FE
    n_evaluations: int


class FNO:
    """Farthest Better or Nearest Worse Optimizer -- ported from FNO_final.m.

    Parameters
    ----------
    obj_func : Callable[[np.ndarray], float]
        Objective function; receives a 1-D vector of length `dim`, returns a
        scalar fitness (lower is better).
    dim : int
        Problem dimensionality (N / nVar in the code).
    lb, ub : float or np.ndarray
        Lower/upper bounds (scalar broadcast or per-dimension array).
    n_pop : int
        Population size (nPop).
    max_fes : int
        Total evaluation budget, INCLUDING the initial population (see
        deviation #8 above regarding the official code's slightly different
        convention).
    seed : int, optional
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        lb,
        ub,
        n_pop: int = 30,
        max_fes: int = 10000,
        seed: Optional[int] = None,
    ):
        self.obj_func = obj_func
        self.dim = dim
        self.n_pop = n_pop
        self.max_fes = max_fes
        self.rng = np.random.default_rng(seed)

        self.lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    # ------------------------------------------------------------------ #
    def _init_population(self):
        # Run_FNO.m: Population(i).Position = LB + rand(1,N).*(UB-LB)
        pop = self.lb + self.rng.random((self.n_pop, self.dim)) * (self.ub - self.lb)
        fitness = np.array([self.obj_func(ind) for ind in pop])
        return pop, fitness

    def _repair_bounds(self, x: np.ndarray) -> np.ndarray:
        # FNO_final.m Clipping(): out-of-bound dims get a FRESH random value
        # in [LB, UB], not clamped to the boundary.
        x = x.copy()
        under = x < self.lb
        over = x > self.ub
        if np.any(under):
            x[under] = self.lb[under] + self.rng.random(np.sum(under)) * (self.ub[under] - self.lb[under])
        if np.any(over):
            x[over] = self.lb[over] + self.rng.random(np.sum(over)) * (self.ub[over] - self.lb[over])
        return x

    def _find_fb_nw(self, pop: np.ndarray, fitness: np.ndarray, fes_at_sweep_start: int):
        """FB/NW per FNO_final.m's rank-based logic with random truncation
        of the Betters pool. Computed once per outer sweep."""
        n = self.n_pop
        dist = cdist(pop, pop, metric="euclidean")
        sorted_idx = np.argsort(fitness, kind="stable")  # rank 1 (best) first
        rank_of = np.empty(n, dtype=int)
        rank_of[sorted_idx] = np.arange(1, n + 1)  # 1-indexed rank, matches MATLAB

        fb_idx = np.arange(n)  # sentinel default: FB(i) = i
        nw_idx = np.arange(n)  # sentinel default: NW(i) = i

        shrink = max(0.0, 1.0 - np.sqrt(fes_at_sweep_start / self.max_fes))

        for i in range(n):
            r = rank_of[i]  # 1-indexed
            if r == 1:
                betters = sorted_idx[0:1]          # just [i] itself (self)
                worses = sorted_idx[1:]
            elif r == n:
                betters = sorted_idx[0 : n - 1]
                worses = sorted_idx[n - 1 : n]      # just [i] itself (self)
            else:
                betters = sorted_idx[0 : r - 1]
                worses = sorted_idx[r:]

            # Random truncation of the Betters pool (keeps best-ranked subset)
            keep_n = int(np.ceil(len(betters) * shrink * self.rng.random()))
            betters = betters[:keep_n]

            if len(betters) > 0:
                cand = betters[betters != i]
                if len(cand) == 0:
                    cand = betters  # degenerate self-only pool -> FB=self
                fb_idx[i] = cand[np.argmax(dist[i, cand])]

            if len(worses) > 0:
                cand = worses[worses != i]
                if len(cand) == 0:
                    cand = worses  # degenerate self-only pool -> NW=self
                nw_idx[i] = cand[np.argmin(dist[i, cand])]

        return fb_idx, nw_idx

    # ------------------------------------------------------------------ #
    def optimize(self) -> FNOResult:
        pop, fitness = self._init_population()
        fes = self.n_pop  # see deviation #8: init counts toward the budget here

        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        curve = [best_fit] * fes  # one entry per evaluation, incl. init sweep

        while fes < self.max_fes:
            fb_idx, nw_idx = self._find_fb_nw(pop, fitness, fes_at_sweep_start=fes)

            for i in range(self.n_pop):
                if fes >= self.max_fes:
                    break

                Xi = pop[i]
                FBi = pop[fb_idx[i]]
                NWi = pop[nw_idx[i]]

                # alpha recomputed with LIVE fes (FNO_final.m: inside inner loop)
                alpha = np.sqrt(max(0.0, 1.0 - fes / self.max_fes))

                mu = self.rng.random()
                dff = np.abs(self.rng.uniform(mu - alpha, mu + alpha, size=self.dim))

                # W: SCALAR, (2 + U(-1,1)) * randi([1,2])
                W = (2.0 + self.rng.uniform(-1.0, 1.0)) * self.rng.choice([1.0, 2.0])

                # r: VECTOR, one draw per dimension (Eq. 7 / Pos_temp)
                r_vec = self.rng.random(self.dim)

                # phi = max(0, sign(FB-X) * sign(NW-X))  [code uses max, not min]
                phi = np.maximum(0.0, np.sign(FBi - Xi) * np.sign(NWi - Xi))

                Pos_temp = (
                    phi * NWi
                    + (1.0 - phi) * Xi
                    + np.sign(FBi - Xi) * r_vec * np.abs(Xi - NWi)
                )

                # Branch on (1 - FEs/MaxFEs), using LIVE fes -- NOT alpha
                if self.rng.random() < (1.0 - fes / self.max_fes):
                    x_new = Pos_temp  # jump over / escape the NW region
                else:
                    Exp = dff * (FBi - Pos_temp)
                    x_new = Pos_temp + W * Exp  # explore around FB via DFS

                x_new = self._repair_bounds(x_new)

                f_new = self.obj_func(x_new)
                fes += 1

                if f_new < fitness[i]:
                    pop[i] = x_new
                    fitness[i] = f_new
                    if f_new < best_fit:
                        best_fit = f_new
                        best_pos = x_new.copy()

                curve.append(best_fit)

        return FNOResult(
            best_position=best_pos,
            best_fitness=best_fit,
            convergence_curve=np.array(curve[: self.max_fes]),
            n_evaluations=fes,
        )