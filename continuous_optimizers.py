"""
continuous_optimizers.py
-------------------------
PSO, DE, and CDO, implemented with the SAME constructor/optimize()
interface as fno_core.FNO (obj_func, dim, lb, ub, n_pop, max_fes, seed ->
.optimize() returning an object with .best_position, .best_fitness,
.convergence_curve, .n_evaluations). This lets binary_wrapper.
BinaryWrapperOptimizer bind to any of FNO/PSO/DE/CDO interchangeably for
wrapper-based binary feature selection, without duplicating the
S-shape/V-shape binarization logic per algorithm.

PSO and DE are standard textbook formulations (not tuned/modified for this
project), using widely-cited default hyperparameters so the comparison
against FNO is against "vanilla" baselines, consistent with how the FNO
paper itself benchmarks against standard PSO/DE configurations (Sect. 3.3):

- PSO: Kennedy & Eberhart (1995), with linearly-decreasing inertia weight
  per Shi & Eberhart (1998) -- w: 0.9 -> 0.4 over the run, c1=c2=2.0.
- DE: Storn & Price (1997), DE/rand/1/bin variant -- F=0.5, CR=0.9 (the
  most commonly cited default combination in the DE literature).

CDO is a direct port of the official author source code (see the CDO
class docstring for details and documented adaptations), included as a
"recent metaheuristic" comparator with a confirmed precedent in prior
SDP feature-selection literature (Anand, Jena & Das 2024, F1000Research
13:844, DOI:10.12688/f1000research.150927.2 -- CDO-based FS on 12
PROMISE/NASA datasets with DT/KNN/NB/QDA classifiers, closely matching
this study's own setup):
- CDO: Shehadeh (2023), Neural Computing and Applications 35:10733-10749,
  DOI:10.1007/s00521-023-08261-1.

Convention: minimization.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class OptResult:
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: np.ndarray
    n_evaluations: int


class PSO:
    """Particle Swarm Optimization (Kennedy & Eberhart 1995), linearly
    decreasing inertia weight (Shi & Eberhart 1998)."""

    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        lb,
        ub,
        n_pop: int = 30,
        max_fes: int = 10000,
        seed: Optional[int] = None,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
    ):
        self.obj_func = obj_func
        self.dim = dim
        self.n_pop = n_pop
        self.max_fes = max_fes
        self.rng = np.random.default_rng(seed)
        self.w_max, self.w_min, self.c1, self.c2 = w_max, w_min, c1, c2

        self.lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    def optimize(self) -> OptResult:
        pos = self.lb + self.rng.random((self.n_pop, self.dim)) * (self.ub - self.lb)
        vmax = 0.2 * (self.ub - self.lb)  # common heuristic velocity clamp
        vel = self.rng.uniform(-1, 1, (self.n_pop, self.dim)) * vmax

        fitness = np.array([self.obj_func(p) for p in pos])
        fes = self.n_pop

        pbest_pos = pos.copy()
        pbest_fit = fitness.copy()
        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        curve = [gbest_fit] * fes
        max_iter = max(1, (self.max_fes - self.n_pop) // self.n_pop)
        it = 0

        while fes < self.max_fes:
            w = self.w_max - (self.w_max - self.w_min) * (it / max(1, max_iter))
            for i in range(self.n_pop):
                if fes >= self.max_fes:
                    break
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                vel[i] = (
                    w * vel[i]
                    + self.c1 * r1 * (pbest_pos[i] - pos[i])
                    + self.c2 * r2 * (gbest_pos - pos[i])
                )
                vel[i] = np.clip(vel[i], -vmax, vmax)
                pos[i] = np.clip(pos[i] + vel[i], self.lb, self.ub)

                f = self.obj_func(pos[i])
                fes += 1

                if f < pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest_pos[i] = pos[i].copy()
                    if f < gbest_fit:
                        gbest_fit = f
                        gbest_pos = pos[i].copy()
                curve.append(gbest_fit)
            it += 1

        return OptResult(
            best_position=gbest_pos,
            best_fitness=gbest_fit,
            convergence_curve=np.array(curve[: self.max_fes]),
            n_evaluations=fes,
        )


class CDO:
    """Chernobyl Disaster Optimizer (Shehadeh 2023, Neural Computing and
    Applications 35:10733-10749, DOI:10.1007/s00521-023-08261-1).

    Ported directly from the author's official MATLAB source
    (github.com/sh7adeh1990/CDO/CDO.m, fetched 2026-09-10; kept alongside
    fno_matlab_reference/ for audit) rather than from the paper's prose,
    following the same practice used for FNO -- code is authoritative.

    Structurally this is a GWO-family algorithm: three "particles" (alpha,
    beta, gamma -- by analogy to GWO's alpha/beta/delta wolves) track the
    best, second-best, and third-best solutions found so far. Each
    dimension of every individual's position is recomputed EVERY iteration
    as the average of three pull-terms, one toward each of alpha/beta/gamma,
    using position-dependent coefficients built from log10 of large
    constants (16000, 270000, 300000 -- the original paper motivates these
    as inspired by physical decay/half-life-like quantities of the three
    radiation types; not derived further here, just reproduced exactly).

    IMPORTANT DEVIATION FROM THE ORIGINAL CODE: unlike FNO/PSO/DE in this
    package, the official CDO has NO per-individual greedy acceptance --
    every individual's position is unconditionally overwritten each
    iteration (only the alpha/beta/gamma global-best trackers persist).
    We reproduce this faithfully; it is not a bug, just a different search
    dynamic than the other three algorithms.

    ADAPTATIONS for interface/budget parity with FNO/PSO/DE in this package
    (documented, not present in the original, which runs a fixed
    `Max_iter` outer-loop count regardless of any live evaluation budget):
      - We track `fes` live and stop evaluating mid-sweep once max_fes is
        reached (the original always evaluates the full population every
        iteration), so all algorithms in this study use exactly the same
        total evaluation budget.
      - convergence_curve records the best-so-far (alpha_score) once PER
        EVALUATION (matching FNO/PSO/DE's convention), not once per outer
        iteration as in the original -- so curves are directly comparable
        at matched FEs across algorithms.
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

    def optimize(self) -> OptResult:
        pop = self.lb + self.rng.random((self.n_pop, self.dim)) * (self.ub - self.lb)

        alpha_pos = np.zeros(self.dim)
        alpha_score = np.inf
        beta_pos = np.zeros(self.dim)
        beta_score = np.inf
        gamma_pos = np.zeros(self.dim)
        gamma_score = np.inf

        fes = 0
        curve = []
        max_iter = max(1, self.max_fes // self.n_pop)  # used only to shape the `a` decay schedule
        it = 0

        # NOTE: stop condition is `fes < self.max_fes` ONLY -- not also
        # `it < max_iter`. An earlier version also capped on `it < max_iter`,
        # but max_iter = max_fes // n_pop truncates by integer division
        # (e.g. max_fes=10000, n_pop=30 -> max_iter=333 -> only 9990 evals
        # possible), which silently under-spent the evaluation budget by up
        # to n_pop-1 evaluations. Caught by cross-checking n_evaluations ==
        # max_fes across several benchmark functions/seeds before trusting
        # this as a baseline -- every algorithm in this study must exhaust
        # the SAME total budget for the comparison to be fair.
        while fes < self.max_fes:
            # --- evaluate population, update alpha/beta/gamma trackers ---
            for i in range(self.n_pop):
                if fes >= self.max_fes:
                    break
                pop[i] = np.clip(pop[i], self.lb, self.ub)
                fitness = self.obj_func(pop[i])
                fes += 1

                if fitness < alpha_score:
                    alpha_score, alpha_pos = fitness, pop[i].copy()
                if alpha_score < fitness < beta_score:
                    beta_score, beta_pos = fitness, pop[i].copy()
                if alpha_score < fitness and beta_score < fitness < gamma_score:
                    gamma_score, gamma_pos = fitness, pop[i].copy()
                curve.append(alpha_score)

            if fes >= self.max_fes:
                break

            # `a` decreases linearly 3 -> 0 over the run (Eq. 9 in the paper).
            # Clipped at 0 (rather than going negative) if the actual number
            # of outer sweeps needed to exhaust max_fes ends up exceeding the
            # `max_iter` estimate used to shape the schedule -- can happen
            # since max_iter = max_fes // n_pop is only an approximation
            # (floor division) of how many full sweeps fit in the budget.
            a = max(0.0, 3.0 - it * (3.0 / max_iter))
            a1 = np.log10((16000.0 - 1.0) * self.rng.random() + 16000.0)
            a2 = np.log10((270000.0 - 1.0) * self.rng.random() + 270000.0)
            a3 = np.log10((300000.0 - 1.0) * self.rng.random() + 300000.0)

            # --- update every individual's position (no greedy acceptance) ---
            for i in range(self.n_pop):
                for j in range(self.dim):
                    r1, r2 = self.rng.random(), self.rng.random()
                    pa = np.pi * r1 * r1 / (0.25 * a1) - a * self.rng.random()
                    C1 = r2 * r2 * np.pi
                    D_alpha = abs(C1 * alpha_pos[j] - pop[i, j])
                    va = 0.25 * (alpha_pos[j] - pa * D_alpha)

                    r1, r2 = self.rng.random(), self.rng.random()
                    pb = np.pi * r1 * r1 / (0.5 * a2) - a * self.rng.random()
                    C2 = r2 * r2 * np.pi
                    D_beta = abs(C2 * beta_pos[j] - pop[i, j])
                    vb = 0.5 * (beta_pos[j] - pb * D_beta)

                    r1, r2 = self.rng.random(), self.rng.random()
                    py = (np.pi * r1 * r1) / a3 - a * self.rng.random()
                    C3 = r2 * r2 * np.pi
                    D_gamma = abs(C3 * gamma_pos[j] - pop[i, j])
                    vy = gamma_pos[j] - py * D_gamma

                    pop[i, j] = (va + vb + vy) / 3.0

            it += 1

        return OptResult(
            best_position=alpha_pos,
            best_fitness=alpha_score,
            convergence_curve=np.array(curve[: self.max_fes]),
            n_evaluations=fes,
        )


class DE:
    """Differential Evolution, DE/rand/1/bin (Storn & Price 1997)."""

    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        lb,
        ub,
        n_pop: int = 30,
        max_fes: int = 10000,
        seed: Optional[int] = None,
        F: float = 0.5,
        CR: float = 0.9,
    ):
        self.obj_func = obj_func
        self.dim = dim
        self.n_pop = n_pop
        self.max_fes = max_fes
        self.rng = np.random.default_rng(seed)
        self.F, self.CR = F, CR

        self.lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    def optimize(self) -> OptResult:
        pop = self.lb + self.rng.random((self.n_pop, self.dim)) * (self.ub - self.lb)
        fitness = np.array([self.obj_func(ind) for ind in pop])
        fes = self.n_pop

        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        curve = [best_fit] * fes

        while fes < self.max_fes:
            for i in range(self.n_pop):
                if fes >= self.max_fes:
                    break
                candidates = [j for j in range(self.n_pop) if j != i]
                a, b, c = pop[self.rng.choice(candidates, size=3, replace=False)]

                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Binomial crossover, guaranteeing at least one mutant gene
                cross_mask = self.rng.random(self.dim) < self.CR
                j_rand = self.rng.integers(self.dim)
                cross_mask[j_rand] = True
                trial = np.where(cross_mask, mutant, pop[i])

                f_trial = self.obj_func(trial)
                fes += 1

                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fit:
                        best_fit = f_trial
                        best_pos = trial.copy()
                curve.append(best_fit)

        return OptResult(
            best_position=best_pos,
            best_fitness=best_fit,
            convergence_curve=np.array(curve[: self.max_fes]),
            n_evaluations=fes,
        )