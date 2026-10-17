"""
binary_wrapper.py
------------------
Generic wrapper-based binary feature selection: any continuous optimizer
implementing the shared interface (constructor: obj_func, dim, lb, ub,
n_pop, max_fes, seed; method: .optimize() -> object with .best_position,
.best_fitness, .convergence_curve, .n_evaluations) can be plugged in via
`optimizer_cls`. Used to binarize FNO, PSO, and DE identically, so the
comparison across algorithms isolates the *search strategy* and doesn't
also vary the binarization scheme.

This binarization scheme (continuous optimizer + transfer function) is a
design choice on our part -- standard practice in the wrapper-FS literature
(e.g., binary PSO/GA/GWO/DE), not something specific to any one of these
optimizers.

See fno_core.py, continuous_optimizers.py for the underlying algorithms,
and sdp_fitness.py for the fitness function typically passed in.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Literal, Type

TransferKind = Literal["s-shape", "v-shape", "v-shape-ind"]


def s_shape(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def v_shape(x: np.ndarray) -> np.ndarray:
    return np.abs(np.tanh(x))


@dataclass
class BinaryWrapperResult:
    best_mask: np.ndarray
    best_fitness: float
    n_selected: int
    convergence_curve: np.ndarray
    n_evaluations: int
    continuous_result: object  # the underlying optimizer's raw result, for diagnostics


class BinaryWrapperOptimizer:
    """Wrapper-based binary feature selection using `optimizer_cls` (FNO/PSO/DE
    from this package, or anything with a compatible interface) + a transfer
    function.

    Parameters
    ----------
    optimizer_cls : type
        A class with the FNO-compatible interface (see module docstring).
    fitness_func : Callable[[np.ndarray], float]
        Takes a boolean feature mask, returns a scalar to MINIMIZE.
    n_features : int
    n_pop, max_fes, seed : passed through to optimizer_cls.
    transfer : "s-shape" or "v-shape"
    x_bounds : bounds of the continuous search space (saturates sigmoid/tanh
        near the edges).
    min_selected : minimum number of features that must be selected (mask
        is repaired if fewer are selected).
    optimizer_kwargs : extra kwargs forwarded to optimizer_cls (e.g. PSO's
        w_max/w_min/c1/c2, DE's F/CR).
    """

    def __init__(
        self,
        optimizer_cls: Type,
        fitness_func: Callable[[np.ndarray], float],
        n_features: int,
        n_pop: int = 20,
        max_fes: int = 5000,
        seed: Optional[int] = None,
        transfer: TransferKind = "s-shape",
        x_bounds=(-4.0, 4.0),
        min_selected: int = 1,
        optimizer_kwargs: Optional[dict] = None,
    ):
        self.optimizer_cls = optimizer_cls
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.n_pop = n_pop
        self.max_fes = max_fes
        self.seed = seed
        self.transfer = transfer
        self.x_bounds = x_bounds
        self.min_selected = min_selected
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.rng = np.random.default_rng(seed)

        self._transfer_fn = s_shape if transfer == "s-shape" else v_shape
        # v-shape-ind: per-individual state. All of FNO, PSO and DE here evaluate
        # individuals in a fixed order (slot = evaluation index mod n_pop), the
        # initial population first. DE and FNO replace an individual only if the
        # candidate is strictly better (so the slot reference is updated only then);
        # PSO always moves the particle (reference always updated).
        self._greedy = getattr(optimizer_cls, "__name__", "") in ("FNO", "DE")
        if transfer == "v-shape-ind" and getattr(optimizer_cls, "__name__", "") not in ("FNO", "PSO", "DE"):
            raise ValueError("v-shape-ind assumes fixed per-individual evaluation order (FNO, PSO, DE only)")
        self._n_eval = 0
        self._slot_x = {}
        self._slot_mask = {}
        self._slot_fit = {}
        self._prev_x = None
        self._prev_mask = None

        # Track the (mask, fitness) pair actually evaluated during search --
        # see note in optimize(): re-binarizing post-hoc would be stochastic
        # and could report a mask that was never actually evaluated as best.
        self._best_mask = None
        self._best_fit = np.inf

    def _binarize(self, x: np.ndarray) -> np.ndarray:
        if self.transfer == "v-shape-ind":
            slot = self._n_eval % self.n_pop
            if slot not in self._slot_x:  # initial population: random mask
                mask = self.rng.random(self.n_features) < 0.5
            else:
                prob = self._transfer_fn(x - self._slot_x[slot])
                flip = self.rng.random(self.n_features) < prob
                prev = self._slot_mask[slot]
                mask = np.where(flip, ~prev, prev)
        elif self.transfer == "s-shape":
            prob = self._transfer_fn(x)
            mask = self.rng.random(self.n_features) < prob
        else:  # v-shape acts on the update delta
            delta = x if self._prev_x is None else x - self._prev_x
            prob = self._transfer_fn(delta)
            flip = self.rng.random(self.n_features) < prob
            if self._prev_mask is None:
                mask = self.rng.random(self.n_features) < 0.5
            else:
                mask = np.where(flip, ~self._prev_mask, self._prev_mask)
            self._prev_mask = mask

        if mask.sum() < self.min_selected:
            zero_idx = np.where(~mask)[0]
            need = self.min_selected - int(mask.sum())
            pick = self.rng.choice(zero_idx, size=min(need, len(zero_idx)), replace=False)
            mask[pick] = True

        return mask

    def _wrapped_objective(self, x: np.ndarray) -> float:
        mask = self._binarize(x)
        self._prev_x = x
        fit = self.fitness_func(mask)
        if self.transfer == "v-shape-ind":
            slot = self._n_eval % self.n_pop
            accept = (not self._greedy) or slot not in self._slot_fit or fit < self._slot_fit[slot]
            if accept:
                self._slot_x[slot] = np.array(x, dtype=float, copy=True)
                self._slot_mask[slot] = mask.copy()
                self._slot_fit[slot] = fit
            self._n_eval += 1
        if fit < self._best_fit:
            self._best_fit = fit
            self._best_mask = mask.copy()
        return fit

    def optimize(self) -> BinaryWrapperResult:
        # Defensive reset: if this object's .optimize() is ever called more
        # than once, stale state from a prior call must not leak in. (Not
        # currently triggered by this package's usage pattern -- run_one()
        # always constructs a fresh instance per run -- but cheap to make
        # safe regardless.)
        self._prev_mask = None
        self._prev_x = None
        self._n_eval = 0
        self._slot_x, self._slot_mask, self._slot_fit = {}, {}, {}
        self._best_mask = None
        self._best_fit = np.inf

        lb, ub = self.x_bounds

        optimizer = self.optimizer_cls(
            obj_func=self._wrapped_objective,
            dim=self.n_features,
            lb=lb,
            ub=ub,
            n_pop=self.n_pop,
            max_fes=self.max_fes,
            seed=self.seed,
            **self.optimizer_kwargs,
        )
        raw_result = optimizer.optimize()

        best_mask = self._best_mask
        best_fit = self._best_fit

        return BinaryWrapperResult(
            best_mask=best_mask,
            best_fitness=best_fit,
            n_selected=int(best_mask.sum()),
            convergence_curve=raw_result.convergence_curve,
            n_evaluations=raw_result.n_evaluations,
            continuous_result=raw_result,
        )