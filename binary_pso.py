"""
binary_pso.py
-------------
PSO-specialized binary wrapper -- baseline for comparison against
binary-FNO. See binary_wrapper.py for the shared binarization logic and
continuous_optimizers.py for the underlying PSO (Kennedy & Eberhart 1995).
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from binary_wrapper import BinaryWrapperOptimizer, TransferKind
from continuous_optimizers import PSO


class BinaryPSO(BinaryWrapperOptimizer):
    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],
        n_features: int,
        n_pop: int = 20,
        max_fes: int = 5000,
        seed: Optional[int] = None,
        transfer: TransferKind = "s-shape",
        x_bounds=(-4.0, 4.0),
        min_selected: int = 1,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
    ):
        super().__init__(
            optimizer_cls=PSO,
            fitness_func=fitness_func,
            n_features=n_features,
            n_pop=n_pop,
            max_fes=max_fes,
            seed=seed,
            transfer=transfer,
            x_bounds=x_bounds,
            min_selected=min_selected,
            optimizer_kwargs=dict(w_max=w_max, w_min=w_min, c1=c1, c2=c2),
        )