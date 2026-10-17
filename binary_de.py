"""
binary_de.py
------------
DE-specialized binary wrapper -- baseline for comparison against
binary-FNO. See binary_wrapper.py for the shared binarization logic and
continuous_optimizers.py for the underlying DE (Storn & Price 1997,
DE/rand/1/bin).
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from binary_wrapper import BinaryWrapperOptimizer, TransferKind
from continuous_optimizers import DE


class BinaryDE(BinaryWrapperOptimizer):
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
        F: float = 0.5,
        CR: float = 0.9,
    ):
        super().__init__(
            optimizer_cls=DE,
            fitness_func=fitness_func,
            n_features=n_features,
            n_pop=n_pop,
            max_fes=max_fes,
            seed=seed,
            transfer=transfer,
            x_bounds=x_bounds,
            min_selected=min_selected,
            optimizer_kwargs=dict(F=F, CR=CR),
        )