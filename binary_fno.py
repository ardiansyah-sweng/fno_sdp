"""
binary_fno.py
-------------
FNO-specialized binary wrapper. Thin convenience subclass of
binary_wrapper.BinaryWrapperOptimizer -- kept as its own module/class name
for backward compatibility with earlier scripts (`from binary_fno import
BinaryFNO`). See binary_wrapper.py for the actual binarization logic
(shared across BinaryFNO/BinaryPSO/BinaryDE).
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from binary_wrapper import BinaryWrapperOptimizer, BinaryWrapperResult, TransferKind
from fno_core import FNO

# Backward-compat alias: earlier code imports BinaryFNOResult
BinaryFNOResult = BinaryWrapperResult


class BinaryFNO(BinaryWrapperOptimizer):
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
    ):
        super().__init__(
            optimizer_cls=FNO,
            fitness_func=fitness_func,
            n_features=n_features,
            n_pop=n_pop,
            max_fes=max_fes,
            seed=seed,
            transfer=transfer,
            x_bounds=x_bounds,
            min_selected=min_selected,
        )