"""
binary_cdo.py
-------------
CDO-specialized binary wrapper -- baseline for comparison against
binary-FNO. See binary_wrapper.py for the shared binarization logic and
continuous_optimizers.py for the underlying CDO (Shehadeh 2023, ported
from the official MATLAB source).
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from binary_wrapper import BinaryWrapperOptimizer, TransferKind
from continuous_optimizers import CDO


class BinaryCDO(BinaryWrapperOptimizer):
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
            optimizer_cls=CDO,
            fitness_func=fitness_func,
            n_features=n_features,
            n_pop=n_pop,
            max_fes=max_fes,
            seed=seed,
            transfer=transfer,
            x_bounds=x_bounds,
            min_selected=min_selected,
        )