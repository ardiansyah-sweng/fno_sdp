"""
filter_embedded_fs.py
----------------------
Two cheap, non-search-based feature-selection baselines, added to compare
FNO/GA/PSO/DE/CDO's wrapper metaheuristic search against alternative FS
paradigms (filter and embedded), not just other wrapper searches.

  - Filter: Mutual Information ranking (sklearn.feature_selection.
    mutual_info_classif) between each feature and the target, computed once
    on the training partition. Chosen over Chi-square because MI handles
    continuous features directly -- the McCabe/Halstead metrics used
    throughout this study are continuous, and Chi-square would require an
    extra, hyperparameter-laden discretization step.

  - Embedded: Decision Tree feature importance (Gini importance),
    computed once on the training partition (same max_depth cap as the
    DT classifier used elsewhere in this study, for consistency).

Neither is a metaheuristic search: each produces a single feature RANKING,
not iteratively searched candidate subsets. To make the "how many features
to keep" decision non-arbitrary -- and to keep the comparison against the
wrapper methods fair rather than favoring whichever paradigm happens to
land on a lucky cutoff -- both baselines sweep every possible cutoff
k = 1..n_features, evaluate each top-k mask with the IDENTICAL fitness
function used by the wrapper methods (sdp_fitness.make_sdp_fitness:
cross-validated AUC + feature-ratio, computed on the training partition
only), and keep the best-scoring k. This costs at most n_features fitness
evaluations (<=39 in this study's datasets) -- negligible next to the
wrapper methods' thousands of evaluations -- while keeping every other
part of the comparison identical: same fitness function, same
training-only evaluation during search, same one-shot held-out reporting
(sdp_fitness.evaluate_mask_holdout). Only the SEARCH STRATEGY differs
(ranking + cutoff sweep vs. population-based iterative search).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier


@dataclass
class FilterEmbeddedResult:
    best_mask: np.ndarray
    best_fitness: float
    n_selected: int
    convergence_curve: np.ndarray  # best-so-far fitness across the k=1..n_features sweep
    n_evaluations: int


def _sweep_top_k(ranking: np.ndarray, fitness_func: Callable[[np.ndarray], float],
                  n_features: int, min_selected: int = 1) -> FilterEmbeddedResult:
    """ranking: feature indices sorted best-first. Evaluate top-k masks for
    k = min_selected..n_features via fitness_func, keep the best (lowest)."""
    best_fit = np.inf
    best_mask = None
    curve = []
    for k in range(min_selected, n_features + 1):
        mask = np.zeros(n_features, dtype=bool)
        mask[ranking[:k]] = True
        fit = fitness_func(mask)
        if fit < best_fit:
            best_fit = fit
            best_mask = mask.copy()
        curve.append(best_fit)
    return FilterEmbeddedResult(
        best_mask=best_mask,
        best_fitness=best_fit,
        n_selected=int(best_mask.sum()),
        convergence_curve=np.array(curve),
        n_evaluations=len(curve),
    )


class FilterMI:
    """Mutual-Information filter ranking + fitness-guided cutoff sweep.

    Unlike the search-based algorithms in this package (FNO/PSO/DE/CDO/GA),
    this needs direct access to the training data (to compute MI scores),
    not just the fitness_func closure -- so its constructor takes X_train/
    y_train explicitly rather than only n_features/n_pop/max_fes.
    """

    def __init__(self, fitness_func: Callable[[np.ndarray], float], n_features: int,
                 X_train: np.ndarray = None, y_train: np.ndarray = None,
                 seed: Optional[int] = None, min_selected: int = 1, **_ignored_kwargs):
        # _ignored_kwargs absorbs n_pop/max_fes/transfer/etc. so this class
        # can be called with the same **algo_kwargs dict batch_runner.py
        # already builds for the search-based algorithms, without needing
        # a separate call-site branch for every argument.
        if X_train is None or y_train is None:
            raise ValueError("FilterMI requires X_train and y_train to compute mutual "
                              "information scores.")
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.X_train = X_train
        self.y_train = y_train
        self.seed = seed
        self.min_selected = min_selected

    def optimize(self) -> FilterEmbeddedResult:
        mi = mutual_info_classif(self.X_train, self.y_train, random_state=self.seed)
        ranking = np.argsort(-mi)  # descending MI, most relevant first
        return _sweep_top_k(ranking, self.fitness_func, self.n_features, self.min_selected)


class EmbeddedDT:
    """Decision Tree Gini-importance ranking + fitness-guided cutoff sweep.

    max_depth matches the DT classifier used elsewhere in this study
    (sdp_fitness.SDPFitnessConfig.dt_max_depth default) so the importance
    ranking comes from a comparably regularized tree, not an unconstrained
    one prone to overfitting on noisy/imbalanced defect data.
    """

    def __init__(self, fitness_func: Callable[[np.ndarray], float], n_features: int,
                 X_train: np.ndarray = None, y_train: np.ndarray = None,
                 seed: Optional[int] = None, min_selected: int = 1,
                 dt_max_depth: int = 10, **_ignored_kwargs):
        if X_train is None or y_train is None:
            raise ValueError("EmbeddedDT requires X_train and y_train to compute feature "
                              "importances.")
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.X_train = X_train
        self.y_train = y_train
        self.seed = seed
        self.min_selected = min_selected
        self.dt_max_depth = dt_max_depth

    def optimize(self) -> FilterEmbeddedResult:
        clf = DecisionTreeClassifier(random_state=self.seed, max_depth=self.dt_max_depth)
        clf.fit(self.X_train, self.y_train)
        ranking = np.argsort(-clf.feature_importances_)
        return _sweep_top_k(ranking, self.fitness_func, self.n_features, self.min_selected)