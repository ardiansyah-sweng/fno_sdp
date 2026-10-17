"""
sdp_fitness.py
--------------
Fitness function (drives the search, INTERNAL cross-validation on the
TRAINING partition only) and a separate, one-shot HELD-OUT evaluation
function (for post-hoc reporting) for binary-FNO/PSO/DE/GA applied to
Software Defect Prediction (SDP) feature selection.

IMPORTANT (fixed after an earlier version of this pipeline reused the same
CV folds for both search and final reporting -- a documented selection-bias
problem, see Ambroise & McLachlan 2002, PNAS 99(10):6562-6566, and Cawley &
Talbot 2010, JMLR 11): callers MUST split data into a search/train partition
and a held-out test partition BEFORE calling anything here -- see
data_loaders.stratified_holdout_split. `make_sdp_fitness` must only ever see
the train partition; `evaluate_mask_holdout` is the only function allowed to
see the test partition, and only ONCE, after search has finished.

Two distinct things live here, deliberately kept separate:

  1. `make_sdp_fitness(X_train, y_train, ...)` -> the scalar objective the
     search minimizes: alpha * ErrorRate + (1-alpha) * feature_ratio,
     ErrorRate estimated via cross-validation WITHIN the training partition
     only (the standard wrapper-FS weighting; see class docstring below).

  2. `evaluate_mask_holdout(X_train, y_train, X_test, y_test, mask, ...)` ->
     fit once on the full training partition (selected features only),
     evaluate once on the held-out test partition. This is what should go
     into result tables/plots for the paper.

Classifier choice (research plan: compare KNN / Naive Bayes / Decision
Tree, per prior SDP feature-selection papers):
  "knn" -> KNeighborsClassifier
  "nb"  -> GaussianNB
  "dt"  -> DecisionTreeClassifier (max_depth capped by default -- see
           make_classifier note: both a speed and an overfitting concern
           on noisy/imbalanced SDP data)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ClassifierKind = Literal["knn", "nb", "dt"]


def make_classifier(kind: ClassifierKind, n_neighbors: int = 5, seed: Optional[int] = 42,
                     dt_max_depth: Optional[int] = 10):
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=n_neighbors)
    elif kind == "nb":
        return GaussianNB()
    elif kind == "dt":
        # max_depth capped by default: unconstrained trees are both much
        # slower on larger SDP sets (e.g. jm1, ~13k rows) AND prone to
        # overfitting noisy/imbalanced defect data -- bounding depth is
        # standard practice in the SDP-FS literature, not just a speed hack.
        return DecisionTreeClassifier(random_state=seed, max_depth=dt_max_depth)
    else:
        raise ValueError(f"Unknown classifier kind '{kind}'. Use 'knn', 'nb', or 'dt'.")


@dataclass
class SDPFitnessConfig:
    alpha: float = 0.99              # weight on classification error vs. feature ratio
    classifier: ClassifierKind = "knn"
    n_neighbors: int = 5             # only used when classifier == "knn"
    dt_max_depth: Optional[int] = 10 # only used when classifier == "dt"; see make_classifier note
    n_splits: int = 5                # stratified k-fold
    scoring: str = "roc_auc"         # objective driving the *search* fitness -- AUC (not
                                      # accuracy) by design: SDP datasets are class-imbalanced
                                      # (defect rates ~7-20% in this study), where a trivial
                                      # majority-class classifier already scores ~80-93%
                                      # accuracy. Driving the search with accuracy risks
                                      # selecting features that help the majority class while
                                      # ignoring the minority (defective) class, which is the
                                      # class that actually matters for SDP. AUC is
                                      # threshold-independent and standard in the SDP-FS
                                      # literature for imbalanced data.
    seed: Optional[int] = 42


def make_sdp_fitness(X: np.ndarray, y: np.ndarray, config: SDPFitnessConfig = SDPFitnessConfig()):
    """Returns a fitness_func(mask) -> float, minimized by BinaryFNO.

    See module docstring. Uses cached StratifiedKFold splits (fixed seed) so
    every fitness evaluation within a run is CV'd on the identical folds --
    important for a fair comparison across candidate solutions within one
    search run.
    """
    n_features = X.shape[1]
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    cv_splits = list(skf.split(X, y))  # materialize once -> identical folds every call

    # KNN's n_neighbors must be < the number of samples in the SMALLEST
    # training fold (not related to how many features are selected -- an
    # earlier version of this code incorrectly tied n_neighbors to the
    # feature count `k`, which is a different quantity entirely).
    min_fold_train_size = min(len(train_idx) for train_idx, _ in cv_splits)

    scoring_map = {"accuracy": "accuracy", "f1": "f1_macro", "roc_auc": "roc_auc"}
    sk_scoring = scoring_map[config.scoring]

    def fitness(mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool)
        k = int(mask.sum())
        if k == 0:
            return 1.0  # maximal penalty; BinaryFNO's repair step should prevent this anyway

        X_sub = X[:, mask]
        n_neighbors = (min(config.n_neighbors, min_fold_train_size - 1)
                       if config.classifier == "knn" else config.n_neighbors)
        clf = make_classifier(config.classifier, n_neighbors=n_neighbors, seed=config.seed,
                               dt_max_depth=config.dt_max_depth)

        scores = cross_val_score(clf, X_sub, y, cv=cv_splits, scoring=sk_scoring)
        mean_score = scores.mean()
        error_rate = 1.0 - mean_score

        feature_ratio = k / n_features
        return config.alpha * error_rate + (1.0 - config.alpha) * feature_ratio

    return fitness


def evaluate_mask_holdout(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           mask: np.ndarray, config: SDPFitnessConfig):
    """One-shot, leakage-free report: fit ONCE on the full training partition
    (selected features only), evaluate ONCE on the held-out test partition.

    X_test/y_test must be data the search never touched (see module
    docstring and data_loaders.stratified_holdout_split). Calling this
    function is the ONLY point at which test data may be used, and it
    should be called at most once per (algorithm, dataset, classifier,
    seed) combination -- i.e., after the search has already finished and
    committed to a final mask.

    Returns a dict: {"accuracy":..., "f1_macro":..., "roc_auc":..., "n_selected":...}
    roc_auc is NaN if the mask selects 0 features, only one class is present
    in y_train/y_test, or the classifier can't produce probability scores.
    """
    mask = np.asarray(mask, dtype=bool)
    k = int(mask.sum())
    if k == 0:
        return {"accuracy": np.nan, "f1_macro": np.nan, "roc_auc": np.nan, "n_selected": 0}

    X_train_sub = X_train[:, mask]
    X_test_sub = X_test[:, mask]
    # Same fix as make_sdp_fitness: guard n_neighbors against the number of
    # TRAINING SAMPLES available, not the number of selected features.
    n_neighbors = (min(config.n_neighbors, len(y_train) - 1)
                   if config.classifier == "knn" else config.n_neighbors)
    clf = make_classifier(config.classifier, n_neighbors=n_neighbors, seed=config.seed,
                           dt_max_depth=config.dt_max_depth)

    clf.fit(X_train_sub, y_train)
    y_pred = clf.predict(X_test_sub)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    try:
        y_score = clf.predict_proba(X_test_sub)[:, 1]
        auc = roc_auc_score(y_test, y_score)
    except (ValueError, IndexError, AttributeError):
        # roc_auc undefined if y_test/y_train has only one class present, or
        # the classifier lacks predict_proba
        auc = np.nan

    return {"accuracy": acc, "f1_macro": f1m, "roc_auc": auc, "n_selected": k}