"""
run_experiment.py
------------------
Single-run example/CLI for binary-FNO feature selection on SDP data.
For grid runs across many datasets/classifiers/seeds, use batch_runner.py
instead.

Usage
-----
1) Fetch a real PROMISE/NASA dataset directly from GitHub -- no manual
   download needed:

    python run_experiment.py --promise cm1
    python run_experiment.py --promise kc2   # note: target col differs (problems, yes/no)

2) With your own local CSV (any dataset, binary label column):

    python run_experiment.py --csv path/to/CM1.csv --label defects

3) Without --promise/--csv, runs on a synthetic stand-in dataset, purely to
   smoke-test the pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_loaders import load_dataset, stratified_holdout_split, PROMISE_DATASETS
from binary_fno import BinaryFNO
from binary_pso import BinaryPSO
from binary_de import BinaryDE
from binary_ga import BinaryGA
from binary_cdo import BinaryCDO
from filter_embedded_fs import FilterMI, EmbeddedDT
from sdp_fitness import make_sdp_fitness, evaluate_mask_holdout, SDPFitnessConfig

ALGORITHMS = {"fno": BinaryFNO, "pso": BinaryPSO, "de": BinaryDE, "ga": BinaryGA, "cdo": BinaryCDO}
NON_SEARCH_ALGORITHMS = {"filter_mi": FilterMI, "embedded_dt": EmbeddedDT}
ALL_ALGORITHM_NAMES = list(ALGORITHMS) + list(NON_SEARCH_ALGORITHMS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--promise", type=str, default=None, choices=PROMISE_DATASETS,
                         help="Fetch a PROMISE/NASA dataset directly from GitHub "
                              "(no --csv/--label needed): " + ", ".join(PROMISE_DATASETS))
    parser.add_argument("--algorithm", choices=ALL_ALGORITHM_NAMES, default="fno")
    parser.add_argument("--classifier", choices=["knn", "nb", "dt"], default="knn")
    parser.add_argument("--transfer", choices=["s-shape", "v-shape"], default="s-shape",
                         help="Only used by fno/pso/de (GA is natively binary).")
    parser.add_argument("--n_pop", type=int, default=20)
    parser.add_argument("--max_fes", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.99, help="error-rate vs. feature-ratio weight")
    parser.add_argument("--test_size", type=float, default=0.3,
                         help="Fraction held out as a leakage-free test set (never seen by the "
                              "search). See data_loaders.stratified_holdout_split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, feature_names = load_dataset(csv_path=args.csv, label_col=args.label,
                                        promise=args.promise, seed=args.seed)

    # Split BEFORE anything touches the data -- the search may only see
    # train; test is used exactly once, at the end. See stratified_holdout_
    # split's docstring for why (CV-leakage/selection-bias fix).
    X_train, X_test, y_train, y_test = stratified_holdout_split(
        X, y, seed=args.seed, test_size=args.test_size)
    scaler = StandardScaler().fit(X_train)  # fit on train only
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    config = SDPFitnessConfig(alpha=args.alpha, classifier=args.classifier, seed=args.seed)
    fitness_func = make_sdp_fitness(X_train, y_train, config)

    # --- Baseline: all features -------------------------------------------------
    all_mask = np.ones(X.shape[1], dtype=bool)
    baseline_fit = fitness_func(all_mask)
    baseline_metrics = evaluate_mask_holdout(X_train, y_train, X_test, y_test, all_mask, config)
    print(f"\nBaseline (all {X.shape[1]} features, {args.classifier}): "
          f"fitness={baseline_fit:.4f}, accuracy={baseline_metrics['accuracy']:.4f}, "
          f"f1_macro={baseline_metrics['f1_macro']:.4f}, roc_auc={baseline_metrics['roc_auc']:.4f}")

    # --- Wrapper-based / filter / embedded feature selection ---------------------
    if args.algorithm in NON_SEARCH_ALGORITHMS:
        algo_cls = NON_SEARCH_ALGORITHMS[args.algorithm]
        optimizer = algo_cls(
            fitness_func=fitness_func, n_features=X.shape[1],
            X_train=X_train, y_train=y_train, seed=args.seed, min_selected=1,
        )
        transfer_label = "n/a (ranking + cutoff sweep)"
    else:
        algo_cls = ALGORITHMS[args.algorithm]
        algo_kwargs = dict(
            fitness_func=fitness_func,
            n_features=X.shape[1],
            n_pop=args.n_pop,
            max_fes=args.max_fes,
            seed=args.seed,
            min_selected=1,
        )
        if args.algorithm != "ga":
            algo_kwargs["transfer"] = args.transfer
        optimizer = algo_cls(**algo_kwargs)
        transfer_label = args.transfer if args.algorithm != "ga" else "native-binary"

    result = optimizer.optimize()
    metrics = evaluate_mask_holdout(X_train, y_train, X_test, y_test, result.best_mask, config)

    selected = [feature_names[i] for i, s in enumerate(result.best_mask) if s]
    print(f"\n{args.algorithm.upper()} ({transfer_label}, "
          f"{args.classifier}): fitness={result.best_fitness:.4f}, "
          f"selected {result.n_selected}/{X.shape[1]} features")
    print(f"  accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}, "
          f"roc_auc={metrics['roc_auc']:.4f}")
    print("Selected features:", selected)
    print(f"Evaluations used: {result.n_evaluations}")
    print("Convergence/sweep (first 5 / last 5):")
    print(" ", np.round(result.convergence_curve[:5], 4))
    print(" ", np.round(result.convergence_curve[-5:], 4))


if __name__ == "__main__":
    main()