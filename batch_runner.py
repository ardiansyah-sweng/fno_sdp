"""
batch_runner.py
----------------
Runs binary-FNO across a grid of {dataset x classifier x seed}, recording
one row per run to a CSV (incrementally, so progress survives interrupts),
with resume support so a long local run can be stopped/restarted.

INTENDED USAGE: validate on a tiny grid in a constrained/sandboxed
environment first (few seeds, small max_fes), then run the real, full-scale
grid (many seeds, paper-scale max_fes) on your own machine -- see the
`--estimate` flag below to project total runtime before committing to it.

Examples
--------
# Tiny validation run (fast, just to confirm the pipeline end-to-end):
python batch_runner.py --algorithms fno --datasets cm1 --classifiers knn --seeds 2 \\
    --n_pop 10 --max_fes 300 --out validation_results.csv

# Estimate total wall-clock time for a full-scale grid WITHOUT running it
# (runs exactly one combination, times it, projects the rest):
python batch_runner.py --algorithms all --datasets cm1,kc1,kc2,jm1,pc1 --classifiers knn,nb,dt \\
    --seeds 30 --n_pop 20 --max_fes 3000 --estimate

# Full-scale run, all 4 algorithms (intended for your local machine, not the sandbox):
python batch_runner.py --algorithms all --datasets cm1,kc1,kc2,jm1,pc1 --classifiers knn,nb,dt \\
    --seeds 30 --n_pop 20 --max_fes 3000 --n_jobs 8 --out results_all.csv

# Resume an interrupted run (same command -- already-completed rows in
# --out are detected and skipped automatically):
python batch_runner.py --algorithms all --datasets cm1,kc1,kc2,jm1,pc1 --classifiers knn,nb,dt \\
    --seeds 30 --n_pop 20 --max_fes 3000 --n_jobs 8 --out results_all.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.preprocessing import StandardScaler

from data_loaders import load_dataset, stratified_holdout_split, PROMISE_DATASETS
from binary_fno import BinaryFNO
from binary_pso import BinaryPSO
from binary_de import BinaryDE
from binary_ga import BinaryGA
from binary_cdo import BinaryCDO
from filter_embedded_fs import FilterMI, EmbeddedDT
from sdp_fitness import SDPFitnessConfig, make_sdp_fitness, evaluate_mask_holdout

ALGORITHMS = {"fno": BinaryFNO, "pso": BinaryPSO, "de": BinaryDE, "ga": BinaryGA, "cdo": BinaryCDO}
# Filter/embedded baselines are dispatched separately in run_one() (they need
# direct X_train/y_train access, not just the fitness_func closure), but are
# included here so --algorithms all / completeness checks / CLI choices see them.
NON_SEARCH_ALGORITHMS = {"filter_mi": FilterMI, "embedded_dt": EmbeddedDT}
ALL_ALGORITHM_NAMES = list(ALGORITHMS) + list(NON_SEARCH_ALGORITHMS)
ALL_CLASSIFIERS = ["knn", "nb", "dt"]
CSV_FIELDS = [
    "algorithm", "dataset", "classifier", "seed", "transfer",
    "n_pop", "max_fes", "alpha", "test_size",
    "fitness", "n_selected", "n_total_features",
    "accuracy", "f1_macro", "roc_auc",
    "selected_features", "n_evaluations", "wall_time_sec",
]


def parse_list_arg(s: str, all_values: list[str]) -> list[str]:
    if s.strip().lower() == "all":
        return list(all_values)
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def parse_seeds_arg(s: str) -> list[int]:
    # Either an integer count ("30" -> seeds 0..29) or an explicit comma list ("1,2,7")
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return list(range(int(s)))


def load_completed_keys(out_path: str) -> set:
    completed = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["algorithm"], row["dataset"], row["classifier"], row["seed"])
                completed.add(key)
    return completed


def append_row(out_path: str, row: dict):
    file_exists = os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(dataset_name: str, X: np.ndarray, y: np.ndarray, feature_names: list[str],
            algorithm: str, classifier: str, seed: int, args) -> dict:
    t0 = time.time()

    # Split BEFORE anything else touches the data: the search (fitness_func)
    # may only ever see X_train/y_train. X_test/y_test are used exactly once,
    # at the very end, by evaluate_mask_holdout. See data_loaders.
    # stratified_holdout_split's docstring for why (CV-leakage/selection-bias
    # fix, Ambroise & McLachlan 2002; Cawley & Talbot 2010).
    X_train, X_test, y_train, y_test = stratified_holdout_split(
        X, y, seed=seed, test_size=args.test_size)

    scaler = StandardScaler().fit(X_train)  # fit on train only -- same leakage principle
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    config = SDPFitnessConfig(alpha=args.alpha, classifier=classifier, seed=seed)
    fitness_func = make_sdp_fitness(X_train, y_train, config)

    if algorithm == "baseline_all":
        # No search: evaluate the all-features mask directly. Included in
        # the same grid/CSV/resume mechanism as the search algorithms (see
        # main()) so the "No-FS" comparison row is always available without
        # a separate, uncoordinated run.
        all_mask = np.ones(X.shape[1], dtype=bool)
        fit = fitness_func(all_mask)
        metrics = evaluate_mask_holdout(X_train, y_train, X_test, y_test, all_mask, config)
        return {
            "algorithm": "baseline_all",
            "dataset": dataset_name,
            "classifier": classifier,
            "seed": seed,
            "transfer": "n/a",
            "n_pop": "n/a",
            "max_fes": "n/a",
            "alpha": args.alpha,
            "test_size": args.test_size,
            "fitness": fit,
            "n_selected": X.shape[1],
            "n_total_features": X.shape[1],
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "roc_auc": metrics["roc_auc"],
            "selected_features": "|".join(feature_names),
            "n_evaluations": 1,
            "wall_time_sec": round(time.time() - t0, 3),
        }

    if algorithm in NON_SEARCH_ALGORITHMS:
        # Filter (MI) / embedded (DT importance) baselines: rank once, sweep
        # every cutoff k=1..n_features against the SAME fitness function the
        # search algorithms use, keep the best k. See filter_embedded_fs.py
        # for why this keeps the comparison fair despite not being a
        # population-based search.
        algo_cls = NON_SEARCH_ALGORITHMS[algorithm]
        optimizer = algo_cls(
            fitness_func=fitness_func,
            n_features=X.shape[1],
            X_train=X_train,
            y_train=y_train,
            seed=seed,
            min_selected=1,
        )
        result = optimizer.optimize()
        metrics = evaluate_mask_holdout(X_train, y_train, X_test, y_test, result.best_mask, config)

        if args.save_curves:
            curves_dir = Path(args.curves_dir)
            curves_dir.mkdir(parents=True, exist_ok=True)
            curve_path = curves_dir / f"{algorithm}_{dataset_name}_{classifier}_seed{seed}.npy"
            np.save(curve_path, result.convergence_curve)

        selected = [feature_names[i] for i, s in enumerate(result.best_mask) if s]
        return {
            "algorithm": algorithm,
            "dataset": dataset_name,
            "classifier": classifier,
            "seed": seed,
            "transfer": "n/a",
            "n_pop": "n/a",
            "max_fes": "n/a",
            "alpha": args.alpha,
            "test_size": args.test_size,
            "fitness": result.best_fitness,
            "n_selected": result.n_selected,
            "n_total_features": X.shape[1],
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "roc_auc": metrics["roc_auc"],
            "selected_features": "|".join(selected),
            "n_evaluations": result.n_evaluations,
            "wall_time_sec": round(time.time() - t0, 3),
        }

    algo_cls = ALGORITHMS[algorithm]
    algo_kwargs = dict(
        fitness_func=fitness_func,
        n_features=X.shape[1],
        n_pop=args.n_pop,
        max_fes=args.max_fes,
        seed=seed,
        min_selected=1,
    )
    if algorithm != "ga":  # GA is natively binary, no transfer function
        algo_kwargs["transfer"] = args.transfer
    optimizer = algo_cls(**algo_kwargs)
    result = optimizer.optimize()
    metrics = evaluate_mask_holdout(X_train, y_train, X_test, y_test, result.best_mask, config)

    if args.save_curves:
        curves_dir = Path(args.curves_dir)
        curves_dir.mkdir(parents=True, exist_ok=True)
        curve_path = curves_dir / f"{algorithm}_{dataset_name}_{classifier}_seed{seed}.npy"
        np.save(curve_path, result.convergence_curve)

    selected = [feature_names[i] for i, s in enumerate(result.best_mask) if s]

    return {
        "algorithm": algorithm,
        "dataset": dataset_name,
        "classifier": classifier,
        "seed": seed,
        "transfer": args.transfer if algorithm != "ga" else "n/a",
        "n_pop": args.n_pop,
        "max_fes": args.max_fes,
        "alpha": args.alpha,
        "test_size": args.test_size,
        "fitness": result.best_fitness,
        "n_selected": result.n_selected,
        "n_total_features": X.shape[1],
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "roc_auc": metrics["roc_auc"],
        "selected_features": "|".join(selected),
        "n_evaluations": result.n_evaluations,
        "wall_time_sec": round(time.time() - t0, 3),
    }


# --- Worker-process machinery for --n_jobs > 1 --------------------------- #
# Each ProcessPoolExecutor worker is a persistent process reused across
# submitted tasks, so a module-level cache here avoids re-fetching the same
# dataset from GitHub on every task handled by that worker. Cache stores RAW
# (unscaled) data -- scaling must happen AFTER the train/test split inside
# run_one() (fit on train only), so it can't be cached across seeds/splits.
_worker_data_cache: dict[str, tuple] = {}


def _get_data_cached(name: str):
    if name not in _worker_data_cache:
        _worker_data_cache[name] = load_dataset(promise=name, verbose=False)
    return _worker_data_cache[name]


def _worker_run(task):
    dataset_name, algorithm, classifier, seed, args = task
    X, y, feature_names = _get_data_cached(dataset_name)
    return run_one(dataset_name, X, y, feature_names, algorithm, classifier, seed, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", type=str, default="all",
                         help=f"Comma list or 'all'. Available: {ALL_ALGORITHM_NAMES}")
    parser.add_argument("--datasets", type=str, default="all",
                         help=f"Comma list or 'all'. Available: {PROMISE_DATASETS}")
    parser.add_argument("--classifiers", type=str, default="all",
                         help=f"Comma list or 'all'. Available: {ALL_CLASSIFIERS}")
    parser.add_argument("--seeds", type=str, default="5",
                         help="Integer count (uses 0..N-1) or explicit comma list, e.g. '1,2,7'")
    parser.add_argument("--transfer", choices=["s-shape", "v-shape"], default="s-shape",
                         help="Only used by fno/pso/de (GA is natively binary).")
    parser.add_argument("--n_pop", type=int, default=20)
    parser.add_argument("--max_fes", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--test_size", type=float, default=0.3,
                         help="Fraction of data held out as a leakage-free test set, split "
                              "fresh per seed (stratified). See data_loaders.stratified_holdout_split.")
    parser.add_argument("--out", type=str, default="results_binary_fno.csv")
    parser.add_argument("--no_baseline", action="store_true",
                         help="Skip the automatic all-features 'baseline_all' row that is "
                              "otherwise added once per (dataset, classifier, seed) regardless "
                              "of --algorithms, so the No-FS comparison is always available "
                              "without a separate uncoordinated run.")
    parser.add_argument("--save_curves", dest="save_curves", action="store_true", default=True,
                         help="Save each run's convergence curve to --curves_dir as a .npy file "
                              "(default: on). Needed for convergence plots later without "
                              "re-running -- storage cost is small (~a few tens of MB total for "
                              "a full-scale grid at max_fes=3000).")
    parser.add_argument("--no_save_curves", dest="save_curves", action="store_false")
    parser.add_argument("--curves_dir", type=str, default="curves")
    parser.add_argument("--n_jobs", type=int, default=1,
                         help="Parallel worker processes (use >1 on your local machine to "
                              "speed up the full-scale grid; keep at 1 in constrained/sandboxed "
                              "environments). Each grid combination is independent, so this "
                              "scales close to linearly with CPU cores.")
    parser.add_argument("--estimate", action="store_true",
                         help="Run exactly ONE combination, time it, and project total "
                              "runtime for the full requested grid -- does not run the grid.")
    args = parser.parse_args()

    algorithms = parse_list_arg(args.algorithms, ALL_ALGORITHM_NAMES)
    datasets = parse_list_arg(args.datasets, PROMISE_DATASETS)
    classifiers = parse_list_arg(args.classifiers, ALL_CLASSIFIERS)
    seeds = parse_seeds_arg(args.seeds)

    grid = [(a, d, c, s) for a in algorithms for d in datasets for c in classifiers for s in seeds]
    if not args.no_baseline:
        # One baseline_all row per (dataset, classifier, seed), independent
        # of which algorithms were requested -- see run_one()'s handling of
        # algorithm=="baseline_all". Resume/skip logic treats it like any
        # other algorithm, so re-running with different --algorithms later
        # won't recompute baselines already present in --out.
        grid += [("baseline_all", d, c, s) for d in datasets for c in classifiers for s in seeds]
    print(f"Grid: {len(algorithms)} algorithms x {len(datasets)} datasets x "
          f"{len(classifiers)} classifiers x {len(seeds)} seeds = {len(algorithms)*len(datasets)*len(classifiers)*len(seeds)} runs "
          f"{'+ ' + str(len(datasets)*len(classifiers)*len(seeds)) + ' baseline_all rows ' if not args.no_baseline else ''}"
          f"(n_pop={args.n_pop}, max_fes={args.max_fes}); total={len(grid)}")

    def load_raw(name: str):
        return load_dataset(promise=name)

    if args.estimate:
        a, d, c, s = grid[0]
        X, y, feature_names = load_raw(d)
        print(f"\nTiming one run: algorithm={a}, dataset={d}, classifier={c}, seed={s} ...")
        row = run_one(d, X, y, feature_names, a, c, s, args)
        t = row["wall_time_sec"]
        total_sec = t * len(grid)
        print(f"\nSingle run took {t:.2f}s.")
        print(f"Projected total for the full grid ({len(grid)} runs): "
              f"~{timedelta(seconds=round(total_sec))} (sequential, single-process)")
        print("Note: actual PROMISE datasets vary a lot in size (jm1 ~13k rows vs cm1 ~500), "
              "and algorithms differ in per-evaluation overhead too -- this projection assumes "
              "uniform per-run cost from the single timed combination above. For a tighter "
              "estimate, time a couple of combinations spanning your largest dataset and "
              "slowest algorithm/classifier.")
        return

    completed = load_completed_keys(args.out)
    if completed:
        print(f"Resuming: {len(completed)} runs already found in {args.out}, will be skipped.")

    pending = [(a, d, c, s) for a, d, c, s in grid if (a, d, c, str(s)) not in completed]
    n_skipped = len(grid) - len(pending)
    n_run = 0
    t_start = time.time()

    if args.n_jobs <= 1:
        # Sequential: cache RAW datasets in the main process's dict (scaling
        # happens per-seed inside run_one(), after the train/test split).
        data_cache: dict[str, tuple] = {}

        def get_data(name: str):
            if name not in data_cache:
                data_cache[name] = load_dataset(promise=name)
            return data_cache[name]

        for a, d, c, s in pending:
            X, y, feature_names = get_data(d)
            row = run_one(d, X, y, feature_names, a, c, s, args)
            append_row(args.out, row)
            n_run += 1
            elapsed = time.time() - t_start
            print(f"[{n_run+n_skipped}/{len(grid)}] {a}/{d}/{c}/seed{s}: "
                  f"fitness={row['fitness']:.4f} acc={row['accuracy']:.4f} "
                  f"n_sel={row['n_selected']}/{row['n_total_features']} "
                  f"({row['wall_time_sec']:.1f}s, elapsed {timedelta(seconds=round(elapsed))})")
    else:
        # Parallel: submit all pending tasks to a process pool; append each
        # result to the CSV as it completes (only the main process writes,
        # so no risk of interleaved/corrupted CSV rows).
        print(f"Running with {args.n_jobs} parallel worker processes...")
        tasks = [(d, a, c, s, args) for a, d, c, s in pending]
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {pool.submit(_worker_run, t): t for t in tasks}
            for fut in as_completed(futures):
                d, a, c, s, _ = futures[fut]
                row = fut.result()
                append_row(args.out, row)
                n_run += 1
                elapsed = time.time() - t_start
                print(f"[{n_run+n_skipped}/{len(grid)}] {a}/{d}/{c}/seed{s}: "
                      f"fitness={row['fitness']:.4f} acc={row['accuracy']:.4f} "
                      f"n_sel={row['n_selected']}/{row['n_total_features']} "
                      f"({row['wall_time_sec']:.1f}s, elapsed {timedelta(seconds=round(elapsed))})")

    print(f"\nDone. {n_run} runs executed, {n_skipped} skipped (already in {args.out}). "
          f"Total elapsed: {timedelta(seconds=round(time.time() - t_start))}")


if __name__ == "__main__":
    main()