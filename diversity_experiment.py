"""
diversity_experiment.py
-----------------------
Tests two things the main grid could not:

1. MASK DIVERSITY over the course of a search (RQ3, attractor hypothesis).
   Every mask passed to the fitness function is recorded by a thin wrapper
   (no optimizer is modified). Evaluations are grouped into windows of n_pop
   consecutive evaluations ("generations"); for each window we record the
   mean pairwise Hamming distance between the masks evaluated in it,
   normalized by the number of features (0 = identical, about 0.5 = random).
   If attractor-based methods (FNO, PSO, CDO) collapse diversity earlier than
   GA and DE, their curves should fall faster and end lower.

2. TRANSFER FUNCTION SENSITIVITY (S-shape vs V-shape). Run the same grid with
   --transfer v-shape and compare final held-out AUC and the diversity curves.
   GA is natively binary and ignores --transfer (its rows are identical
   across transfer settings and are skipped for v-shape).

Same split, scaling, fitness, budget and seeds as batch_runner.py, so the
s-shape results here are directly comparable with results_all.csv.

Outputs (one row per run) to --out CSV, plus --curves_out .npz holding the
per-run diversity curves. Resumable: rows already in --out are skipped.

Example (small subset, run locally):
  python diversity_experiment.py --datasets cm1,mw1,kc3,pc1 --classifiers knn \\
      --algorithms fno,pso,cdo,de,ga --seeds 10 --transfer s-shape \\
      --out diversity_s.csv --curves_out diversity_s.npz --n_jobs 4
  python diversity_experiment.py --datasets cm1,mw1,kc3,pc1 --classifiers knn \\
      --algorithms fno,pso,cdo,de --seeds 10 --transfer v-shape \\
      --out diversity_v.csv --curves_out diversity_v.npz --n_jobs 4
Use --estimate to time one run and project the total before committing.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from data_loaders import load_dataset, stratified_holdout_split
from sdp_fitness import SDPFitnessConfig, make_sdp_fitness, evaluate_mask_holdout
from binary_fno import BinaryFNO
from binary_pso import BinaryPSO
from binary_de import BinaryDE
from binary_ga import BinaryGA
from binary_cdo import BinaryCDO

ALGOS = {"fno": BinaryFNO, "pso": BinaryPSO, "de": BinaryDE, "ga": BinaryGA, "cdo": BinaryCDO}


def mean_pairwise_hamming(masks: np.ndarray) -> float:
    """Mean pairwise Hamming distance of a (m, n) boolean matrix, normalized by n."""
    m, n = masks.shape
    if m < 2:
        return np.nan
    ones = masks.sum(axis=0).astype(float)           # per-feature count of ones
    # pairs differing on feature j: ones_j * (m - ones_j); total pairs: m(m-1)/2
    diff_pairs = (ones * (m - ones)).sum()
    return float(diff_pairs / (m * (m - 1) / 2.0) / n)


def diversity_curve(masks: list, window: int) -> np.ndarray:
    arr = np.asarray(masks, dtype=bool)
    out = []
    for s in range(0, len(arr) - window + 1, window):
        out.append(mean_pairwise_hamming(arr[s:s + window]))
    return np.array(out)


def run_one(ds, X, y, names, algo, clf, seed, transfer, n_pop, max_fes, alpha, test_size):
    t0 = time.time()
    Xtr, Xte, ytr, yte = stratified_holdout_split(X, y, seed=seed, test_size=test_size)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    cfg = SDPFitnessConfig(alpha=alpha, classifier=clf, seed=seed)
    base_fit = make_sdp_fitness(Xtr, ytr, cfg)

    recorded = []

    def fit(mask):
        recorded.append(np.asarray(mask, dtype=bool).copy())
        return base_fit(mask)

    kw = dict(fitness_func=fit, n_features=X.shape[1], n_pop=n_pop, max_fes=max_fes,
              seed=seed, min_selected=1)
    if algo != "ga":
        kw["transfer"] = transfer
    res = ALGOS[algo](**kw).optimize()
    metrics = evaluate_mask_holdout(Xtr, ytr, Xte, yte, res.best_mask, cfg)
    curve = diversity_curve(recorded, n_pop)
    distinct = len({m.tobytes() for m in recorded}) / max(len(recorded), 1)
    row = dict(algorithm=algo, dataset=ds, classifier=clf, seed=seed, transfer=transfer if algo != "ga" else "n/a",
               fitness=res.best_fitness, n_selected=res.n_selected, roc_auc=metrics["roc_auc"],
               div_first=float(curve[0]), div_mid=float(curve[len(curve) // 2]), div_last=float(curve[-1]),
               div_mean=float(np.nanmean(curve)), distinct_frac=distinct,
               n_evals=len(recorded), wall_time_sec=round(time.time() - t0, 2))
    return row, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="cm1,mw1,kc3,pc1")
    ap.add_argument("--classifiers", default="knn")
    ap.add_argument("--algorithms", default="fno,pso,cdo,de,ga")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--transfer", choices=["s-shape", "v-shape", "v-shape-ind"], default="s-shape")
    ap.add_argument("--n_pop", type=int, default=20)
    ap.add_argument("--max_fes", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--out", default="diversity.csv")
    ap.add_argument("--curves_out", default="diversity.npz")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()

    algos = a.algorithms.split(",")
    if a.transfer in ("v-shape", "v-shape-ind"):
        algos = [x for x in algos if x != "ga"]  # GA ignores the transfer function
    dss, clfs = a.datasets.split(","), a.classifiers.split(",")

    data = {}
    for d in dss:
        X, y, names = load_dataset(promise=d, verbose=False)
        data[d] = (X, y, names)

    done = set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        done = set(zip(prev.algorithm, prev.dataset, prev.classifier, prev.seed))
    jobs = [(d, c, al, s) for d, c, al, s in itertools.product(dss, clfs, algos, range(a.seeds))
            if (al, d, c, s) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already in {a.out}), transfer={a.transfer}")

    if a.estimate and jobs:
        d, c, al, s = jobs[0]
        t = time.time()
        run_one(d, *data[d], al, c, s, a.transfer, a.n_pop, a.max_fes, a.alpha, a.test_size)
        one = time.time() - t
        print(f"one run ({al}/{d}/{c}) took {one:.1f}s; projected serial total "
              f"{one * len(jobs) / 3600:.1f} h, with n_jobs={a.n_jobs}: {one * len(jobs) / a.n_jobs / 3600:.1f} h "
              "(rough: dataset sizes differ)")
        return

    curves = {}
    if os.path.exists(a.curves_out):
        with np.load(a.curves_out, allow_pickle=False) as z:
            curves = {k: z[k] for k in z.files}

    def work(j):
        d, c, al, s = j
        row, curve = run_one(d, *data[d], al, c, s, a.transfer, a.n_pop, a.max_fes, a.alpha, a.test_size)
        return row, f"{al}__{d}__{c}__{s}__{a.transfer}", curve

    chunk = max(a.n_jobs * 4, 1)
    for i in range(0, len(jobs), chunk):
        out = Parallel(n_jobs=a.n_jobs)(delayed(work)(j) for j in jobs[i:i + chunk])
        pd.DataFrame([o[0] for o in out]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
        for _, key, curve in out:
            curves[key] = curve
        np.savez_compressed(a.curves_out, **curves)
        print(f"  {min(i + chunk, len(jobs))}/{len(jobs)} done", flush=True)


if __name__ == "__main__":
    main()