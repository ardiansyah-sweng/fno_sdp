"""
stats_analysis.py
------------------
Statistical comparison of algorithms' results from batch_runner.py's output
CSV. Methodology follows:

  - The FNO paper itself (Taheri et al. 2026, Sect. 3.5): a "multiple-problem
    Wilcoxon signed-rank test" (Corder & Foreman 2009) comparing the
    proposed algorithm against each baseline, aggregated across all problem
    instances.
  - Demšar (2006), "Statistical Comparisons of Classifiers over Multiple
    Data Sets," JMLR 7:1-30: Friedman test (omnibus, rank-based, multiple
    algorithms x multiple problems) + Holm's step-down procedure as
    post-hoc when comparing ONE control algorithm against several others
    (our case: FNO is the proposed/control algorithm, compared against
    PSO/DE/GA/CDO) -- Demšar explicitly recommends Holm over Nemenyi for
    this control-vs-rest scenario, since it's more powerful.
  - Arcuri & Briand (2011, ICSE; extended 2014, STVR "A Hitchhiker's guide
    to statistical tests..."): Vargha-Delaney A-hat-12 effect size
    alongside significance tests for randomized-algorithm comparisons in
    software engineering -- a p-value alone doesn't convey practical
    magnitude, which matters given our sample sizes (many seeds) can make
    even trivial differences "significant".

WHAT COUNTS AS ONE "PROBLEM INSTANCE": a (dataset, classifier) pair (e.g.
"cm1+knn"). With this study's 12 datasets x 3 classifiers, that's 36
problem instances. Per problem instance, each algorithm has N seed
repetitions (raw samples) and one mean value (used for the
rank-based/aggregate tests, matching how the original FNO paper and
Demšar's framework both treat "one dataset = one observation per
algorithm, typically the mean over repeated runs").

COMPLETENESS CHECK: Wilcoxon/Friedman require paired, complete, equal-N
samples across algorithms per problem instance. This module checks seed-
count completeness BEFORE running any test and refuses (with a clear,
specific message) rather than silently computing on unbalanced data --
important since this may be pointed at results from a still-running
batch_runner.py session on the user's own machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from scipy.stats import wilcoxon, friedmanchisquare, mannwhitneyu


# --------------------------------------------------------------------- #
# Loading + completeness
# --------------------------------------------------------------------- #

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # numeric columns may have been written as "n/a" for baseline_all rows
    # (n_pop, max_fes, transfer) -- coerce what's needed for analysis.
    df["seed"] = df["seed"].astype(int)
    return df


@dataclass
class CompletenessReport:
    is_complete: bool
    expected_seeds: int
    problems_checked: int
    missing: list  # list of (algorithm, dataset, classifier, n_found) tuples

    def __str__(self):
        if self.is_complete:
            return (f"COMPLETE: all {self.problems_checked} (algorithm, dataset, classifier) "
                    f"combinations have exactly {self.expected_seeds} seeds.")
        lines = [f"INCOMPLETE ({len(self.missing)} of {self.problems_checked} combinations "
                 f"below {self.expected_seeds} seeds):"]
        for algo, ds, clf, n in self.missing[:20]:
            lines.append(f"  {algo}/{ds}/{clf}: {n}/{self.expected_seeds} seeds")
        if len(self.missing) > 20:
            lines.append(f"  ... and {len(self.missing) - 20} more")
        return "\n".join(lines)


def check_completeness(df: pd.DataFrame, algorithms: list, datasets: list,
                        classifiers: list, expected_seeds: int) -> CompletenessReport:
    """Verify every (algorithm, dataset, classifier) combination has exactly
    `expected_seeds` rows before any statistical test is run. Call this
    first, always -- see module docstring for why."""
    missing = []
    n_checked = 0
    for algo in algorithms:
        for ds in datasets:
            for clf in classifiers:
                n_checked += 1
                n_found = len(df[(df.algorithm == algo) & (df.dataset == ds) & (df.classifier == clf)])
                if n_found != expected_seeds:
                    missing.append((algo, ds, clf, n_found))
    return CompletenessReport(
        is_complete=(len(missing) == 0),
        expected_seeds=expected_seeds,
        problems_checked=n_checked,
        missing=missing,
    )


def require_complete(df: pd.DataFrame, algorithms: list, datasets: list,
                      classifiers: list, expected_seeds: int):
    """Raises with a clear message if incomplete. Use this as a guard at the
    top of any analysis function that assumes balanced, paired data."""
    report = check_completeness(df, algorithms, datasets, classifiers, expected_seeds)
    if not report.is_complete:
        raise ValueError(
            "Cannot run statistical tests on incomplete/unbalanced data.\n" + str(report) +
            "\n\nThis likely means batch_runner.py is still running, or was interrupted. "
            "Wait for it to finish (or resume it) before running this analysis, or pass "
            "a smaller --seeds count / --datasets/--classifiers subset that IS complete."
        )


# --------------------------------------------------------------------- #
# Effect size
# --------------------------------------------------------------------- #

def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """A-hat-12 (Vargha & Delaney 2000): probability that a randomly chosen
    value from x is greater than a randomly chosen value from y, plus half
    the probability of a tie. 0.5 = no difference; >0.5 means x tends to be
    larger. Computed via the Mann-Whitney U statistic: A12 = U / (n_x * n_y).

    Interpretation thresholds (Vargha & Delaney 2000, commonly cited):
    negligible <0.56, small 0.56-0.64, medium 0.64-0.71, large >0.71 (and
    symmetric below 0.5: negligible >0.44, small 0.36-0.44, medium 0.29-0.36,
    large <0.29).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
        return 0.5  # both constant and identical -- no difference, avoid mannwhitneyu's zero-variance error
    try:
        u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
        return u_stat / (len(x) * len(y))
    except ValueError:
        # mannwhitneyu raises if all values in both groups are identical
        return 0.5


def a12_magnitude(a12: float) -> str:
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "negligible"
    elif d < 0.14:
        return "small"
    elif d < 0.21:
        return "medium"
    else:
        return "large"


# --------------------------------------------------------------------- #
# Per-problem-instance Wilcoxon (paired by seed) -- matches the style of
# the FNO paper's own per-function comparison before aggregation.
# --------------------------------------------------------------------- #

def wilcoxon_per_problem(df: pd.DataFrame, metric: str, control_algo: str,
                          baseline_algo: str, datasets: list, classifiers: list,
                          alpha: float = 0.05) -> pd.DataFrame:
    """For each (dataset, classifier) problem instance, a paired Wilcoxon
    signed-rank test between control_algo and baseline_algo on `metric`,
    paired by seed. Returns one row per problem instance."""
    rows = []
    for ds in datasets:
        for clf in classifiers:
            c = df[(df.algorithm == control_algo) & (df.dataset == ds) & (df.classifier == clf)]
            b = df[(df.algorithm == baseline_algo) & (df.dataset == ds) & (df.classifier == clf)]
            c = c.sort_values("seed")
            b = b.sort_values("seed")
            assert list(c.seed) == list(b.seed), (
                f"Seed mismatch for {ds}/{clf}: {control_algo} vs {baseline_algo} -- "
                f"this should have been caught by require_complete()."
            )
            x = c[metric].values
            y = b[metric].values
            diff = x - y
            if np.all(diff == 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = wilcoxon(x, y, zero_method="wilcox")
            a12 = vargha_delaney_a12(x, y)
            rows.append({
                "dataset": ds, "classifier": clf,
                f"{control_algo}_mean": x.mean(), f"{baseline_algo}_mean": y.mean(),
                "wilcoxon_stat": stat, "p_value": p,
                "significant": p < alpha,
                "favors": control_algo if x.mean() > y.mean() else baseline_algo,
                "a12": a12, "a12_magnitude": a12_magnitude(a12),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Aggregate multiple-problem Wilcoxon (matches FNO paper's Table 9-10 style)
# --------------------------------------------------------------------- #

def multiple_problem_wilcoxon(df: pd.DataFrame, metric: str, control_algo: str,
                               baseline_algo: str, datasets: list, classifiers: list) -> dict:
    """Aggregate test: for each of the (dataset x classifier) problem
    instances, take the MEAN metric value per algorithm, then run ONE
    paired Wilcoxon signed-rank test across all problem instances --
    exactly the "multiple-problem Wilcoxon test" convention used in the
    FNO paper itself (Sect. 3.5, Tables 9-10; Corder & Foreman 2009)."""
    c_means, b_means = [], []
    for ds in datasets:
        for clf in classifiers:
            c = df[(df.algorithm == control_algo) & (df.dataset == ds) & (df.classifier == clf)][metric]
            b = df[(df.algorithm == baseline_algo) & (df.dataset == ds) & (df.classifier == clf)][metric]
            c_means.append(c.mean())
            b_means.append(b.mean())
    c_means, b_means = np.array(c_means), np.array(b_means)
    diff = c_means - b_means

    r_plus = diff[diff > 0].sum() if np.any(diff > 0) else 0.0   # informal signed-rank-style summary
    r_minus = -diff[diff < 0].sum() if np.any(diff < 0) else 0.0
    if np.all(diff == 0):
        stat, p = np.nan, 1.0
    else:
        stat, p = wilcoxon(c_means, b_means, zero_method="wilcox")
    a12 = vargha_delaney_a12(c_means, b_means)

    return {
        "control": control_algo, "baseline": baseline_algo, "metric": metric,
        "n_problems": len(diff),
        f"{control_algo}_wins": int(np.sum(diff > 0)),
        f"{baseline_algo}_wins": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
        "wilcoxon_stat": stat, "p_value": p,
        "a12": a12, "a12_magnitude": a12_magnitude(a12),
    }


# --------------------------------------------------------------------- #
# Friedman test (omnibus) + Holm post-hoc (Demšar 2006)
# --------------------------------------------------------------------- #

def friedman_test(df: pd.DataFrame, metric: str, algorithms: list,
                   datasets: list, classifiers: list) -> dict:
    """Friedman test across all `algorithms`, using the MEAN `metric` value
    per (dataset, classifier) problem instance as each algorithm's single
    observation per problem (Demšar 2006 convention). Also returns each
    algorithm's average rank across problems (lower rank = better if
    metric is an error; for accuracy/F1/AUC where higher is better, ranks
    are computed on the NEGATED metric so rank 1 = best, consistent with
    Demšar's convention of rank 1 being best)."""
    problem_means = {algo: [] for algo in algorithms}
    for ds in datasets:
        for clf in classifiers:
            for algo in algorithms:
                v = df[(df.algorithm == algo) & (df.dataset == ds) & (df.classifier == clf)][metric].mean()
                problem_means[algo].append(v)

    samples = [np.array(problem_means[a]) for a in algorithms]
    stat, p = friedmanchisquare(*samples)

    # Average ranks: rank 1 = best (highest metric value, since higher is
    # better for accuracy/f1/roc_auc)
    stacked = np.vstack(samples)  # shape (n_algorithms, n_problems)
    ranks = np.apply_along_axis(lambda col: pd.Series(-col).rank(method="average").values, 0, stacked)
    avg_ranks = ranks.mean(axis=1)

    return {
        "algorithms": algorithms,
        "friedman_stat": stat, "p_value": p,
        "avg_ranks": dict(zip(algorithms, avg_ranks)),
        "n_problems": stacked.shape[1],
    }


def holm_correction(p_values: list) -> list:
    """Holm's step-down procedure (Holm 1979, as recommended by Demšar 2006
    for comparing one control algorithm against several others). Returns
    adjusted p-values in the SAME order as the input list.
    """
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    k = len(p_values)
    adjusted = np.empty(k)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (k - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def holm_posthoc(df: pd.DataFrame, metric: str, control_algo: str, other_algorithms: list,
                  datasets: list, classifiers: list, alpha: float = 0.05) -> pd.DataFrame:
    """Control-vs-rest post-hoc: a Wilcoxon test per (control_algo vs each
    algorithm in other_algorithms), aggregated across all problem instances
    (via multiple_problem_wilcoxon), with Holm-corrected p-values across
    the whole family of comparisons."""
    results = [multiple_problem_wilcoxon(df, metric, control_algo, b, datasets, classifiers)
               for b in other_algorithms]
    raw_p = [r["p_value"] for r in results]
    adj_p = holm_correction(raw_p)
    for r, ap in zip(results, adj_p):
        r["p_value_holm_adjusted"] = ap
        r["significant_after_holm"] = ap < alpha
    return pd.DataFrame(results)


# --------------------------------------------------------------------- #
# Descriptive summary table
# --------------------------------------------------------------------- #

def summary_table(df: pd.DataFrame, metrics: list = None) -> pd.DataFrame:
    """Mean +/- std of each metric, grouped by (algorithm, dataset, classifier)."""
    if metrics is None:
        metrics = ["fitness", "accuracy", "f1_macro", "roc_auc", "n_selected"]
    agg = {m: ["mean", "std"] for m in metrics}
    grouped = df.groupby(["algorithm", "dataset", "classifier"]).agg(agg)
    grouped.columns = ["_".join(c) for c in grouped.columns]
    return grouped.reset_index()



def collapse_to_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Relabel every row's classifier as 'all' so that downstream functions
    (called with classifiers=['all']) average over classifiers and seeds per
    dataset. Blocks are then datasets, which are independent; the 36
    (dataset, classifier) pairs are not (three share each dataset)."""
    out = df.copy()
    out["classifier"] = "all"
    return out


def friedman_zdiff_holm(avg_ranks: dict, n_blocks: int, control: str) -> pd.DataFrame:
    """Demsar (2006) control-vs-rest post hoc on Friedman ranks:
    z = (R_j - R_c) / sqrt(k(k+1)/(6N)), two-sided normal p, Holm-adjusted."""
    from scipy.stats import norm
    k = len(avg_ranks)
    se = np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    rows = [(a, 2 * (1 - norm.cdf(abs((r - avg_ranks[control]) / se))))
            for a, r in avg_ranks.items() if a != control]
    p = [x[1] for x in rows]
    adj = holm_correction(p)
    return pd.DataFrame({"baseline": [x[0] for x in rows], "p_value": p, "p_value_holm_adjusted": adj})


def bootstrap_diff_ci(df: pd.DataFrame, metric: str, control: str, other: str,
                      n_boot: int = 10000, seed: int = 0) -> tuple:
    """Mean difference (control minus other) of per-dataset mean metric, with
    a percentile bootstrap 95% interval over datasets."""
    m = df.groupby(["dataset", "algorithm"])[metric].mean().unstack()
    d = (m[control] - m[other]).values
    rng = np.random.default_rng(seed)
    bs = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return float(d.mean()), tuple(np.percentile(bs, [2.5, 97.5]))

# --------------------------------------------------------------------- #
# Orchestration: full report
# --------------------------------------------------------------------- #

def full_report(csv_path: str, metric: str = "roc_auc", control_algo: str = "fno",
                 algorithms: list = None, datasets: list = None, classifiers: list = None,
                 expected_seeds: int = None, alpha: float = 0.05, verbose: bool = True,
                 level: str = "dataset"):
    """Runs the complete pipeline: completeness check -> descriptive summary
    -> Friedman omnibus test -> Holm-corrected control-vs-rest post-hoc ->
    per-problem Wilcoxon+A12 detail tables. Returns a dict of all results;
    prints a readable report if verbose=True.

    Raises ValueError (via require_complete) if the data is incomplete --
    this is intentional, see module docstring.
    """
    df = load_results(csv_path)

    algorithms = algorithms or sorted(df.algorithm.unique())
    datasets = datasets or sorted(df.dataset.unique())
    classifiers = classifiers or sorted(df.classifier.unique())
    if expected_seeds is None:
        expected_seeds = df.groupby(["algorithm", "dataset", "classifier"]).size().max()

    require_complete(df, algorithms, datasets, classifiers, expected_seeds)

    other_algorithms = [a for a in algorithms if a != control_algo]

    summary = summary_table(df)
    raw_df, raw_classifiers = df, classifiers
    if level == "dataset":   # primary analysis: independent blocks = datasets
        df, classifiers = collapse_to_datasets(df), ["all"]
    friedman = friedman_test(df, metric, algorithms, datasets, classifiers)
    posthoc = holm_posthoc(df, metric, control_algo, other_algorithms, datasets, classifiers, alpha)
    per_problem = {
        b: wilcoxon_per_problem(df, metric, control_algo, b, datasets, classifiers, alpha)
        for b in other_algorithms
    }

    if verbose:
        print(f"=== Statistical report: metric={metric}, control={control_algo} ===")
        print(f"Algorithms: {algorithms}")
        print(f"Problem instances: {len(datasets)} datasets x {len(classifiers)} classifiers "
              f"= {len(datasets)*len(classifiers)}, {expected_seeds} seeds each\n")

        print("--- Friedman omnibus test ---")
        print(f"chi2={friedman['friedman_stat']:.4f}, p={friedman['p_value']:.6f}")
        print("Average ranks (1=best):")
        for algo, rank in sorted(friedman["avg_ranks"].items(), key=lambda x: x[1]):
            marker = " <- control" if algo == control_algo else ""
            print(f"  {algo}: {rank:.3f}{marker}")
        print()

        print(f"--- Holm-corrected post-hoc: {control_algo} vs each baseline ---")
        print(posthoc[["baseline", f"{control_algo}_wins", "p_value",
                        "p_value_holm_adjusted", "significant_after_holm",
                        "a12", "a12_magnitude"]].to_string(index=False))
        print()

        for b, table in per_problem.items():
            n_sig_favor_control = ((table.significant) & (table.favors == control_algo)).sum()
            n_sig_favor_baseline = ((table.significant) & (table.favors == b)).sum()
            print(f"--- Per-problem detail: {control_algo} vs {b} ---")
            print(f"  significant in favor of {control_algo}: {n_sig_favor_control}/{len(table)} problems")
            print(f"  significant in favor of {b}: {n_sig_favor_baseline}/{len(table)} problems")

    zholm = friedman_zdiff_holm(friedman["avg_ranks"], friedman["n_problems"], control_algo)
    ci = {b: bootstrap_diff_ci(raw_df, metric, control_algo, b) for b in other_algorithms}
    if verbose:
        print(f"\n--- Friedman z post hoc (Holm), control={control_algo}, N={friedman['n_problems']} blocks ---")
        print(zholm.to_string(index=False))
        print("--- Mean difference and 95% bootstrap CI over datasets ---")
        for b, (d, (lo, hi)) in ci.items():
            print(f"  {control_algo} - {b}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    return {
        "summary": summary,
        "friedman": friedman,
        "posthoc": posthoc,
        "friedman_z_holm": zholm,
        "bootstrap_ci": ci,
        "per_problem": per_problem,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--metric", type=str, default="roc_auc",
                         choices=["accuracy", "f1_macro", "roc_auc", "fitness"])
    parser.add_argument("--control", type=str, default="fno")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--level", choices=["dataset", "problem"], default="dataset",
                         help="dataset (default): blocks are the independent datasets. "
                              "problem: blocks are (dataset, classifier) pairs (sensitivity).")
    args = parser.parse_args()
    full_report(args.csv_path, metric=args.metric, control_algo=args.control, alpha=args.alpha,
                level=args.level)