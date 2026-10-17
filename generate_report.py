"""
generate_report.py
-------------------
Publication-ready figures generated from batch_runner.py's results CSV
(and, for convergence plots, the optional curves/*.npy files saved by
--save_curves).

Figure 1: Critical Difference (CD) diagram (Demsar 2006 style). The bar
length uses the Nemenyi critical difference (the standard convention for
this visualization, used across the vast majority of papers that draw CD
diagrams, including many that report Holm-corrected p-values in their
tables as this study does). This is a deliberate, documented choice: the
paper's actual pairwise significance decisions (Table 2, Section 4.1) use
the more powerful Holm control-versus-rest procedure recommended by Demsar
(2006) for a one-control-versus-several-baselines design, NOT the Nemenyi
all-pairs procedure the diagram's bar length is drawn from. The diagram is
a visual aid for the magnitude of rank differences, not a restatement of
the Holm-based significance decisions; the caption says so explicitly so a
reader cannot mistake one for the other.

Figure 2: Convergence curves (best-so-far fitness vs. evaluation number),
one panel per dataset/classifier combination, one line per search
algorithm. Requires curves/*.npy files (see batch_runner.py --save_curves).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Nemenyi critical value q_alpha (two-tailed, alpha=0.05), for k=2..12
# algorithms (Demsar 2006, Table 5(b), standard published values).
NEMENYI_Q_ALPHA_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
}


def nemenyi_critical_difference(k: int, n_problems: int, alpha: float = 0.05) -> float:
    """CD = q_alpha * sqrt(k(k+1) / (6N)), the standard Nemenyi critical
    difference used for CD-diagram bar lengths (Demsar 2006, Eq. 6)."""
    if k not in NEMENYI_Q_ALPHA_005:
        raise ValueError(f"No tabulated q_alpha for k={k} algorithms; add it to "
                          f"NEMENYI_Q_ALPHA_005 or supply one directly.")
    q_alpha = NEMENYI_Q_ALPHA_005[k]
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_problems))


def compute_avg_ranks(df: pd.DataFrame, metric: str, algorithms: list,
                       datasets: list, classifiers: list) -> dict:
    """Average rank per algorithm (1=best) across all (dataset, classifier)
    problem instances, using each algorithm's mean metric value per problem
    instance -- same convention as stats_analysis.friedman_test."""
    problem_means = {algo: [] for algo in algorithms}
    for ds in datasets:
        for clf in classifiers:
            for algo in algorithms:
                v = df[(df.algorithm == algo) & (df.dataset == ds) & (df.classifier == clf)][metric].mean()
                problem_means[algo].append(v)
    stacked = np.vstack([problem_means[a] for a in algorithms])
    ranks = np.apply_along_axis(lambda col: pd.Series(-col).rank(method="average").values, 0, stacked)
    avg_ranks = dict(zip(algorithms, ranks.mean(axis=1)))
    return avg_ranks, stacked.shape[1]


def plot_cd_diagram(avg_ranks: dict, n_problems: int, control: str = None,
                     alpha: float = 0.05, title: str = None,
                     out_path: str = "cd_diagram.png", label_map: dict = None):
    """Draw a Demsar-style Critical Difference diagram.

    avg_ranks: {algorithm_name: avg_rank}, lower is better.
    control: if given, that algorithm's marker is highlighted (this study's
        proposed method, FNO).
    label_map: optional {internal_name: display_name} for cleaner labels.
    """
    algos = list(avg_ranks.keys())
    k = len(algos)
    cd = nemenyi_critical_difference(k, n_problems, alpha)

    sorted_algos = sorted(algos, key=lambda a: avg_ranks[a])
    ranks_sorted = [avg_ranks[a] for a in sorted_algos]
    labels = [label_map.get(a, a) if label_map else a for a in sorted_algos]

    min_rank, max_rank = min(ranks_sorted), max(ranks_sorted)
    lo = np.floor(min_rank - 0.5)
    hi = np.ceil(max_rank + 0.5)
    x_span = hi - lo
    # Minimum x-separation (in data units) before two labels are considered
    # to collide and one is pushed up a level; tuned for the fontsize used.
    min_sep = x_span * 0.11

    # Greedy level assignment, ALL labels above the axis: walk left to
    # right, place each point one level above the axis (level 0); if it is
    # within min_sep of the most recently placed point AT THAT LEVEL, bump
    # it to the next level up instead.
    levels = []
    last_x_at_level = {}
    for rank in ranks_sorted:
        lvl = 0
        while lvl in last_x_at_level and (rank - last_x_at_level[lvl]) < min_sep:
            lvl += 1
        levels.append(lvl)
        last_x_at_level[lvl] = rank
    n_levels = max(levels) + 1

    axis_y = 0.30
    level_height = 0.16
    top_of_labels = axis_y + (n_levels + 1) * level_height

    fig_height = 1.8 + n_levels * 0.5
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top_of_labels + 0.35)
    ax.axhline(axis_y, color="black", linewidth=1)
    ax.set_xticks(np.arange(lo, hi + 0.01, 1.0))
    ax.tick_params(axis="x", labelsize=10)
    ax.get_yaxis().set_visible(False)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    # CD reference bar, drawn well above the highest label level.
    cd_y = top_of_labels + 0.20
    cd_x0 = lo + 0.15
    ax.plot([cd_x0, cd_x0 + cd], [cd_y, cd_y], color="black", linewidth=1.5)
    ax.plot([cd_x0, cd_x0], [cd_y - 0.04, cd_y + 0.04], color="black", linewidth=1.5)
    ax.plot([cd_x0 + cd, cd_x0 + cd], [cd_y - 0.04, cd_y + 0.04], color="black", linewidth=1.5)
    ax.text(cd_x0 + cd / 2, cd_y + 0.06, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=9)

    # Tick marks + labels, each at its assigned level above the axis.
    for algo, rank, label, lvl in zip(sorted_algos, ranks_sorted, labels, levels):
        y_label = axis_y + (lvl + 1) * level_height
        ax.plot([rank, rank], [axis_y, y_label - 0.03], color="#888888", linewidth=0.8, zorder=1)
        is_control = (control is not None and algo == control)
        ax.text(rank, y_label, f"{label} ({rank:.2f})", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold" if is_control else "normal",
                 color="#B03A2E" if is_control else "black")
        if is_control:
            ax.plot(rank, axis_y, marker="o", color="#B03A2E", markersize=6, zorder=5)

    # Horizontal bar(s) below the axis connecting algorithms not
    # significantly different from each other (within CD of one another).
    groups = []
    i = 0
    while i < len(sorted_algos):
        j = i
        while j + 1 < len(sorted_algos) and (ranks_sorted[j + 1] - ranks_sorted[i]) <= cd:
            j += 1
        if j > i:
            groups.append((i, j))
        i += 1
    y0 = axis_y - 0.10
    for gi, (i, j) in enumerate(groups):
        y = y0 - gi * 0.10
        ax.plot([ranks_sorted[i], ranks_sorted[j]], [y, y], color="black", linewidth=2.5,
                 solid_capstyle="butt")
    if groups:
        ax.set_ylim(y0 - len(groups) * 0.10 - 0.15, top_of_labels + 0.35)

    if title:
        ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Average rank (lower = better)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path, cd


def plot_convergence(curves_dir: str, dataset: str, classifier: str,
                      algorithms: list, seed: int = 0, out_path: str = None,
                      label_map: dict = None):
    """One convergence panel: best-so-far fitness vs. evaluation index, one
    line per algorithm, for a single (dataset, classifier, seed). Requires
    curves/{algorithm}_{dataset}_{classifier}_seed{seed}.npy files."""
    curves_dir = Path(curves_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = 0
    for algo in algorithms:
        path = curves_dir / f"{algo}_{dataset}_{classifier}_seed{seed}.npy"
        if not path.exists():
            continue
        curve = np.load(path)
        label = label_map.get(algo, algo) if label_map else algo
        ax.plot(np.arange(1, len(curve) + 1), curve, label=label, linewidth=1.3)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return None
    ax.set_xlabel("Fitness evaluations", fontsize=10)
    ax.set_ylabel("Best-so-far fitness (lower = better)", fontsize=10)
    ax.set_title(f"{dataset} / {classifier} (seed {seed})", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if out_path is None:
        out_path = f"convergence_{dataset}_{classifier}_seed{seed}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_convergence_grid(curves_dir: str, datasets: list, classifier: str, algorithms: list,
                           seeds: range = range(30), out_path: str = "convergence_grid.png",
                           label_map: dict = None, ncols: int = 2):
    """Aggregate convergence figure: one panel per dataset, one line per
    algorithm, showing the MEDIAN best-so-far fitness over seeds with a
    25th to 75th percentile band. Reads curves/{algo}_{dataset}_{classifier}_seed{k}.npy.
    Fitness here is the inner-CV objective on the search partition (not the
    held-out AUC), and for the sigmoid-binarized optimizers it is evaluated on
    stochastically binarized masks, so curves are noisy by construction."""
    curves_dir = Path(curves_dir)
    n = len(datasets)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.6 * nrows), squeeze=False)
    colors = plt.cm.tab10.colors
    for k, ds in enumerate(datasets):
        ax = axes[k // ncols][k % ncols]
        for j, algo in enumerate(algorithms):
            runs = []
            for sd in seeds:
                p = curves_dir / f"{algo}_{ds}_{classifier}_seed{sd}.npy"
                if p.exists():
                    runs.append(np.load(p))
            if not runs:
                continue
            m = min(len(r) for r in runs)
            arr = np.vstack([r[:m] for r in runs])
            x = np.arange(1, m + 1)
            med = np.median(arr, axis=0)
            lo, hi = np.percentile(arr, [25, 75], axis=0)
            label = (label_map.get(algo, algo) if label_map else algo) + f" (n={len(runs)})"
            ax.plot(x, med, color=colors[j % 10], linewidth=1.4, label=label)
            ax.fill_between(x, lo, hi, color=colors[j % 10], alpha=0.15, linewidth=0)
        ax.set_title(f"{ds.upper()} / {classifier}", fontsize=10)
        ax.set_xlabel("Fitness evaluations", fontsize=9)
        ax.set_ylabel("Best-so-far fitness", fontsize=9)
        ax.grid(alpha=0.25)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--metric", type=str, default="roc_auc")
    parser.add_argument("--control", type=str, default="fno")
    parser.add_argument("--out", type=str, default="cd_diagram.png")
    parser.add_argument("--level", choices=["dataset", "problem"], default="dataset",
                        help="dataset: one block per dataset (mean over classifiers and seeds; "
                             "independent blocks, the primary analysis). problem: one block per "
                             "(dataset, classifier) pair (sensitivity analysis; blocks are correlated).")
    parser.add_argument("--curves_dir", type=str, default=None,
                         help="If given, also generate convergence plots from this directory.")
    parser.add_argument("--conv_dataset", type=str, default=None)
    parser.add_argument("--conv_classifier", type=str, default=None)
    parser.add_argument("--conv_seed", type=int, default=0)
    parser.add_argument("--conv_grid", type=str, default=None,
                        help="comma-separated datasets for the aggregate median+IQR convergence "
                             "figure over all seeds (needs --curves_dir and --conv_classifier)")
    parser.add_argument("--conv_out", type=str, default="convergence_grid.png")
    args = parser.parse_args()

    LABELS = {"fno": "FNO", "ga": "GA", "pso": "PSO", "de": "DE", "cdo": "CDO",
              "filter_mi": "Filter-MI", "embedded_dt": "Embedded-DT",
              "baseline_all": "No-FS"}

    df = pd.read_csv(args.csv_path)
    if args.level == "dataset":
        df["classifier"] = "all"  # collapse classifiers: blocks = datasets
    algorithms = sorted(df["algorithm"].unique())
    datasets = sorted(df["dataset"].unique())
    classifiers = sorted(df["classifier"].unique())

    avg_ranks, n_problems = compute_avg_ranks(df, args.metric, algorithms, datasets, classifiers)
    path, cd = plot_cd_diagram(
        avg_ranks, n_problems, control=args.control,
        title=f"Critical Difference diagram ({args.metric.upper()}, N={n_problems} {'datasets' if args.level == 'dataset' else 'problem instances'})",
        out_path=args.out, label_map=LABELS,
    )
    print(f"Saved {path} (Nemenyi CD = {cd:.4f}, k={len(algorithms)}, N={n_problems})")
    for a, r in sorted(avg_ranks.items(), key=lambda x: x[1]):
        print(f"  {LABELS.get(a,a)}: {r:.3f}")

    if args.curves_dir and args.conv_dataset and args.conv_classifier:
        search_algos = [a for a in algorithms if a != "baseline_all"]
        out = plot_convergence(args.curves_dir, args.conv_dataset, args.conv_classifier,
                                search_algos, seed=args.conv_seed, label_map=LABELS)
        if out:
            print(f"Saved {out}")
        else:
            print("No curve files found for the requested dataset/classifier/seed.")

    if args.conv_grid and args.curves_dir and args.conv_classifier:
        wrappers = [a for a in ["fno", "ga", "de", "pso", "cdo"]]
        out = plot_convergence_grid(args.curves_dir, args.conv_grid.split(","), args.conv_classifier,
                                    wrappers, out_path=args.conv_out, label_map=LABELS)
        print(f"Saved {out}")