"""
diversity_report.py
-------------------
Summarize and plot the output of diversity_experiment.py.

  python diversity_report.py --s_csv diversity_s.csv --s_npz diversity_s.npz \
      [--v_csv diversity_v.csv --v_npz diversity_v.npz] --out_prefix div

Produces:
  <prefix>_curves.png   median mask-diversity vs generation (25-75% band), one panel per dataset
  <prefix>_summary.csv  per algorithm: diversity at first/mid/last window, mean diversity, final AUC
  prints the S vs V comparison (paired by dataset, classifier, seed) when V files are given.
Diversity = mean pairwise Hamming distance between masks evaluated in a window of n_pop
evaluations, divided by n_features (0 identical, about 0.5 random).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"fno": "FNO", "ga": "GA", "pso": "PSO", "de": "DE", "cdo": "CDO"}


def load(csv, npz):
    df = pd.read_csv(csv)
    z = np.load(npz, allow_pickle=False)
    curves = {k: z[k] for k in z.files}
    return df, curves


def plot_curves(df, curves, transfer, out):
    dss = sorted(df.dataset.unique())
    algos = [a for a in ["fno", "pso", "cdo", "de", "ga"] if a in set(df.algorithm)]
    fig, axes = plt.subplots(1, len(dss), figsize=(4.6 * len(dss), 3.6), squeeze=False)
    cols = plt.cm.tab10.colors
    for k, ds in enumerate(dss):
        ax = axes[0][k]
        for j, al in enumerate(algos):
            keys = [f"{al}__{ds}__{c}__{s}__{transfer if al != 'ga' else transfer}" for c, s in
                    df[(df.algorithm == al) & (df.dataset == ds)][["classifier", "seed"]].itertuples(index=False)]
            runs = [curves[key] for key in keys if key in curves]
            if not runs:
                continue
            m = min(len(r) for r in runs)
            arr = np.vstack([r[:m] for r in runs])
            x = np.arange(1, m + 1)
            ax.plot(x, np.nanmedian(arr, 0), color=cols[j], lw=1.4, label=f"{LABELS[al]} (n={len(runs)})")
            lo, hi = np.nanpercentile(arr, [25, 75], axis=0)
            ax.fill_between(x, lo, hi, color=cols[j], alpha=0.15, lw=0)
        ax.set_title(ds.upper(), fontsize=10)
        ax.set_xlabel("Generation (window of n_pop evaluations)", fontsize=9)
        if k == 0:
            ax.set_ylabel("Normalized mean pairwise Hamming distance", fontsize=9)
            ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s_csv", required=True)
    ap.add_argument("--s_npz", required=True)
    ap.add_argument("--v_csv")
    ap.add_argument("--v_npz")
    ap.add_argument("--out_prefix", default="div")
    a = ap.parse_args()

    s, sc = load(a.s_csv, a.s_npz)
    plot_curves(s, sc, "s-shape", f"{a.out_prefix}_curves.png")
    summ = s.groupby("algorithm")[["div_first", "div_mid", "div_last", "div_mean", "distinct_frac", "roc_auc", "n_selected"]].mean().round(4)
    summ.to_csv(f"{a.out_prefix}_summary.csv")
    print("S-shape summary (mean over runs):"); print(summ.to_string())

    if a.v_csv and a.v_npz:
        v, vc = load(a.v_csv, a.v_npz)
        plot_curves(v, vc, "v-shape", f"{a.out_prefix}_curves_vshape.png")
        key = ["algorithm", "dataset", "classifier", "seed"]
        j = s.merge(v, on=key, suffixes=("_s", "_v"))
        j["d_auc"] = j.roc_auc_v - j.roc_auc_s
        j["d_div_last"] = j.div_last_v - j.div_last_s
        print("\nV minus S, paired by dataset/classifier/seed (mean; positive = V higher):")
        print(j.groupby("algorithm")[["d_auc", "d_div_last"]].mean().round(4).to_string())
        from scipy.stats import wilcoxon
        for al, g in j.groupby("algorithm"):
            if len(g) >= 6 and (g.d_auc != 0).any():
                print(f"  {al}: Wilcoxon on AUC difference p={wilcoxon(g.d_auc).pvalue:.3f} (n={len(g)})")


if __name__ == "__main__":
    main()