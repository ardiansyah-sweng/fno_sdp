"""
check_results.py
-----------------
Quick completeness check for results_all.csv. Run from the same folder as
your batch_runner.py / stats_analysis.py / results_all.csv:

    python check_results.py

(On Windows, use `python`, not `python3`, unless you specifically installed
a `python3` alias.)
"""

import pandas as pd
from stats_analysis import check_completeness

df = pd.read_csv("results_all.csv")
print("Total baris:", len(df))
print(df.groupby("algorithm").size())
print()

algorithms = ["fno", "ga", "pso", "de", "cdo", "filter_mi", "embedded_dt"]
datasets = sorted(df["dataset"].unique())
classifiers = sorted(df["classifier"].unique())

report = check_completeness(df, algorithms, datasets, classifiers, expected_seeds=30)
print(report)