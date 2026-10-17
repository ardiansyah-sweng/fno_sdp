"""
data_loaders.py
----------------
Shared dataset-loading utilities. Used by both run_experiment.py
(single-run CLI) and batch_runner.py (grid runner) to avoid duplicating
the loading logic.

PRIMARY SOURCE: NASA MDP "D''" cleaned datasets (12 sets), fetched as ARFF
directly from github.com/klainfo/NASADefectDataset (raw.githubusercontent.com,
reachable under a GitHub-only network whitelist, no auth needed).

Why D'' over the earlier PROMISE-22-attribute CSV source (used in an
earlier version of this pipeline for only 5 datasets: cm1/kc1/kc2/jm1/pc1):
  - D'' is the Shepperd, Song, Sun & Mair (2013) cleaning methodology
    (already cited elsewhere in this project) applied by its own
    maintainers -- an established, peer-reviewed process, not our own
    ad-hoc drop_duplicates(). Confirmed directly (2026-09-10): all 12 D''
    files have ZERO missing values and ZERO duplicate rows already, and
    published defect-rate/row-count statistics (cross-checked against a
    secondary source, Suryanto et al., TELKOMNIKA) match exactly.
  - D'' additionally removes "inconsistent instances" (identical feature
    vectors with conflicting class labels) -- a data-quality issue our own
    exact-row drop_duplicates() could not catch, since it only matches
    full rows including the label.
  - Covers 12 of the 13 canonical NASA MDP datasets in ONE consistent
    format/cleaning pipeline: cm1, jm1, kc1, kc3, mc1, mc2, mw1, pc1, pc2,
    pc3, pc4, pc5. (kc2 is NOT available in this D'' collection -- confirmed
    404 on direct fetch -- and is deliberately EXCLUDED here rather than
    pulled from a different cleaning pipeline/format, to keep the whole
    study on one uniform methodology. See module-level NASA_MDP_DATASETS.)
  - Feature set is 22-40 McCabe/Halstead/derived metrics depending on the
    dataset (uppercase names, e.g. CYCLOMATIC_COMPLEXITY, HALSTEAD_EFFORT),
    NOT the smaller 21-attribute PROMISE CSV set used in the earlier
    5-dataset version of this pipeline -- the two are not feature-compatible,
    which is exactly why this switch replaces rather than supplements the
    earlier source.
"""

from __future__ import annotations

import urllib.request
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

NASA_MDP_BASE_URL = "https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/CleanedData/MDP/D%27%27/"
NASA_MDP_DATASETS = ["cm1", "jm1", "kc1", "kc3", "mc1", "mc2", "mw1", "pc1", "pc2", "pc3", "pc4", "pc5"]

# Kept as an alias since earlier code/CLI flags in this project refer to
# "--promise" -- NASA MDP is the specific corpus within the broader PROMISE
# software-engineering-dataset tradition (Sayyad Shirabad & Menzies 2005),
# so the flag name remains accurate even though the underlying source
# changed from PROMISE-CSV-mirror to NASA-MDP-D''-ARFF.
PROMISE_DATASETS = NASA_MDP_DATASETS

# Target attribute name is NOT uniform across files: JM1 uses "label",
# every other D'' file uses "Defective". Values are uniformly {Y,N}
# (confirmed by direct inspection of all 12 files).
_TARGET_ATTR_OVERRIDES = {"jm1": "label"}
_DEFAULT_TARGET_ATTR = "Defective"


def _parse_arff(text: str):
    """Minimal ARFF parser for this dataset family: numeric attributes plus
    one trailing nominal {Y,N} (or similar) class attribute. Avoids
    scipy.io.arff's byte-string/quirk handling since the format here is
    simple and regular -- written directly for full control and easy
    debugging if a future file deviates.
    """
    lines = text.splitlines()
    attr_names = []
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("@attribute"):
            # e.g. "@attribute LOC_TOTAL numeric" or "@attribute Defective {Y,N}"
            parts = stripped.split(None, 2)
            if len(parts) >= 2:
                attr_names.append(parts[1])
        elif low == "@data":
            data_start = i
            break
    if data_start is None:
        raise ValueError("No @data section found in ARFF content")

    rows = []
    for line in lines[data_start + 1:]:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        rows.append([v.strip() for v in s.split(",")])

    df = pd.DataFrame(rows, columns=attr_names)
    return df


def load_nasa_mdp_dataset(name: str):
    """Fetch a NASA MDP D'' cleaned dataset directly from GitHub -- no
    manual download needed. See module docstring for source/rationale."""
    name = name.lower()
    if name not in NASA_MDP_DATASETS:
        raise ValueError(f"Unknown NASA MDP dataset '{name}'. Available: {NASA_MDP_DATASETS}")

    url = NASA_MDP_BASE_URL + f"{name.upper()}.arff"
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    df = _parse_arff(text)
    target_col = _TARGET_ATTR_OVERRIDES.get(name, _DEFAULT_TARGET_ATTR)
    if target_col not in df.columns:
        # Defensive fallback: use the last column (the class attribute is
        # always declared last in these files) if the expected name isn't
        # found, rather than silently mis-selecting a feature as the target.
        target_col = df.columns[-1]

    y = df[target_col].astype(str).str.strip().str.upper().map({"Y": 1, "N": 0}).values
    if np.isnan(y).any():
        bad = df[target_col][pd.isna(y)].unique()
        raise ValueError(f"Unexpected target values in {name}: {bad}")
    y = y.astype(int)

    feature_df = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
    if feature_df.isna().any().any():
        raise ValueError(f"Non-numeric or missing values found in {name}'s feature columns")

    X = feature_df.values.astype(float)
    feature_names = list(feature_df.columns)
    return X, y, feature_names


# Backward-compat name (used by earlier code in this project)
def load_promise_dataset(name: str, deduplicate: bool = True):
    """Alias to load_nasa_mdp_dataset. `deduplicate` is accepted for
    backward compatibility but has no effect: D'' data is already
    deduplicated by its established cleaning methodology (confirmed
    zero duplicate rows across all 12 files on direct inspection)."""
    return load_nasa_mdp_dataset(name)


def load_local_csv(csv_path: str, label_col: str):
    df = pd.read_csv(csv_path)
    y = df[label_col].values
    if y.dtype == object or str(y.dtype).startswith("bool"):
        y = (pd.Series(y).astype(str).str.lower().isin(
            ["true", "yes", "y", "1", "defective"])).astype(int).values
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
    feature_names = list(df.drop(columns=[label_col]).select_dtypes(include=[np.number]).columns)
    return X, y, feature_names


def load_synthetic(seed: int = 0):
    X, y = make_classification(
        n_samples=500, n_features=25, n_informative=8, n_redundant=6, n_repeated=2,
        weights=[0.85, 0.15], flip_y=0.02, random_state=seed,
    )
    feature_names = [f"metric_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


def load_dataset(csv_path: str | None = None, label_col: str | None = None,
                  promise: str | None = None, seed: int = 0, verbose: bool = True,
                  deduplicate: bool = True):
    if promise is not None:
        X, y, names = load_nasa_mdp_dataset(promise)
        if verbose:
            print(f"[NASA-MDP-D'':{promise}] {X.shape[0]} samples, {X.shape[1]} features, "
                  f"defect rate={y.mean():.2%}")
        return X, y, names
    if csv_path is not None:
        if label_col is None:
            raise ValueError("label_col is required when csv_path is given")
        X, y, names = load_local_csv(csv_path, label_col)
        if verbose:
            print(f"[{csv_path}] {X.shape[0]} samples, {X.shape[1]} features, "
                  f"defect rate={np.mean(y):.2%}")
        return X, y, names
    X, y, names = load_synthetic(seed=seed)
    if verbose:
        print(f"[synthetic] {X.shape[0]} samples, {X.shape[1]} features, "
              f"defect rate={y.mean():.2%} (replace with --promise or --csv for real data)")
    return X, y, names


def stratified_holdout_split(X: np.ndarray, y: np.ndarray, seed: int, test_size: float = 0.3):
    """Split data into a SEARCH partition (train) and a HELD-OUT test partition,
    stratified on the label so class balance is preserved in both.

    This exists to avoid the selection-bias/CV-leakage problem documented in
    Ambroise & McLachlan (2002, PNAS 99(10):6562-6566) and Cawley & Talbot
    (2010, JMLR 11): using the SAME cross-validation folds to both (a) guide
    a wrapper feature-selection search and (b) report that search's final
    performance produces optimistically biased results.

    The held-out test partition returned here must NEVER be touched by the
    fitness function driving the search -- only by a one-shot final
    evaluation after search completes (see sdp_fitness.evaluate_mask_holdout).

    Reusing `seed` for both the search's own randomness AND this split means
    each independent run (see batch_runner.py's --seeds) evaluates on a
    genuinely different train/test partition too -- similar in spirit to
    repeated random sub-sampling / Monte Carlo cross-validation.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)