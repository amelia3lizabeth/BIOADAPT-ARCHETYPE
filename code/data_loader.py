"""
BIOADAPT-ARCHETYPE: Data Loading & Normalisation
================================================
Leak-free data loading with within-cohort z-score normalisation.

KEY PRINCIPLE: All normalisation parameters (means, stds) are fit on
TRAINING patients only and applied to test. No information from the
test set leaks into preprocessing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import config as cfg


def load_cohort(cohort_id: str, cohorts: list = None) -> pd.DataFrame:
    """
    Load a single cohort's raw data file.

    Returns a DataFrame with:
        - one row per patient
        - gene columns (lowercase)
        - 'response' column (0/1)
        - 'cohort' column (cohort_id)
        - '_normalisation' column (RPKM, FPKM, or zscored)
    """
    path = cfg.COHORT_FILES[cohort_id]
    df = pd.read_csv(path)
    return df


def load_pooled(cohort_ids: list = None, common_genes_only: bool = True) -> tuple:
    """
    Load multiple cohorts, return pooled expression matrix.

    Args:
        cohort_ids: list of cohort IDs to load (default: PRIMARY_COHORTS)
        common_genes_only: if True, restrict to genes present in ALL loaded cohorts

    Returns:
        X        : (n_patients, n_genes) raw expression matrix (log-transformed if needed)
        y        : (n_patients,) response labels (0/1)
        cohort   : (n_patients,) cohort labels
        genes    : list of gene names matching column order
        norm_state: dict mapping cohort -> normalisation state (RPKM/FPKM/zscored)

    NOTE: Within-cohort z-score is NOT applied here. That happens inside
          the CV loop on training patients only. See `apply_train_normalisation`.
    """
    if cohort_ids is None:
        cohort_ids = cfg.PRIMARY_COHORTS

    dfs = [load_cohort(cid) for cid in cohort_ids]

    # Find common genes
    meta_cols = {'response', 'cohort', '_normalisation'}
    gene_sets = [set(c for c in df.columns if c not in meta_cols) for df in dfs]
    if common_genes_only:
        common = sorted(set.intersection(*gene_sets))
    else:
        common = sorted(set.union(*gene_sets))

    if cfg.VERBOSE:
        print(f"Loaded {len(dfs)} cohorts | {len(common)} common genes")

    X_list, y_list, c_list, norm_state = [], [], [], {}
    for cid, df in zip(cohort_ids, dfs):
        norm = df['_normalisation'].iloc[0]
        norm_state[cid] = norm
        # Get expression — fill missing genes with NaN if non-common requested
        expr = df.reindex(columns=common).astype(float).values
        # Apply log2(x+1) for FPKM/RPKM cohorts
        if norm in ('RPKM', 'FPKM'):
            expr = np.log2(np.where(np.isnan(expr), 0, expr) + 1)
        # Already z-scored cohorts: leave as-is
        X_list.append(expr)
        y_list.append(df['response'].astype(int).values)
        c_list.extend([cid] * len(df))
        if cfg.VERBOSE:
            print(f"  {cid}: n={len(df):>3}, R={df['response'].sum():>3}, "
                  f"NR={(df['response']==0).sum():>3}, norm={norm}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    cohort = np.array(c_list)

    return X, y, cohort, common, norm_state


def fit_train_normalisation(X_train: np.ndarray, cohort_train: np.ndarray) -> dict:
    """
    Fit within-cohort z-score parameters on training patients only.

    Returns dict mapping cohort_id -> (mean_vector, std_vector).
    """
    params = {}
    for cid in np.unique(cohort_train):
        mask = cohort_train == cid
        if mask.sum() == 0:
            continue
        mu  = X_train[mask].mean(axis=0)
        sig = X_train[mask].std(axis=0)
        sig = np.where(sig < 1e-8, 1.0, sig)
        params[cid] = (mu, sig)
    return params


def apply_train_normalisation(X: np.ndarray, cohort: np.ndarray,
                                params: dict) -> np.ndarray:
    """
    Apply within-cohort z-score using parameters fit on training data.
    Patients from cohorts not in `params` get the global training mean/std
    (handles edge cases like LOCO with held-out cohort).
    """
    X_out = X.copy()
    for cid in np.unique(cohort):
        mask = cohort == cid
        if cid in params:
            mu, sig = params[cid]
        else:
            # Held-out cohort: use mean across all training cohorts
            all_mu  = np.mean([p[0] for p in params.values()], axis=0)
            all_sig = np.mean([p[1] for p in params.values()], axis=0)
            mu, sig = all_mu, all_sig
        X_out[mask] = (X[mask] - mu) / sig
    return X_out


def fit_variance_filter(X_train: np.ndarray, percentile: float = 5,
                          cohort_train: np.ndarray = None,
                          norm_state: dict = None) -> np.ndarray:
    """
    Return boolean mask of genes to KEEP (above the percentile-th percentile
    of variance). Default 5 keeps 95% of genes.

    IMPORTANT: For per-cohort z-scored data, this filter must operate on
    log-transformed PRE-NORMALISATION values, because z-scoring forces every
    gene to variance≈1 and the filter becomes meaningless.

    If `cohort_train` and `norm_state` are provided, the filter computes
    variance on within-cohort log values (more biologically meaningful).
    Otherwise it falls back to plain variance on whatever X_train is.

    HSFX1 sits at ~15th percentile of pre-normalisation log variance,
    so percentile=5 preserves it.
    """
    if cohort_train is not None and norm_state is not None:
        # Compute variance per cohort then average — captures within-cohort
        # biological variability without the z-score collapse
        cohort_vars = []
        for cid in np.unique(cohort_train):
            mask = cohort_train == cid
            if mask.sum() < 2:
                continue
            cohort_vars.append(X_train[mask].var(axis=0))
        var = np.mean(cohort_vars, axis=0)
    else:
        var = X_train.var(axis=0)

    cutoff = np.percentile(var, percentile)
    return var >= cutoff


def compute_axis_scores(X: np.ndarray, genes: list,
                         axis_definitions: dict = None) -> tuple:
    """
    Compute composite axis scores by averaging member genes.

    Returns:
        axis_matrix : (n_patients, n_axes)
        axis_names  : list of axis names
        n_present   : dict mapping axis -> number of constituent genes found
    """
    if axis_definitions is None:
        axis_definitions = cfg.AXIS_DEFINITIONS

    gene_idx = {g: i for i, g in enumerate(genes)}
    axis_names = list(axis_definitions.keys())
    axis_data = np.zeros((X.shape[0], len(axis_names)))
    n_present = {}

    for ai, (name, members) in enumerate(axis_definitions.items()):
        present = [m for m in members if m in gene_idx]
        n_present[name] = len(present)
        if not present:
            raise ValueError(f"Axis '{name}': no genes found in dataset")
        cols = [gene_idx[m] for m in present]
        axis_data[:, ai] = X[:, cols].mean(axis=1)

    return axis_data, axis_names, n_present


if __name__ == '__main__':
    # Smoke test
    X, y, cohort, genes, norm = load_pooled()
    print(f"\nPooled: {X.shape}, response: {y.sum()}/{len(y)}")
    print(f"Cohorts: {dict(zip(*np.unique(cohort, return_counts=True)))}")
    print(f"Normalisation states: {norm}")

    # Compute axis scores
    axes, axis_names, n_pres = compute_axis_scores(X, genes)
    print(f"\nAxis scores: {axes.shape}")
    for name, n in n_pres.items():
        print(f"  {name}: {n} member genes found")
