"""
bdiff.py — Python replication of Stata's ``bdiff`` command (with parallel support)
====================================================================================
Tests whether regression coefficients differ across two subgroups.

Original Stata command
----------------------
    bdiff — Coefficient Difference Test across Two Groups
    Version 1.04, 24 Nov 2020
    Author:  Yujun Lian, Sun Yat-sen University
    Email:   arlionn@163.com
    Blog:    https://www.lianxh.cn

    This Python package is a reimplementation of the above Stata command.
    All methodological credit belongs to the original author.

Python implementation
---------------------
    Zhiyu Lu, Central University of Finance and Economics
    Email:   zhiyu.lu.econ@icloud.com
    Web:     https://zhiyu-lu.com

References
----------
Efron, B., Tibshirani, R., 1993.
    *An Introduction to the Bootstrap*. Chapman & Hall.

Three modes
-----------
1. Permutation test (default)  — non-parametric, label-shuffling
2. Bootstrap test  (method="bootstrap")  — non-parametric, resampling
3. Wald test       (method="wald")       — parametric, block-diagonal VCV
                                           (equivalent to suest for non-overlapping samples)

Parallelism
-----------
Set n_jobs=-1 to use all CPU cores, or n_jobs=4 for a fixed number.
Parallelism is applied to the resampling loop (permutation / bootstrap).
The Wald test is already instant and needs no parallelism.

Reproducibility
---------------
Uses numpy.random.SeedSequence to split the master seed into per-worker
child seeds, so results are identical regardless of n_jobs.

Dependencies
------------
    pip install pyfixest pandas numpy joblib tqdm scipy

Usage
-----
    from pybdiff import bdiff

    result = bdiff(
        df      = df,
        group   = "treated",
        formula = "y ~ x1 + x2 | firm + year",
        vcov    = {"CRV1": "firm"},
        method  = "permutation",
        reps    = 500,
        seed    = 42,
        n_jobs  = -1,          # <- parallel: use all cores
    )
    print(result)
"""

from __future__ import annotations

import re
import time
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import pyfixest as pf
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Formula column extractor
# -----------------------------------------------------------------------------

def _formula_cols(formula: str) -> set[str]:
    """
    Extract all bare variable names from a pyfixest formula string such as:
        "y ~ x1 + x2 + i(rel, treated, ref=-1.0) | firm + year"

    Strategy: strip known syntax scaffolding, then collect word-tokens that
    look like column names (start with a letter/underscore, not a pure number).
    This is intentionally broad — false positives (extra cols) are harmless;
    false negatives (missing cols) would break feols.
    """
    # Remove pyfixest-specific wrappers: i(...), C(...), sw(...), csw(...), etc.
    # Then tokenise everything that looks like an identifier.
    stripped = re.sub(r'\b(?:i|C|sw|csw|sw0|csw0)\s*\(', ' ', formula)
    stripped = re.sub(r'[^\w]', ' ', stripped)
    tokens   = stripped.split()
    cols: set[str] = set()
    for tok in tokens:
        # Skip pure numeric tokens (ref values, lag numbers, etc.)
        try:
            float(tok)
            continue
        except ValueError:
            pass
        if re.match(r'^[A-Za-z_]\w*$', tok):
            cols.add(tok)
    return cols


# -----------------------------------------------------------------------------
# Top-level worker  (must be at module level so joblib/loky can pickle it)
# -----------------------------------------------------------------------------

def _one_rep(
    j: int,
    df_values: np.ndarray,
    df_columns: list[str],
    group: str,
    formula: str,
    vcov,
    method: str,
    n1: int,
    common_vars: list[str],
    child_seed: int,
    feols_kwargs: dict,
) -> np.ndarray | None:
    """
    Single resampling iteration.

    Returns a 1-D float array of length k (one coefficient diff per variable),
    or None if the iteration failed (e.g. perfect collinearity in a bootstrap
    draw or a missing level after shuffling).

    Defined at module level so the loky process pool can pickle it.
    """
    rng = np.random.default_rng(child_seed)
    df  = pd.DataFrame(df_values, columns=df_columns)
    k   = len(common_vars)

    try:
        if method == "permutation":
            # Randomly reorder all rows, then split at the original group-0 size
            perm_idx = rng.permutation(len(df))
            df_sh    = df.iloc[perm_idx].reset_index(drop=True)
            df_g0    = df_sh.iloc[:n1]
            df_g1    = df_sh.iloc[n1:]

        else:  # bootstrap
            # Resample each group independently with replacement
            mask0 = (df[group] == 0).values
            mask1 = ~mask0
            idx0  = rng.choice(np.where(mask0)[0], size=mask0.sum(), replace=True)
            idx1  = rng.choice(np.where(mask1)[0], size=mask1.sum(), replace=True)
            df_g0 = df.iloc[idx0].reset_index(drop=True)
            df_g1 = df.iloc[idx1].reset_index(drop=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0 = pf.feols(formula, data=df_g0, vcov=vcov, **feols_kwargs)
            f1 = pf.feols(formula, data=df_g1, vcov=vcov, **feols_kwargs)

        b0_j   = f0.coef()
        b1_j   = f1.coef()
        shared = b0_j.index.intersection(b1_j.index).intersection(common_vars)
        if len(shared) < k:
            return None   # some variable dropped — skip this rep
        return (b0_j.loc[common_vars] - b1_j.loc[common_vars]).values

    except Exception:
        return None


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

def bdiff(
    df: pd.DataFrame,
    group: str,
    formula: str,
    vcov: str | dict = "iid",
    method: Literal["permutation", "bootstrap", "wald"] = "permutation",
    reps: int = 500,
    seed: int | None = 42,
    n_jobs: int = 1,
    verbose: bool = True,
    **feols_kwargs,
) -> pd.DataFrame:
    """
    Test whether regression coefficients differ between group==0 and group==1.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset. Must contain `group` (values 0 and 1 only).
    group : str
        Name of the 0/1 grouping variable.
    formula : str
        PyFixest formula, e.g. "y ~ x1 + x2 | firm + year".
        Do NOT include if/in conditions — filter `df` beforehand.
    vcov : str or dict
        Variance-covariance type, e.g. "iid", "HC1", {"CRV1": "firm"}.
    method : {"permutation", "bootstrap", "wald"}
        "permutation" -- randomly shuffle the group label (default, non-parametric).
        "bootstrap"   -- resample each group with replacement (non-parametric).
        "wald"        -- chi-squared test using block-diagonal VCV (parametric).
    reps : int
        Number of replications for permutation/bootstrap. Ignored for "wald".
    seed : int or None
        Master random seed. Child seeds are derived deterministically via
        numpy.random.SeedSequence, so output is reproducible for any n_jobs.
    n_jobs : int
        Number of parallel worker processes.
          1  -> serial (default, always safe)
         -1  -> use all CPU cores
          k  -> use exactly k cores
        Ignored when method="wald".
    verbose : bool
        Show tqdm progress bar and print the results table.
    **feols_kwargs
        Extra arguments forwarded to pf.feols()
        (e.g. ssc=pf.ssc(G_df="conventional")).

    Returns
    -------
    pd.DataFrame
        Index  = variable names.
        Columns: b_group0, b_group1, diff, stat, p_value
                 + se_diff (Wald only), valid_reps (resampling only)
    """
    t0 = time.time()

    # -- 0. Validate and clean ------------------------------------------------
    df = df.copy()
    df = df[df[group].isin([0, 1])].dropna(subset=[group]).reset_index(drop=True)

    # Factorize object columns (except the group flag) so that df.values produces
    # a float64 array instead of an object array.  An object array makes loky
    # pickle every element as a Python object, which inflates memory by ~6x and
    # causes OOM / SIGTERM(-15) when the dataset is large.
    for _col in df.columns:
        if _col != group and df[_col].dtype == object:
            df[_col] = pd.factorize(df[_col])[0]

    unique_vals = sorted(df[group].unique().tolist())
    if unique_vals != [0, 1]:
        raise ValueError(
            f"`group` must contain exactly 0 and 1.  Found: {unique_vals}"
        )
    if method not in ("permutation", "bootstrap", "wald"):
        raise ValueError(
            f"method must be 'permutation', 'bootstrap', or 'wald'. Got: {method!r}"
        )

    # -- 1. Observed (true) coefficients --------------------------------------
    fit0 = pf.feols(formula, data=df[df[group] == 0], vcov=vcov, **feols_kwargs)
    fit1 = pf.feols(formula, data=df[df[group] == 1], vcov=vcov, **feols_kwargs)

    b0 = fit0.coef()
    b1 = fit1.coef()

    common_vars = list(b0.index.intersection(b1.index))
    if len(common_vars) == 0:
        raise ValueError("No common coefficients found between the two groups.")
    if len(common_vars) < max(len(b0), len(b1)):
        warnings.warn(
            "Some coefficients differ across groups — only common ones are tested."
        )

    b0  = b0.loc[common_vars]
    b1  = b1.loc[common_vars]
    D0  = (b0 - b1).values          # shape (k,)
    n1  = int(fit0._N)               # group-0 sample size

    # -- 2. Branch by method --------------------------------------------------
    if method == "wald":
        results = _wald_test(fit0, fit1, D0, common_vars)
    else:
        results = _resampling_parallel(
            df=df, group=group, formula=formula, vcov=vcov,
            method=method, reps=reps, seed=seed, n_jobs=n_jobs,
            D0=D0, common_vars=common_vars, n1=n1,
            verbose=verbose, feols_kwargs=feols_kwargs,
        )

    # -- 3. Assemble output ---------------------------------------------------
    out = (
        pd.DataFrame({
            "variable": common_vars,
            "b_group0": b0.values,
            "b_group1": b1.values,
            "diff":     D0,
        })
        .merge(results, on="variable", how="left")
    )

    if verbose:
        _print_summary(out, method, reps, n_jobs, group, time.time() - t0)

    return out.set_index("variable")


# -----------------------------------------------------------------------------
# Mode 3: Wald (parametric)
# -----------------------------------------------------------------------------

def _wald_test(fit0, fit1, D0: np.ndarray, common_vars: list[str]) -> pd.DataFrame:
    """
    Since the two subsamples are non-overlapping, Cov(b0, b1) = 0, so:

        Var(b0 - b1) = Var(b0) + Var(b1)

    Per-coefficient two-sided z-test:
        z_k = (b0_k - b1_k) / sqrt(se0_k^2 + se1_k^2)
        p_k = 2 * Phi(-|z_k|)
    """
    from scipy import stats as sc

    se0     = fit0.se().loc[common_vars].values
    se1     = fit1.se().loc[common_vars].values
    se_diff = np.sqrt(se0 ** 2 + se1 ** 2)
    z       = D0 / se_diff

    return pd.DataFrame({
        "variable":   common_vars,
        "se_diff":    se_diff,
        "stat":       z ** 2,
        "stat_label": "Chi2",
        "p_value":    2 * sc.norm.sf(np.abs(z)),
    })


# -----------------------------------------------------------------------------
# Mode 1 & 2: Resampling with joblib parallelism
# -----------------------------------------------------------------------------

def _resampling_parallel(
    df, group, formula, vcov, method, reps, seed, n_jobs,
    D0, common_vars, n1, verbose, feols_kwargs,
) -> pd.DataFrame:
    """
    Seed splitting
    --------------
    np.random.SeedSequence spawns `reps` statistically independent child
    sequences.  Each child drives exactly one call to _one_rep(), so the
    full result matrix is reproducible regardless of n_jobs or scheduling.

    Transfer overhead
    -----------------
    df is converted to a plain numpy array before dispatch to minimise
    pickle size (DataFrame metadata roughly doubles the payload).
    """
    k = len(D0)

    # Deterministic, independent child seeds
    ss          = np.random.SeedSequence(seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(reps)]

    # Slim the DataFrame to only the columns feols needs (lhs, rhs, FE, group).
    # Sending a 40-column pooled panel through loky pickles 40 cols × 750k rows
    # per task, which causes SIGTERM(-15) even after factorizing object columns.
    _needed = _formula_cols(formula) | {group}
    _keep   = [c for c in df.columns if c in _needed]
    df_slim = df[_keep]

    # Numpy array is smaller to pickle than a DataFrame
    df_values  = df_slim.values
    df_columns = list(df_slim.columns)

    tasks = (
        delayed(_one_rep)(
            j, df_values, df_columns, group, formula, vcov,
            method, n1, common_vars, child_seeds[j], feols_kwargs,
        )
        for j in range(reps)
    )

    raw = Parallel(
        n_jobs  = n_jobs,
        backend = "loky",      # process-based; safe with pyfixest
    )(
        tqdm(tasks, total=reps, desc=f"bdiff [{method}]", disable=not verbose)
    )

    # Collect into matrix; failed reps stay as NaN
    D_mat = np.full((reps, k), np.nan)
    for j, res in enumerate(raw):
        if res is not None:
            D_mat[j] = res

    valid = int(np.sum(~np.isnan(D_mat[:, 0])))
    if valid == 0:
        raise RuntimeError(
            "All resampling iterations failed. Check formula and data."
        )

    freq_list, p_vals = [], []
    for j in range(k):
        col  = D_mat[:, j][~np.isnan(D_mat[:, j])]
        freq = int(np.sum(col >= D0[j]))
        p    = freq / len(col)
        p    = min(p, 1.0 - p)      # two-sided
        freq_list.append(freq)
        p_vals.append(p)

    return pd.DataFrame({
        "variable":   common_vars,
        "stat":       freq_list,
        "stat_label": "Freq",
        "p_value":    p_vals,
        "valid_reps": valid,
    })


# -----------------------------------------------------------------------------
# Pretty-print
# -----------------------------------------------------------------------------

def _print_summary(out, method, reps, n_jobs, group, elapsed):
    cores = f", {n_jobs} cores" if n_jobs != 1 and method != "wald" else ""
    label = {
        "permutation": f"Permutation Test ({reps} reps{cores})",
        "bootstrap":   f"Bootstrap Test ({reps} reps{cores})",
        "wald":        "Wald Test (parametric)",
    }[method]
    stat_col = out["stat_label"].iloc[0] if "stat_label" in out.columns else "Stat"
    W = 66
    print(f"\n{'─'*W}")
    print(f"  bdiff — {label}")
    print(f"  Group variable: {group}  (0 vs 1)")
    print(f"{'─'*W}")
    print(f"  {'Variable':<20} {'b0':>9} {'b1':>9} {'b0-b1':>9} {stat_col:>8} {'p-value':>9}")
    print(f"{'─'*W}")
    for _, row in out.iterrows():
        p = row["p_value"]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(
            f"  {str(row['variable']):<20}"
            f" {row['b_group0']:>9.4f}"
            f" {row['b_group1']:>9.4f}"
            f" {row['diff']:>9.4f}"
            f" {row['stat']:>8.1f}"
            f" {p:>9.4f}  {stars}"
        )
    print(f"{'─'*W}")
    print("  Signif. codes:  *** p<0.01   ** p<0.05   * p<0.1")
    if "valid_reps" in out.columns:
        print(f"  Valid reps: {int(out['valid_reps'].iloc[0])}/{reps}")
    print(f"  Time elapsed: {elapsed:.1f}s\n")


# -----------------------------------------------------------------------------
# Demo  —  MUST be inside __name__ guard for multiprocessing on Windows/macOS
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import os, time as _time

    np.random.seed(0)
    n = 600
    demo_df = pd.DataFrame({
        "y":     np.random.randn(n),
        "x1":    np.random.randn(n),
        "x2":    np.random.randn(n),
        "group": np.repeat([0, 1], n // 2),
        "firm":  np.tile(np.arange(30), n // 30),
    })
    # Give group=1 a stronger x1 slope so coefficients genuinely differ
    mask1 = demo_df["group"] == 1
    demo_df.loc[mask1, "y"] += 0.9 * demo_df.loc[mask1, "x1"]

    FORMULA  = "y ~ x1 + x2 | firm"
    VCOV     = {"CRV1": "firm"}
    N_CORES  = os.cpu_count() or 1
    REPS     = 300

    print(f"System: {N_CORES} logical cores\n")

    # Serial
    t0 = _time.time()
    res_serial = bdiff(
        df=demo_df, group="group", formula=FORMULA, vcov=VCOV,
        method="permutation", reps=REPS, seed=42, n_jobs=1,
    )
    t_serial = _time.time() - t0

    # Parallel
    t0 = _time.time()
    res_par = bdiff(
        df=demo_df, group="group", formula=FORMULA, vcov=VCOV,
        method="permutation", reps=REPS, seed=42, n_jobs=-1,
    )
    t_par = _time.time() - t0

    print(f"Serial  : {t_serial:.1f}s")
    print(f"Parallel: {t_par:.1f}s  ({N_CORES} cores)")
    print(f"Speedup : {t_serial/t_par:.2f}x\n")

    # Wald for quick comparison
    res_wald = bdiff(
        df=demo_df, group="group", formula=FORMULA, vcov=VCOV,
        method="wald",
    )
    print("Wald results:")
    print(res_wald[["b_group0", "b_group1", "diff", "stat", "p_value"]])