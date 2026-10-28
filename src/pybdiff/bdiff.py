"""Coefficient-difference tests with explicit sampling and inference semantics."""

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from .estimators import fit_estimator, normalize_estimator
from .inference import cluster_columns, resampling_table, wald_table
from .sampling import SamplingPlan, make_plan, materialize


class ResamplingError(RuntimeError):
    """A draw failed; the exception records the replication and original cause."""


def _columns(value):
    return [value] if isinstance(value, str) else list(value or [])


def _sampling_unit(data, group, vcov, requested, compatibility):
    clusters = cluster_columns(vcov)
    if requested != "auto":
        return requested
    if compatibility == "stata" or not clusters:
        return None
    if len(clusters) != 1:
        raise ValueError("Specify resample_unit explicitly for multiway clustering")
    unit = clusters[0]
    if data.groupby(unit, observed=True)[group].nunique().gt(1).any():
        raise ValueError("Clusters span both groups. Specify a design-valid resample_unit "
                         "or explicitly use resample_unit=None with appropriate strata")
    return unit


def _prepare(df, group, vcov, unit, strata, options):
    if not isinstance(df, pd.DataFrame) or group not in df:
        raise ValueError("df must be a DataFrame containing the group column")
    if df.columns.has_duplicates:
        raise ValueError("Input column names must be unique")
    nonmissing = df[group].dropna()
    if not nonmissing.isin([0, 1]).all() or nonmissing.nunique() != 2:
        raise ValueError("group must contain exactly 0 and 1 (missing groups are excluded)")
    extra = cluster_columns(vcov) + strata + ([unit] if unit else [])
    extra += [options[k] for k in ("weights", "offset", "exposure")
              if isinstance(options.get(k), str)]
    missing = [column for column in extra if column not in df]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    complete = df[list(dict.fromkeys([group, *extra]))].notna().all(axis=1).to_numpy()
    clean = df.loc[complete].copy()
    clean = clean.reset_index(drop=True)
    clean.attrs["input_positions"] = np.flatnonzero(complete).tolist()
    if clean[group].nunique() != 2:
        raise ValueError("Missing-value exclusions removed an entire group")
    return clean


def _fit_pair(data, group, formula, estimator, vcov, options):
    return tuple(fit_estimator(data.loc[data[group] == g], formula, estimator, vcov, options)
                 for g in (0, 1))


def _coefficient_names(fit0, fit1, requested):
    common = fit0.coefficients.index.intersection(fit1.coefficients.index).tolist()
    names = _columns(requested) if requested is not None else common
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one uniquely named coefficient")
    missing = [name for name in names if name not in common]
    if missing:
        raise ValueError(f"Coefficients not identified in both groups: {missing}")
    omitted = sorted(set(fit0.coefficients.index).symmetric_difference(fit1.coefficients.index))
    if omitted and requested is None:
        warnings.warn(f"Only common coefficient names are tested; unmatched: {omitted}",
                      UserWarning, stacklevel=3)
    return names, omitted


def _one_rep(index, plan, data, group, formula, estimator, vcov, options, names,
             compatibility):
    try:
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
            draw = plan.draw_at(index, data, group)
            group0, group1 = materialize(data, draw, plan.resample_unit)
            fit0 = fit_estimator(group0, formula, estimator, vcov, options)
            if compatibility == "stata":
                unused = group0.loc[~group0.index.isin(fit0.sample_index)]
                group1 = pd.concat([unused, group1], ignore_index=True)
            fit1 = fit_estimator(group1, formula, estimator, vcov, options)
        missing = set(names) - set(fit0.coefficients.index.intersection(fit1.coefficients.index))
        if missing:
            raise ValueError(f"Coefficients omitted in this draw: {sorted(missing)}")
        difference = (fit0.coefficients.loc[names] - fit1.coefficients.loc[names]).to_numpy()
        return difference, {"rep": index, "n0": fit0.nobs, "n1": fit1.nobs,
                            "warnings": list(dict.fromkeys(str(w.message) for w in captured))}, None
    except Exception as error:
        return None, {"rep": index}, f"{type(error).__name__}: {error}"


def _resample(data, group, plan, formula, estimator, vcov, options, names, compatibility,
              n_jobs, on_error, min_valid_fraction):
    jobs = (delayed(_one_rep)(i, plan, data, group, formula, estimator,
            vcov, options, names, compatibility) for i in range(len(plan)))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(jobs)
    errors = [dict(info, error=error) for _, info, error in results if error is not None]
    if errors and on_error == "raise":
        first = errors[0]
        raise ResamplingError(f"Replication {first['rep']} failed: {first['error']}; "
                              f"{len(errors)}/{len(results)} failed. Use on_error='skip' "
                              "only after examining the failure mechanism.")
    valid = [(value, info) for value, info, error in results if error is None]
    if len(valid) < max(2, int(np.ceil(min_valid_fraction * len(results)))):
        raise ResamplingError(f"Only {len(valid)}/{len(results)} valid draws; first errors: "
                              f"{errors[:3]}")
    if errors:
        warnings.warn(f"Excluded {len(errors)} failed draws; details are in result.attrs", stacklevel=3)
    return np.stack([v for v, _ in valid]), [info for _, info in valid], errors


def bdiff(df, group, formula=None, vcov=None, method="permutation", reps=999, seed=42,
          n_jobs=1, verbose=True, *, estimator="ols", compatibility="modern",
          alternative=None, resample_unit="auto", strata=None, coefficients=None,
          null=0.0, level=0.95, wald_vcov="suest", plan=None, keep_draws=False,
          on_error="raise", min_valid_fraction=0.9, sample=None, **estimator_options):
    """Test coefficient differences (group 0 minus group 1).

    ``estimator`` accepts ols, ppml/poisson, logit, probit, or a callable returning
    :class:`Estimate`. Poisson estimates levels, including zeros; it never logs y.
    ``compatibility='stata'`` selects pooled bootstrap and the original smaller-
    tail probability. Modern mode uses absolute two-sided permutation tails or
    centered bootstrap tails, with a (count+1)/(B+1) Monte Carlo correction.
    ``vcov`` controls estimation and Wald inference. ``resample_unit`` separately
    controls sampling; auto selects a single cluster nested within group.
    ``plan`` replays explicit draws, independently of RNG implementation.
    ``sample`` is estimation (modern default) or original (Stata default).

    The returned DataFrame preserves the 0.1 result columns. Its metadata records
    samples, inference definitions, failures, and joint Wald inference. With
    keep_draws=True it also carries draws and their replayable sampling plan.
    """
    if method not in {"permutation", "bootstrap", "wald"}:
        raise ValueError("method must be permutation, bootstrap, or wald")
    if compatibility not in {"modern", "stata"}:
        raise ValueError("compatibility must be modern or stata")
    alternative = alternative or ("legacy" if compatibility == "stata" else "two-sided")
    if alternative not in {"two-sided", "greater", "less", "legacy"}:
        raise ValueError("alternative must be two-sided, greater, less, or legacy")
    if alternative == "legacy" and compatibility != "stata":
        raise ValueError("legacy tails require compatibility='stata'")
    if on_error not in {"raise", "skip"} or not 0 < min_valid_fraction <= 1:
        raise ValueError("Invalid failure policy or min_valid_fraction")
    if not 0 < level < 1 or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("level must be between 0 and 1 and n_jobs must be a nonzero integer")
    if method == "wald" and plan is not None:
        raise ValueError("Wald inference does not use a sampling plan")
    estimator = normalize_estimator(estimator)
    vcov = vcov if vcov is not None else ("HC1" if estimator == "poisson" else "iid")
    strata = _columns(strata)
    unit = None if resample_unit == "auto" else resample_unit
    data = _prepare(df, group, vcov, unit, strata, estimator_options)
    if method != "wald":
        unit = _sampling_unit(data, group, vcov, resample_unit, compatibility)
    fits = _fit_pair(data, group, formula, estimator, vcov, estimator_options)
    original_diagnostics = [fit.diagnostics for fit in fits]
    sample = sample or ("original" if compatibility == "stata" else "estimation")
    if sample not in {"estimation", "original"}:
        raise ValueError("sample must be estimation or original")
    prepared_rows = len(data)
    used = np.union1d(fits[0].sample_index, fits[1].sample_index)
    used_positions = [data.attrs["input_positions"][int(i)] for i in used]
    if sample == "estimation" and len(used) != len(data):
        data = data.loc[used].reset_index(drop=True)
        fits = _fit_pair(data, group, formula, estimator, vcov, estimator_options)
    fit0, fit1 = fits
    names, unmatched = _coefficient_names(fit0, fit1, coefficients)
    observed = (fit0.coefficients.loc[names] - fit1.coefficients.loc[names]).to_numpy()
    null_values = (np.full(len(names), float(null)) if np.isscalar(null)
                   else pd.Series(null).reindex(names).to_numpy(dtype=float))
    if not np.isfinite(null_values).all():
        raise ValueError("null must give a finite value for every selected coefficient")
    if (method == "permutation" or compatibility == "stata") and np.any(null_values != 0):
        raise ValueError("Permutation and Stata compatibility test a zero coefficient difference")
    metadata = {"method": method, "estimator": str(estimator), "formula": formula,
                "compatibility": compatibility, "alternative": alternative,
                "difference": "group0 - group1", "vcov": vcov, "null": null_values.tolist(),
                "input_rows": len(df), "prepared_rows": prepared_rows,
                "excluded_preparation_rows": len(df) - prepared_rows,
                "excluded_model_rows": prepared_rows - len(used),
                "sampling_rows": len(data), "n_group0": fit0.nobs, "n_group1": fit1.nobs,
                "sample": sample, "estimation_positions": used_positions,
                "resample_unit": unit, "strata": strata, "seed": seed,
                "unmatched_coefficients": unmatched,
                "group_diagnostics": original_diagnostics}
    if method == "wald":
        table, joint = wald_table(fit0, fit1, data, group, names, vcov, kind=wald_vcov,
                                 null=null_values, alternative=alternative, level=level)
        metadata["joint_wald"] = joint
    else:
        if isinstance(plan, (str, bytes)) or hasattr(plan, "__fspath__"):
            plan = SamplingPlan.load(plan)
        if plan is None:
            plan = make_plan(data, group, method=method, reps=reps, seed=seed,
                             resample_unit=unit, strata=strata, compatibility=compatibility,
                             n_group0=fit0.nobs if compatibility == "stata" and unit is None
                             and not strata else None)
        plan.validate(data, method, unit, strata)
        draws, info, errors = _resample(data, group, plan, formula, estimator, vcov,
                                       estimator_options, names, compatibility, n_jobs,
                                       on_error, min_valid_fraction)
        table = resampling_table(draws, observed, names, method=method,
                                 compatibility=compatibility, alternative=alternative,
                                 null=null_values, level=level)
        metadata.update(requested_reps=len(plan), valid_reps=len(draws),
                        failed_reps=len(errors), failures=errors,
                        p_definition="min(q,1-q)" if alternative == "legacy" else "(tail+1)/(B+1)",
                        centered_bootstrap=method == "bootstrap" and compatibility == "modern",
                        plan_source=plan.source)
        metadata["draw_warning_counts"] = dict(Counter(w for item in info for w in item["warnings"]))
        if keep_draws:
            metadata["draw_samples"] = info
    out = pd.DataFrame({"b_group0": fit0.coefficients.loc[names],
                        "b_group1": fit1.coefficients.loc[names], "diff": observed})
    out.index.name = "variable"
    out = out.join(table)
    out.attrs["metadata"] = metadata
    if method != "wald" and keep_draws:
        out.attrs["draws"] = pd.DataFrame(draws, columns=names)
        out.attrs["plan"] = plan
    if verbose:
        print(f"bdiff: {method}, {compatibility}, {alternative}; difference = group0 - group1")
        print(out.to_string(float_format=lambda v: f"{v:.6g}"))
        if method != "wald":
            print(f"Valid draws: {metadata['valid_reps']}/{metadata['requested_reps']}")
    return out
