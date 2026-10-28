"""Estimator adapters; keep estimation samples and score contributions intact."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyfixest as pf

from .probit import observed_influence, refine_no_fe
from .separation import separated_rows


@dataclass
class Estimate:
    """Result contract for built-in and user-supplied estimators.

    ``sample_index`` contains the input frame's index labels actually used.
    ``influence`` is an uncorrected, per-observation coefficient influence
    matrix; its rows follow ``sample_index``. It enables joint sandwich tests.
    """

    coefficients: pd.Series
    covariance: pd.DataFrame
    sample_index: np.ndarray
    influence: pd.DataFrame | None = None
    model: Any = field(default=None, repr=False)
    diagnostics: dict = field(default_factory=dict)

    @property
    def nobs(self) -> int:
        return len(self.sample_index)


Estimator = str | Callable[..., Estimate]
ALIASES = {"feols": "ols", "regress": "ols", "reghdfe": "ols",
           "fepois": "poisson", "ppml": "poisson", "ppmlhdfe": "poisson"}


def normalize_estimator(estimator: Estimator) -> Estimator:
    if callable(estimator):
        return estimator
    name = ALIASES.get(str(estimator).lower(), str(estimator).lower())
    if name not in {"ols", "poisson", "logit", "probit"}:
        raise ValueError("estimator must be ols, ppml/poisson, logit, probit, or a callable")
    return name


def fit_estimator(data, formula, estimator, vcov, options) -> Estimate:
    """Fit one model without changing string categories or dropping columns."""
    if callable(estimator):
        result = estimator(data=data, formula=formula, vcov=vcov, **options)
        if not isinstance(result, Estimate):
            raise TypeError("A custom estimator must return pybdiff.Estimate")
        return validate_estimate(result, data)
    kwargs = dict(options)
    if kwargs.get("lean") or kwargs.get("store_data") is False:
        raise ValueError("lean=True and store_data=False hide the estimation sample")
    if any(kwargs.get(key) is not None for key in ("split", "fsplit")):
        raise ValueError("Supply a single model; split/fsplit are not supported")
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("A nonempty formula is required for built-in estimators")
    model_data = data.copy()
    row_marker = "__bdiff_input_row__"
    while row_marker in model_data:
        row_marker += "_"
    model_data[row_marker] = data.index.to_numpy()
    removed = []
    exposure = kwargs.pop("exposure", None)
    if exposure is not None:
        if estimator != "poisson" or "offset" in kwargs:
            raise ValueError("exposure requires Poisson and cannot be combined with offset")
        if exposure not in data or not (data[exposure] > 0).all():
            raise ValueError("exposure must name a strictly positive column")
        exposure_marker = row_marker + "exposure"
        model_data[exposure_marker] = np.log(data[exposure].to_numpy())
        kwargs["offset"] = exposure_marker
    if estimator == "ols":
        model = pf.feols(formula, data=model_data, vcov=vcov, **kwargs)
    elif estimator == "poisson":
        kwargs.setdefault("iwls_maxiter", 200)
        kwargs.setdefault("iwls_tol", 1e-10)
        kwargs.setdefault("separation_check", ["fe"])
        model = pf.fepois(formula, data=model_data, vcov=vcov, **kwargs)
        removed = separated_rows(model)
        if removed:
            retained = model._data.drop(index=removed)
            model = pf.fepois(formula, data=retained, vcov=vcov, **kwargs)
    else:
        kwargs.setdefault("iwls_maxiter", 200)
        kwargs.setdefault("iwls_tol", 1e-10)
        model = pf.feglm(formula, data=model_data, family=estimator, vcov=vcov, **kwargs)
    if not hasattr(model, "coef"):
        raise ValueError("The formula produced multiple models; supply one specification")
    if getattr(model, "convergence", True) is False:
        raise RuntimeError("The nonlinear estimator did not converge")
    if estimator == "probit":
        refine_no_fe(model)
    coefficients = model.coef()
    covariance = pd.DataFrame(model._vcov, index=coefficients.index,
                              columns=coefficients.index)
    influence = (observed_influence(model) if estimator == "probit"
                 else _get_influence(model, coefficients.index))
    sample_index = model._data[row_marker].to_numpy()
    if influence is not None:
        influence.index = sample_index
    result = Estimate(coefficients, covariance, sample_index,
                      influence, model, {"estimator": estimator,
                      "dropped": len(data) - len(model._data),
                      "separated": int(getattr(model, "n_separation_na", 0)) + len(removed),
                      "weights": kwargs.get("weights"),
                      "weights_type": kwargs.get("weights_type", "aweights"),
                      "separation_checks": ([*kwargs.get("separation_check", []), "lp"]
                                            if estimator == "poisson" else None),
                      "omitted": list(getattr(model, "_collin_vars", []) or [])})
    return validate_estimate(result, data)


def _get_influence(model, names):
    if getattr(model, "_is_iv", False):
        return None  # IV scores need their own moment-to-parameter projection.
    scores, bread = np.asarray(model._scores), np.asarray(model._bread)
    if scores.ndim != 2 or bread.shape != (len(names), len(names)):
        return None
    values = scores @ bread.T
    return pd.DataFrame(values, index=model._data.index, columns=names)


def validate_estimate(result: Estimate, data: pd.DataFrame) -> Estimate:
    names = result.coefficients.index
    if names.has_duplicates or len(names) == 0:
        raise ValueError("An estimator must return unique, nonempty coefficient names")
    if not np.isfinite(result.coefficients.to_numpy(dtype=float)).all():
        raise ValueError("The estimator returned nonfinite coefficients")
    if not pd.Index(result.sample_index).isin(data.index).all():
        raise ValueError("Estimator sample_index must refer to its input frame")
    if len(np.unique(result.sample_index)) != len(result.sample_index):
        raise ValueError("Estimator sample_index contains duplicates")
    if result.nobs == 0:
        raise ValueError("The estimator used no observations")
    result.covariance = result.covariance.loc[names, names]
    if result.influence is not None:
        result.influence = result.influence.loc[result.sample_index, names]
        if not np.isfinite(result.influence.to_numpy()).all():
            raise ValueError("The estimator returned nonfinite influence contributions")
    return result
