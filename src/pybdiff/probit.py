"""Observed-Hessian influence for the noncanonical probit likelihood."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.stats import norm

from .design import main_matrix


def refine_no_fe(model):
    """Finish no-FE probit at the score root instead of a deviance-only stop."""
    if model._has_fixef:
        return
    matrices = main_matrix(model)
    X = np.asarray(matrices["rhs"][model.coef().index], dtype=float)
    y = np.asarray(matrices["lhs"], dtype=float).reshape(-1)
    sign = 2 * y - 1
    beta = model._beta_hat.copy()
    for _ in range(20):
        t = sign * (X @ beta)
        ratio = np.exp(norm.logpdf(t) - norm.logcdf(t))
        curvature = ratio * (ratio + t)
        score = X.T @ (sign * ratio)
        step = np.linalg.solve(X.T @ (curvature[:, None] * X), score)
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    model._beta_hat = beta
    model._Y_hat_link = X @ beta


def observed_influence(model):
    """Use observed curvature, as Stata suest does, rather than Fisher weights."""
    matrices = main_matrix(model)
    names = model.coef().index
    X = np.asarray(matrices["rhs"][names], dtype=float)
    y = np.asarray(matrices["lhs"], dtype=float).reshape(-1)
    eta = np.asarray(model._Y_hat_link).reshape(-1)
    sign = 2 * y - 1
    t = sign * eta
    ratio = np.exp(norm.logpdf(t) - norm.logcdf(t))
    curvature = ratio * (ratio + t)
    if np.any(curvature <= 0) or not np.isfinite(curvature).all():
        raise ValueError("Probit observed curvature is singular; check separation")
    if model._has_fixef:
        effects = np.asarray(model._fe)
        parts = []
        for column in range(effects.shape[1]):
            codes, levels = pd.factorize(effects[:, column], sort=False)
            parts.append(sparse.csr_matrix((np.ones(len(y)), (np.arange(len(y)), codes)),
                                            shape=(len(y), len(levels))))
        D = sparse.hstack(parts, format="csr")
        sqrt_w = np.sqrt(curvature)
        WD = D.multiply(sqrt_w[:, None]).tocsr()
        for j in range(X.shape[1]):
            projection = lsqr(WD, X[:, j] * sqrt_w, atol=1e-12, btol=1e-12)[0]
            X[:, j] -= np.asarray(D @ projection).reshape(-1)
    bread = np.linalg.inv(X.T @ (curvature[:, None] * X))
    influence = (X * (sign * ratio)[:, None]) @ bread
    return pd.DataFrame(influence, index=model._data.index, columns=names)
