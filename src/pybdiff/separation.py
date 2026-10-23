"""Certify Poisson separation using sparse recession-direction linear programs."""

import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog

from .design import main_matrix


def separated_rows(model):
    """Find zero-outcome rows admitting a strictly negative recession direction.

    For Poisson, a direction d separates if X_positive d = 0 and X_zero d <= 0,
    with a strict inequality for at least one zero. Bounded LPs find the union of
    such rows. Sparse indicator columns include absorbed fixed effects without
    constructing a dense dummy-variable matrix.
    """
    main = main_matrix(model)
    X = np.asarray(main["rhs"], dtype=float)
    y = np.asarray(main["lhs"], dtype=float).reshape(-1)
    zeros = np.flatnonzero(y == 0)
    if len(zeros) == 0:
        return []
    scale = np.maximum(np.max(np.abs(X), axis=0), 1.0)
    X = sparse.csr_matrix(X / scale)
    if getattr(model, "_has_fixef", False):
        effects = np.asarray(model._fe)
        parts = [X]
        for j in range(effects.shape[1]):
            _, codes = np.unique(effects[:, j], return_inverse=True)
            parts.append(sparse.csr_matrix((np.ones(len(y)), (np.arange(len(y)), codes)),
                                            shape=(len(y), int(codes.max()) + 1)))
        X = sparse.hstack(parts, format="csr")
    zero_design = X[zeros]
    positive_design = X[y > 0]
    found = np.zeros(len(zeros), dtype=bool)
    for _ in range(len(zeros) + 1):
        objective = np.asarray(zero_design[~found].sum(axis=0)).reshape(-1)
        # SciPy 将 threads 原样传给 HiGHS；限制每个并行进程的线程数。
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*",
                                    category=OptimizeWarning)
            solution = linprog(objective, A_ub=zero_design, b_ub=np.zeros(len(zeros)),
                               A_eq=positive_design, b_eq=np.zeros(positive_design.shape[0]),
                               bounds=[(-1, 1)] * X.shape[1], method="highs",
                               options={"threads": 1})
        if not solution.success:
            raise RuntimeError(f"Poisson separation certification failed: {solution.message}")
        discovered = np.asarray(zero_design @ solution.x).reshape(-1) < -1e-7
        if not np.any(discovered & ~found):
            break
        found |= discovered
        if found.all():
            break
    return model._data.index.to_numpy()[zeros[found]].tolist()
