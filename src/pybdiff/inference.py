"""Tail definitions, centered bootstrap tests, and joint score-sandwich Wald tests."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def cluster_columns(vcov) -> list[str]:
    if not isinstance(vcov, dict):
        return []
    if len(vcov) != 1 or next(iter(vcov)) not in {"CRV1", "CRV3"}:
        raise ValueError("Cluster covariance must use a single CRV1/CRV3 specification")
    value = next(iter(vcov.values()))
    if not isinstance(value, str):
        raise ValueError("Cluster variables must be specified by column names")
    columns = [name.strip() for name in value.split("+")]
    if not all(columns) or len(set(columns)) != len(columns):
        raise ValueError("Cluster variable names must be nonempty and unique")
    return columns


def resampling_table(draws, observed, names, *, method, compatibility, alternative,
                     null, level):
    centered = method == "bootstrap" and compatibility != "stata"
    reference = draws - observed if centered else draws
    target = observed - null
    rows = []
    for j, name in enumerate(names):
        values = draws[:, j]
        ref = reference[:, j]
        upper = int(np.sum(values >= observed[j]))
        if alternative == "legacy":
            q = upper / len(values)
            p, count = min(q, 1 - q), upper
        else:
            count = _tail_count(ref, target[j], alternative)
            p = min(1.0, (count + 1) / (len(ref) + 1))
        row = {"variable": name, "stat": count, "stat_label": "Freq",
               "p_value": p, "valid_reps": len(values),
               "mc_se": np.sqrt(p * (1 - p) / len(values))}
        if centered:
            alpha = (1 - level) / 2
            lo, hi = np.quantile(values - observed[j], [alpha, 1 - alpha])
            row.update(se_diff=np.std(values, ddof=1),
                       ci_low=observed[j] - hi, ci_high=observed[j] - lo)
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable")


def _tail_count(values, target, alternative):
    tolerance = 1e-12 * max(1.0, abs(target))
    if alternative == "two-sided":
        return int(np.sum(np.abs(values) >= abs(target) - tolerance))
    if alternative == "greater":
        return int(np.sum(values >= target - tolerance))
    if alternative == "less":
        return int(np.sum(values <= target + tolerance))
    raise ValueError("alternative must be two-sided, greater, less, or legacy")


def joint_covariance(fit0, fit1, data, names, vcov):
    if fit0.influence is None or fit1.influence is None:
        raise ValueError("Joint Wald inference requires observation-level influence contributions; "
                         "provide them in the adapter or choose wald_vcov='independent'")
    if isinstance(vcov, dict) and next(iter(vcov)) != "CRV1":
        raise ValueError("Joint Wald inference supports CRV1, not cluster jackknife CRV3")
    if isinstance(vcov, str) and vcov not in {"iid", "HC0", "HC1", "hetero"}:
        raise ValueError("Joint Wald inference supports robust or CRV1 covariance")
    psi = pd.DataFrame(0.0, index=data.index, columns=names)
    psi.loc[fit0.sample_index] += fit0.influence.loc[fit0.sample_index, names]
    psi.loc[fit1.sample_index] -= fit1.influence.loc[fit1.sample_index, names]
    used = np.union1d(fit0.sample_index, fit1.sample_index)
    psi = psi.loc[used]
    clusters = cluster_columns(vcov)
    if not clusters:
        n = len(psi)
        if fit0.diagnostics.get("weights_type") == "fweights":
            column = fit0.diagnostics.get("weights")
            if column != fit1.diagnostics.get("weights"):
                raise ValueError("Frequency-weight specifications must match across groups")
            weights = data.loc[used, column].to_numpy(dtype=float)
            psi = psi.div(np.sqrt(weights), axis=0)
            n = float(weights.sum())
        if n < 2:
            raise ValueError("Joint covariance needs at least two observations")
        return psi.to_numpy().T @ psi.to_numpy() * n / (n - 1)
    covariance = np.zeros((len(names), len(names)))
    for order in range(1, len(clusters) + 1):
        for subset in combinations(clusters, order):
            keys = pd.MultiIndex.from_frame(data.loc[used, list(subset)])
            codes, levels = pd.factorize(keys, sort=False)
            g = len(levels)
            if g < 2:
                raise ValueError("Each cluster dimension needs at least two clusters")
            sums = np.zeros((g, len(names)))
            np.add.at(sums, codes, psi.to_numpy())
            covariance += (-1) ** (order + 1) * g / (g - 1) * (sums.T @ sums)
    return covariance


def wald_table(fit0, fit1, data, group, names, vcov, *, kind, null, alternative, level):
    if alternative == "legacy":
        alternative = "two-sided"  # Stata surtest is an ordinary chi-square Wald test.
    if kind == "suest":
        covariance = joint_covariance(fit0, fit1, data, names, vcov)
    elif kind == "independent":
        for column in cluster_columns(vcov):
            if data.groupby(column, observed=True)[group].nunique().gt(1).any():
                raise ValueError("Independent Wald is invalid when clusters span both groups")
        covariance = (fit0.covariance.loc[names, names].to_numpy()
                      + fit1.covariance.loc[names, names].to_numpy())
    else:
        raise ValueError("wald_vcov must be 'suest' or 'independent'")
    diagonal = np.diag(covariance)
    if not np.isfinite(covariance).all() or np.any(diagonal <= 0):
        raise ValueError("The coefficient-difference covariance is not positive on its diagonal")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() < -1e-10 * max(1.0, float(abs(eigenvalues).max())):
        raise ValueError("The joint covariance is not positive semidefinite")
    difference = fit0.coefficients.loc[names].to_numpy() - fit1.coefficients.loc[names].to_numpy()
    se = np.sqrt(diagonal)
    z = (difference - null) / se
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p = stats.norm.sf(z)
    else:
        p = stats.norm.cdf(z)
    critical = stats.norm.ppf((1 + level) / 2)
    table = pd.DataFrame({"variable": names, "se_diff": se, "stat": z ** 2,
                          "stat_label": "Chi2", "p_value": p,
                          "ci_low": difference - critical * se,
                          "ci_high": difference + critical * se}).set_index("variable")
    rank = int(np.linalg.matrix_rank(covariance))
    contrast = difference - null
    if not np.allclose(covariance @ np.linalg.pinv(covariance) @ contrast, contrast,
                       rtol=1e-7, atol=1e-9):
        raise ValueError("The joint restriction is not estimable from the covariance rank")
    joint_stat = float(contrast @ np.linalg.pinv(covariance) @ contrast)
    joint = {"statistic": joint_stat, "df": rank,
             "p_value": float(stats.chi2.sf(joint_stat, rank)),
             "covariance": covariance.tolist(), "variables": names, "kind": kind}
    return table, joint
