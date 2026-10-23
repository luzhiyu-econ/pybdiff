"""Test inferential definitions against independent calculations."""

import numpy as np
import pytest
from scipy import stats

from pybdiff import bdiff
from pybdiff.inference import resampling_table


def test_bootstrap_detects_large_difference(linear_data):
    result = bdiff(linear_data, "group", "y ~ x + z", method="bootstrap",
                   reps=99, seed=17, verbose=False)
    assert result.loc["x", "p_value"] <= 0.05
    assert result.loc["x", "ci_high"] < 0
    assert result.attrs["metadata"]["centered_bootstrap"]


def test_explicit_tail_definitions():
    values = np.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0]])
    arguments = dict(draws=values, observed=np.array([2.0]), names=["x"],
                     method="permutation", null=np.array([0.0]), level=0.95)
    modern = resampling_table(**arguments, compatibility="modern", alternative="two-sided")
    legacy = resampling_table(**arguments, compatibility="stata", alternative="legacy")
    assert modern.loc["x", "p_value"] == pytest.approx(3 / 6)
    assert legacy.loc["x", "p_value"] == 0.0


def test_wald_matches_direct_joint_sandwich(linear_data):
    data = linear_data
    result = bdiff(data, "group", "y ~ x + z", method="wald", verbose=False)
    influence = np.zeros((len(data), 3))
    for g in (0, 1):
        mask = data.group == g
        X = np.column_stack([np.ones(mask.sum()), data.loc[mask, ["x", "z"]]])
        beta = np.linalg.lstsq(X, data.loc[mask, "y"], rcond=None)[0]
        residual = data.loc[mask, "y"].to_numpy() - X @ beta
        influence[mask] = (1 - 2 * g) * (X * residual[:, None]) @ np.linalg.inv(X.T @ X)
    expected = influence.T @ influence * len(data) / (len(data) - 1)
    np.testing.assert_allclose(result.se_diff, np.sqrt(np.diag(expected)), rtol=1e-9)


def test_shared_clusters_use_cross_covariance(linear_data):
    data = linear_data
    result = bdiff(data, "group", "y ~ x + z", method="wald",
                   vcov={"CRV1": "shared"}, verbose=False)
    assert result.attrs["metadata"]["joint_wald"]["kind"] == "suest"
    with pytest.raises(ValueError, match="span both groups"):
        bdiff(data, "group", "y ~ x + z", method="wald", vcov={"CRV1": "shared"},
              wald_vcov="independent", verbose=False)


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_wald_direction_and_nonzero_null(linear_data, alternative):
    result = bdiff(linear_data, "group", "y ~ x", method="wald", null=-1.0,
                   alternative=alternative, coefficients=["x"], verbose=False)
    z = (result.loc["x", "diff"] + 1) / result.loc["x", "se_diff"]
    expected = {"greater": stats.norm.sf(z), "less": stats.norm.cdf(z),
                "two-sided": 2 * stats.norm.sf(abs(z))}[alternative]
    assert result.loc["x", "p_value"] == pytest.approx(expected)


def test_legacy_requires_compatibility(linear_data):
    with pytest.raises(ValueError, match="legacy tails"):
        bdiff(linear_data, "group", "y ~ x", alternative="legacy", verbose=False)


def test_rejects_nonzero_permutation_null(linear_data):
    with pytest.raises(ValueError, match="zero coefficient difference"):
        bdiff(linear_data, "group", "y ~ x", null=1, reps=3, verbose=False)
