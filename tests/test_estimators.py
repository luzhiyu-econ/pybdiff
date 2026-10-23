"""Model support and estimation-sample handling."""

import numpy as np
import pandas as pd
import pyfixest as pf
import pytest

from pybdiff import Estimate, ResamplingError, bdiff


@pytest.mark.parametrize("estimator", ["ppml", "poisson", "ppmlhdfe"])
def test_ppml_matches_direct_pyfixest(poisson_data, estimator):
    data = poisson_data
    result = bdiff(data, "group", "y ~ x + z | firm + year", estimator=estimator,
                   method="wald", vcov={"CRV1": "firm"}, verbose=False)
    fits = [pf.fepois("y ~ x + z | firm + year", data=data[data.group == g],
                      vcov={"CRV1": "firm"}, iwls_tol=1e-10, iwls_maxiter=200) for g in (0, 1)]
    np.testing.assert_allclose(result.b_group0, fits[0].coef(), rtol=1e-9)
    np.testing.assert_allclose(result.b_group1, fits[1].coef(), rtol=1e-9)


def test_ppml_handles_fractional_levels_and_zeros(poisson_data):
    data = poisson_data.assign(y=poisson_data.y / 2.0)
    result = bdiff(data, "group", "y ~ x", estimator="ppml", method="bootstrap",
                   reps=7, verbose=False)
    assert np.isfinite(result.p_value).all()


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_binary_models(linear_data, estimator):
    rng = np.random.default_rng(201)
    data = linear_data.assign(y=rng.binomial(1, 0.5, len(linear_data)))
    result = bdiff(data, "group", "y ~ x + z", estimator=estimator,
                   method="wald", verbose=False)
    assert result.p_value.between(0, 1).all()


def test_missing_values_freeze_actual_sample(linear_data):
    data = linear_data.copy()
    data.loc[[0, 90], "x"] = np.nan
    data.index = ["duplicate"] * len(data)
    result = bdiff(data, "group", "y ~ x + z", reps=5, verbose=False, keep_draws=True)
    metadata = result.attrs["metadata"]
    assert metadata["sampling_rows"] == 158
    assert metadata["n_group0"] == metadata["n_group1"] == 79
    assert all(info["n0"] == info["n1"] == 79 for info in metadata["draw_samples"])


def test_strings_preserve_category_and_missing_semantics(linear_data):
    data = linear_data.copy()
    data["category"] = np.where(data.year < 2, "north", "south")
    data.loc[0, "category"] = None
    before = data.copy(deep=True)
    result = bdiff(data, "group", "y ~ x + C(category)", method="wald", verbose=False)
    pd.testing.assert_frame_equal(before, data)
    assert any("south" in name for name in result.index)
    assert result.attrs["metadata"]["n_group0"] == 79


def failing_adapter(*, data, formula, vcov, **kwargs):
    if data["group"].nunique() > 1:
        raise ArithmeticError("intentional draw failure")
    return Estimate(pd.Series({"x": 1.0}), pd.DataFrame([[1.0]], index=["x"], columns=["x"]),
                    data.index.to_numpy())


def test_failures_are_not_silently_ignored(linear_data):
    with pytest.raises(ResamplingError, match="intentional draw failure"):
        bdiff(linear_data, "group", estimator=failing_adapter, reps=5, verbose=False)


def test_invalid_group_is_not_silently_filtered(linear_data):
    data = linear_data.copy()
    data.loc[0, "group"] = 2
    with pytest.raises(ValueError, match="exactly 0 and 1"):
        bdiff(data, "group", "y ~ x", verbose=False)
