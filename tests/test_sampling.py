"""Draw replay, clustering, strata, and worker-count invariance."""

import numpy as np
import pandas as pd
import pytest

from pybdiff import SamplingPlan, bdiff, make_plan
from pybdiff.sampling import materialize


@pytest.mark.parametrize("method", ["permutation", "bootstrap"])
def test_plan_roundtrip_and_lazy_eager_identity(linear_data, method, tmp_path):
    data = linear_data
    lazy = make_plan(data, "group", method=method, reps=7, seed=123)
    eager = make_plan(data, "group", method=method, reps=7, seed=123, store_indices=True)
    path = tmp_path / "plan.json"
    lazy.save(path)
    replay = SamplingPlan.load(path)
    replay.validate(data, method, None, [])
    for i in range(len(lazy)):
        np.testing.assert_array_equal(replay.draw_at(i, data, "group").index0,
                                      eager.draws[i].index0)
    with pytest.raises(ValueError, match="fingerprint"):
        replay.validate(data.iloc[::-1], method, None, [])


def test_cluster_permutation_keeps_whole_panels(linear_data):
    plan = make_plan(linear_data, "group", reps=9, resample_unit="firm")
    for i in range(len(plan)):
        draw = plan.draw_at(i, linear_data, "group")
        a = set(linear_data.iloc[draw.index0].firm)
        b = set(linear_data.iloc[draw.index1].firm)
        assert not a.intersection(b)
        assert len(a) == len(b) == 16


def test_cluster_bootstrap_relabels_copies(linear_data):
    plan = make_plan(linear_data, "group", method="bootstrap", reps=3, resample_unit="firm")
    draw = plan.draw_at(0, linear_data, "group")
    a, b = materialize(linear_data, draw, "firm")
    assert a.firm.nunique() == b.firm.nunique() == 16
    assert not set(a.firm).intersection(b.firm)
    assert a.groupby("firm").size().eq(5).all()


def test_stratified_permutation_preserves_assignment_counts(linear_data):
    plan = make_plan(linear_data, "group", reps=5, strata=["year"])
    expected = linear_data[linear_data.group == 0].groupby("year").size()
    for i in range(len(plan)):
        a, _ = materialize(linear_data, plan.draw_at(i, linear_data, "group"), None)
        pd.testing.assert_series_equal(a.groupby("year").size(), expected)


@pytest.mark.parametrize("method", ["permutation", "bootstrap"])
def test_parallel_matches_serial(linear_data, method):
    arguments = dict(df=linear_data, group="group", formula="y ~ x + z",
                     method=method, reps=9, seed=103, verbose=False, keep_draws=True)
    serial = bdiff(**arguments, n_jobs=1)
    parallel = bdiff(**arguments, n_jobs=2)
    np.testing.assert_array_equal(serial.to_numpy(), parallel.to_numpy())
    pd.testing.assert_frame_equal(serial.attrs["draws"], parallel.attrs["draws"])


def test_cluster_covariate_outside_formula_is_preserved(linear_data):
    result = bdiff(linear_data, "group", "y ~ x + z", vcov={"CRV1": "firm"},
                   reps=5, verbose=False)
    assert result.attrs["metadata"]["resample_unit"] == "firm"
    assert result.attrs["metadata"]["valid_reps"] == 5


def test_shared_cluster_requires_explicit_sampling_design(linear_data):
    with pytest.raises(ValueError, match="Clusters span both groups"):
        bdiff(linear_data, "group", "y ~ x", vcov={"CRV1": "shared"}, reps=3, verbose=False)


def test_stata_bootstrap_mixes_original_groups(linear_data):
    plan = make_plan(linear_data, "group", method="bootstrap", compatibility="stata", reps=3)
    draw = plan.draw_at(0, linear_data, "group")
    assert linear_data.iloc[draw.index0].group.nunique() == 2
    within = make_plan(linear_data, "group", method="bootstrap", reps=3)
    assert linear_data.iloc[within.draw_at(0, linear_data, "group").index0].group.eq(0).all()


def test_invalid_plan_is_rejected(linear_data):
    from pybdiff import Draw
    plan = SamplingPlan([Draw([0, 1], [1, 2])], len(linear_data), "permutation")
    with pytest.raises(ValueError, match="partition"):
        plan.validate(linear_data, "permutation", None, [])
