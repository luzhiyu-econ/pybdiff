"""Independent comparisons with licensed Stata, SSC bdiff, and ppmlhdfe."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pybdiff import StataEstimator, bdiff, make_stata_plan
from pybdiff.stata import run_stata, stata_executable

pytestmark = pytest.mark.stata
SCRIPTS = Path(__file__).parent / "stata"


def reference_script(command, joint="yes", cluster=""):
    return (f'local command `"{command}"\'\nlocal joint "{joint}"\n'
            f'local cluster "{cluster}"\ninclude "{SCRIPTS / "reference.do"}"\n')


@pytest.fixture(autouse=True)
def require_stata():
    try:
        stata_executable()
    except FileNotFoundError:
        pytest.skip("A licensed Stata executable is required")


@pytest.mark.parametrize("cluster", ["", "firm", "shared"])
@pytest.mark.parametrize("estimator,command", [("ols", "regress y x z"),
                                               ("poisson", "poisson y x z"),
                                               ("logit", "logit y x z"),
                                               ("probit", "probit y x z, tolerance(1e-12) nrtolerance(1e-12)")])
def test_wald_equals_suest(linear_data, poisson_data, estimator, command, cluster, tmp_path, audit):
    data = linear_data if estimator == "ols" else poisson_data
    if estimator in {"logit", "probit"}:
        data = poisson_data.assign(y=(poisson_data.y > 1).astype(int))
    data.to_stata(tmp_path / "input.dta", write_index=False, version=118)
    script = reference_script(command, cluster=cluster)
    run_stata(tmp_path, script)
    reference = pd.read_stata(tmp_path / "reference.dta").set_index("variable")
    vcov = {"CRV1": cluster} if cluster else "HC1"
    actual = bdiff(data, "group", "y ~ x + z", method="wald", estimator=estimator,
                   vcov=vcov, coefficients=["x", "z"], verbose=False)
    np.testing.assert_allclose(actual["diff"], reference["diff"], atol=1e-7)
    np.testing.assert_allclose(actual["se_diff"], reference["se"], rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(actual["p_value"], reference["p"], atol=1e-6)
    for actual_name, reference_name in (("diff", "diff"), ("se_diff", "se"), ("p_value", "p")):
        audit(f"{estimator}_suest_{cluster or 'robust'}", actual_name,
              actual[actual_name], reference[reference_name])


@pytest.mark.parametrize("method", ["permutation", "bootstrap"])
@pytest.mark.parametrize("estimator,command", [("ols", "regress y x z"),
                                               ("ppml", "poisson y x z")])
def test_original_bdiff_seed_exact(linear_data, poisson_data, estimator, command, method, tmp_path, audit):
    data = linear_data if estimator == "ols" else poisson_data
    data.to_stata(tmp_path / "input.dta", write_index=False, version=118)
    plan = make_stata_plan(data, "group", method=method, reps=19, seed=711)
    script = f'do "{SCRIPTS / "legacy.do"}" "{command}" "{method}" 19 711\n'
    run_stata(tmp_path, script)
    reference = pd.read_stata(tmp_path / "legacy.dta").set_index("variable").loc[["x", "z"]]
    actual = bdiff(data, "group", "y ~ x + z", estimator=estimator, method=method,
                   compatibility="stata", coefficients=["x", "z"], plan=plan, verbose=False)
    np.testing.assert_allclose(actual["diff"], reference["diff"], atol=1e-7)
    np.testing.assert_array_equal(actual["stat"], reference["frequency"])
    np.testing.assert_allclose(actual["p_value"], reference["p_value"], atol=1e-14)
    for actual_name, reference_name in (("diff", "diff"), ("stat", "frequency"), ("p_value", "p_value")):
        audit(f"{estimator}_legacy_{method}", actual_name,
              actual[actual_name], reference[reference_name])


@pytest.mark.parametrize("estimator,command,formula", [
    ("ols", "reghdfe y x z, absorb(firm year)", "y ~ x + z | firm + year"),
    ("ppml", "ppmlhdfe y x z, absorb(firm year) tolerance(1e-10)", "y ~ x + z | firm + year"),
])
def test_every_draw_matches_stata_adapter(linear_data, poisson_data, estimator, command, formula, audit):
    data = linear_data if estimator == "ols" else poisson_data
    common = dict(df=data, group="group", formula=formula, method="bootstrap", reps=5,
                  resample_unit="firm", coefficients=["x", "z"], keep_draws=True, verbose=False)
    actual = bdiff(**common, estimator=estimator)
    reference = bdiff(**common, estimator=StataEstimator(command), plan=actual.attrs["plan"])
    np.testing.assert_allclose(actual["diff"], reference["diff"], atol=1e-6)
    np.testing.assert_allclose(actual.attrs["draws"], reference.attrs["draws"], atol=1e-6)
    audit(f"{estimator}_hdfe_bootstrap", "all_draws", actual.attrs["draws"], reference.attrs["draws"])


def test_ppml_separation_and_zero_fe(poisson_data, tmp_path, audit):
    data = poisson_data.copy()
    data.loc[data.firm.isin([0, 16]), "y"] = 0
    data.to_stata(tmp_path / "input.dta", write_index=False, version=118)
    command = "ppmlhdfe y x z, absorb(firm year) tolerance(1e-10)"
    run_stata(tmp_path, reference_script(command, joint="no"))
    reference = pd.read_stata(tmp_path / "reference.dta").set_index("variable")
    actual = bdiff(data, "group", "y ~ x + z | firm + year", estimator="ppml",
                   coefficients=["x", "z"], method="wald", verbose=False)
    np.testing.assert_allclose(actual["diff"], reference["diff"], atol=1e-6)
    assert actual.attrs["metadata"]["n_group0"] == reference.iloc[0].n0
    assert actual.attrs["metadata"]["n_group1"] == reference.iloc[0].n1
    audit("ppml_zero_fe_separation", "diff", actual["diff"], reference["diff"])


def test_ppml_regressor_separation_without_fe(poisson_data, tmp_path, audit):
    data = poisson_data.copy()
    data["sep"] = (np.arange(len(data)) % 11 == 0).astype(int)
    data.loc[data.sep == 1, "y"] = 0
    data.to_stata(tmp_path / "input.dta", write_index=False, version=118)
    command = "ppmlhdfe y x z sep, tolerance(1e-10)"
    run_stata(tmp_path, reference_script(command, joint="no"))
    reference = pd.read_stata(tmp_path / "reference.dta").set_index("variable")
    actual = bdiff(data, "group", "y ~ x + z + sep", estimator="ppml", method="wald",
                   coefficients=["x", "z"], verbose=False)
    np.testing.assert_allclose(actual["diff"], reference["diff"], atol=1e-6)
    assert actual.attrs["metadata"]["n_group0"] == reference.iloc[0].n0
    assert actual.attrs["metadata"]["n_group1"] == reference.iloc[0].n1
    audit("ppml_regressor_separation", "diff", actual["diff"], reference["diff"])


@pytest.mark.parametrize("weight_type,stata_type", [("aweights", "aw"), ("fweights", "fw")])
@pytest.mark.parametrize("cluster", ["", "shared"])
def test_weighted_wald(linear_data, weight_type, stata_type, cluster, tmp_path, audit):
    data = linear_data.copy()
    data["w"] = np.tile([1, 2, 3, 4], len(data) // 4)
    data.to_stata(tmp_path / "input.dta", write_index=False, version=118)
    command = f"regress y x z [{stata_type}=w]"
    run_stata(tmp_path, reference_script(command, cluster=cluster))
    reference = pd.read_stata(tmp_path / "reference.dta").set_index("variable")
    actual = bdiff(data, "group", "y ~ x + z", estimator="ols", method="wald",
                   weights="w", weights_type=weight_type,
                   vcov={"CRV1": cluster} if cluster else "HC1", coefficients=["x", "z"], verbose=False)
    np.testing.assert_allclose(actual["diff"], reference["diff"], atol=1e-10)
    np.testing.assert_allclose(actual["se_diff"], reference["se"], rtol=1e-8)
    audit(f"{weight_type}_suest_{cluster or 'robust'}", "se_diff", actual["se_diff"], reference["se"])


def test_ppml_exposure(poisson_data, tmp_path, audit):
    data = poisson_data.copy()
    data["exposure"] = np.tile([0.5, 1.0, 1.5, 2.0], len(data) // 4)
    data.to_stata(tmp_path / "input.dta", write_index=False, version=118)
    run_stata(tmp_path, reference_script("poisson y x z, exposure(exposure)"))
    reference = pd.read_stata(tmp_path / "reference.dta").set_index("variable")
    actual = bdiff(data, "group", "y ~ x + z", estimator="ppml", exposure="exposure",
                   method="wald", coefficients=["x", "z"], verbose=False)
    np.testing.assert_allclose(actual["diff"], reference["diff"], atol=1e-7)
    np.testing.assert_allclose(actual["se_diff"], reference["se"], rtol=1e-6)
    audit("ppml_exposure", "se_diff", actual["se_diff"], reference["se"])


@pytest.mark.parametrize("method", ["permutation", "bootstrap"])
@pytest.mark.parametrize("estimator,dependent", [("ols", "wage"), ("logit", "collgrad"), ("iv", "wage")])
def test_original_help_nlsw88_example(estimator, dependent, method, tmp_path, audit):
    setup = ('sysuse nlsw88, clear\nkeep wage collgrad ttl_exp married south hours age union tenure grade\n'
             'drop if missing(union)\nrename union group\nsave "input.dta", replace\n'
             'file open receipt using "success.txt", write replace\n'
             'file write receipt "success"\nfile close receipt\n')
    run_stata(tmp_path, setup)
    data = pd.read_stata(tmp_path / "input.dta", convert_categoricals=False)
    controls = ["ttl_exp", "married", "south", "hours", "age"]
    n0 = len(data[(data.group == 0)].dropna(subset=[dependent, *controls]))
    command = f"{'regress' if estimator == 'ols' else 'logit'} {dependent} " + " ".join(controls)
    formula = f"{dependent} ~ " + " + ".join(controls)
    native_estimator = estimator
    if estimator == "iv":
        command = "ivregress 2sls wage " + " ".join(controls) + " (tenure=grade collgrad)"
        formula += " | tenure ~ grade + collgrad"
        native_estimator = "ols"
        n0 = len(data[data.group == 0].dropna(subset=[dependent, *controls, "tenure", "grade", "collgrad"]))
        controls = [*controls, "tenure"]
    plan = make_stata_plan(data, "group", method=method, reps=19, seed=899, n_group0=n0)
    script = f'do "{SCRIPTS / "legacy.do"}" "{command}" "{method}" 19 899\n'
    run_stata(tmp_path, script)
    reference = pd.read_stata(tmp_path / "legacy.dta").set_index("variable").loc[controls]
    actual = bdiff(data, "group", formula,
                   estimator=native_estimator, compatibility="stata", method=method,
                   coefficients=controls, plan=plan, verbose=False)
    np.testing.assert_allclose(actual["diff"], reference["diff"], atol=1e-6)
    np.testing.assert_array_equal(actual["stat"], reference["frequency"])
    np.testing.assert_allclose(actual["p_value"], reference["p_value"], atol=1e-14)
    audit(f"nlsw88_{estimator}_{method}", "frequency", actual["stat"], reference["frequency"])
