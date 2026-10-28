# Usage guide

Tests whether regression coefficients differ between two groups. Version 0.2
rebuilds the estimation, sampling, and inference engines, adds PPML and portable
draw replay, and retains the original DataFrame result interface.

## Install

```sh
pip install pybdiff
```

For development setup, see [development.md](development.md).

## PPML with fixed effects and clustered bootstrap

```python
from pybdiff import bdiff

result = bdiff(
    df, group="treated", formula="y ~ x1 + x2 | firm + year",
    estimator="ppml", vcov={"CRV1": "firm"},
    method="bootstrap", resample_unit="firm",
    coefficients=["x1"], reps=999, seed=42, n_jobs=4,
)
```

PPML estimates the nonnegative outcome in levels, including zeros and fractional
values. It supports absorbed effects, weights, `offset="column"`, and
`exposure="positive_column"`. All-zero effect groups and regressor separation
are checked; estimation-sample exclusions are recorded. It never substitutes
OLS on `log(y)` for Poisson pseudo-maximum likelihood.

Built-in estimators are `ols`, `poisson`/`ppml`/`ppmlhdfe`, `logit`, and `probit`.
The `ppmlhdfe` name selects the native Poisson adapter; use `StataEstimator` below
to execute the actual Stata command.

The difference is always **group 0 minus group 1**. `group` must be 0/1;
missing groups are excluded and other values are rejected. Inputs are not
modified. String columns retain their categories and missing values.

## Methods

| Mode | Modern default | `compatibility="stata"` |
|---|---|---|
| `permutation` | Exchange whole sampling units between groups | Original random regrouping principle |
| `bootstrap` | Sample independently inside each original group | Resample the pooled sample, then regroup |
| `wald` | Joint score-sandwich covariance | Same principle as `suest` |

Let the observed difference be D and a simulated difference be D*. Modern
permutation tests compare `abs(D*) >= abs(D)`; modern bootstrap tests compare
`abs(D* - D) >= abs(D - null)`. Reported Monte Carlo probabilities use
`(tail_count + 1) / (valid_reps + 1)`. Bootstrap also returns its standard error
and basic bootstrap confidence interval. Directional alternatives are `greater`
and `less`, using the same sign convention.

Stata 1.04 reports `min(q, 1-q)`, where `q = mean(D* >= D)`. The `stata` mode
preserves that smaller-tail probability, including its possible zero values.
It is labelled `legacy`; it is not relabelled an ordinary two-sided p-value.
An explicit `alternative="two-sided"` selects the corrected absolute-tail rule.

Wald inference includes cross-group covariance when clusters span both groups.
It uses the union-sample/cluster correction of `suest`, and returns individual
chi-square tests plus a joint test in `result.attrs["metadata"]["joint_wald"]`.
Probit uses observed likelihood curvature for this joint covariance. Explicit
`wald_vcov="independent"` instead sums model covariances; shared clusters are
rejected in that mode. IV can be estimated via an OLS/IV formula for resampling;
joint IV Wald tests require a custom adapter with valid influence contributions.

## Sampling and data

`vcov` and `resample_unit` have separate meanings. Modern `resample_unit="auto"`
uses a single covariance cluster if it is nested in group. Shared or multiway
clusters require an explicit sampling design. `resample_unit=None` explicitly
requests row sampling; `strata=["region"]` keeps assignment counts within strata.
Permutation requires exchangeability under the chosen assignment mechanism.

Bootstrap duplicates of a sampled panel receive distinct panel IDs, so repeated
draws do not share an absorbed effect accidentally. Both group and strata must
be constant inside a sampling unit. The default modern sampling pool is the
union of the two actual estimation samples, after missingness, singleton and
separation exclusions. Stata mode defaults to the original pool and reproduces
the first model's `e(sample)` complement rule. `sample` can override this choice.

## Exact replay against Stata

The same integer seed does not create the same draws in NumPy and Stata.
Use Stata's actual draw plan when reproducing a Stata example:

```python
from pybdiff import bdiff, make_stata_plan

plan = make_stata_plan(df, "treated", method="bootstrap", reps=99, seed=123)
plan.save("draw_plan.json")
result = bdiff(
    df, "treated", "y ~ x1 + x2", method="bootstrap",
    compatibility="stata", plan=plan, coefficients=["x1"],
)
```

For a baseline with excluded rows, supply its Stata `e(N)` through
`make_stata_plan(..., n_group0=...)`. This exporter matches the original
cross-sectional RNG, float sorting, and pooled-bootstrap sequence. It does not
copy implementation errors in the original `xt*` branch. Explicit cluster plans
can instead be constructed with `SamplingPlan`/`Draw` and replayed in both engines.

Any resampling call with `keep_draws=True` returns its simulated differences in
`result.attrs["draws"]` and the plan in `result.attrs["plan"]`. Plans can be saved,
loaded and replayed across worker counts; data/order fingerprints prevent replay
against a different sample. Default NumPy plans store one seed per replication,
rather than an observations-by-replications index matrix.

Native estimates agree to numerical solver tolerance, rather than bit-for-bit
across different likelihood solvers. To retain a Stata estimator's exact model,
including commands without Python equivalents, execute it directly:

```python
from pybdiff import StataEstimator

result = bdiff(
    df, "treated", estimator=StataEstimator(
        "ppmlhdfe y x1 x2, absorb(firm year) vce(cluster firm)"
    ), method="bootstrap", resample_unit="firm", coefficients=["x1"],
    reps=99, n_jobs=4,
)
```

This optional adapter requires licensed Stata (`STATA_BIN` or an executable
path). Its command defines all estimation options; Python formula/vcov do not
rewrite the command. It extracts `e(b)`, `e(V)` and `e(sample)`. Joint score-based
Wald inference requires an adapter providing scores, or explicit independent
inference where its assumptions hold. `Estimate` defines the custom-adapter
contract for other Python estimators.

## Results and diagnostics

The DataFrame contains `b_group0`, `b_group1`, `diff`, `stat`, `stat_label`,
`p_value`; Wald/bootstrap add `se_diff`, `ci_low`, `ci_high`. Resampling adds
`valid_reps` and `mc_se`. Metadata records the input and estimation samples,
excluded rows, omitted coefficients, tail definition, sampling units, failures,
and warnings emitted inside draws. Failures raise by default. Explicit
`on_error="skip"` requires a minimum valid fraction and retains every failure.

## Validation and performance

```sh
uv run pytest -m "not stata"
uv run pytest tests --audit-dir docs/validation --junitxml=.local/tests.xml
uv run python benchmarks/benchmark.py --output .local/benchmarks
uv run python benchmarks/plot_benchmark.py --input .local/benchmarks
```

The full suite requires Stata, SSC `bdiff` 1.04, `reghdfe`, and `ppmlhdfe`.
It tests original bdiff probabilities using Stata's actual RNG draws, joint
`suest` covariance including shared clusters and weights, and each fixed-effect
OLS/PPML bootstrap draw against the actual Stata estimators. It also checks
Poisson separation, exposure, missingness, categorical data and worker-count
invariance. Published execution evidence is under `docs/validation/`.

Speed measurements and all individual trials are under `docs/benchmarks/`; its
`summary.csv` reports medians, first-call times and speedups. Comparisons use the
same data, model, sampling principle and replication count; the metadata states
thread budgets, timing scope and RNG differences. The benchmark includes slower
configurations as well as faster ones. The PNG/PDF visualize measured times.

Open the [benchmark documentation](benchmarks/README.md) or
[validation summary](validation/summary.json).

## Attribution

Original Stata command: Yujun Lian (连玉君), *bdiff: Coefficient Difference Test
across Two Groups*, version 1.04, 24 Nov 2020. Python implementation: Zhiyu Lu.
The Stata source is not bundled or required by the native engine.

References: [SSC record](https://ideas.repec.org/c/boc/bocode/s458202.html),
[author's algorithm audit](https://lianxhcn.github.io/FinEco/lectures/bootstrap.html#bdiff-的版本核验),
[Stata suest manual](https://www.stata.com/manuals/rsuest.pdf),
[ppmlhdfe documentation](https://scorreia.com/help/ppmlhdfe.html),
[PyFixest](https://pyfixest.org/).

MIT license. See `LICENSE`.
