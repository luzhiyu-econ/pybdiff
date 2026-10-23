# pybdiff

Test whether regression coefficients differ between two groups, with parallel
bootstrap and permutation tests, joint Wald inference, and PPML support.

## Installation

```sh
pip install pybdiff
```

Requires Python 3.10 or later. The native engine runs without Stata.

## Quick start

```python
from pybdiff import bdiff

result = bdiff(
    df, group="treated", formula="y ~ x1 + x2 | firm + year",
    estimator="ppml", vcov={"CRV1": "firm"},
    method="bootstrap", resample_unit="firm",
    coefficients=["x1"], reps=999, seed=42, n_jobs=4,
)
print(result)
```

Coefficient differences are **group 0 minus group 1**. Built-in estimators are
OLS, Poisson/PPML with absorbed fixed effects, Logit, and Probit. Bootstrap and
permutation support sampling units, strata, and reusable draw plans. Joint Wald
inference includes covariance between groups when clusters cross both groups.

See the [usage guide](https://github.com/luzhiyu-econ/pybdiff/blob/v0.2.0/docs/usage.md)
for options, diagnostics, custom estimators, and exact Stata draw replay.

## Stata compatibility

Use `compatibility="stata"` to reproduce the original bdiff 1.04 resampling
principle and its smaller-tail probability. Modern inference is the default;
the two definitions are documented separately. Equal integer seeds in Stata and
NumPy do not produce equal draws. `make_stata_plan` exports Stata's actual draws;
`StataEstimator` can execute the original Stata regression command directly.
These optional integrations require licensed Stata.

Version 0.2 changes inference defaults and fixes problems in 0.1. Read the
[migration notes](https://github.com/luzhiyu-econ/pybdiff/blob/v0.2.0/CHANGELOG.md)
before comparing results from different versions.

## Measured performance

![OLS and PPML runtimes: Stata/MP and serial or parallel Python](https://raw.githubusercontent.com/luzhiyu-econ/pybdiff/v0.2.0/docs/benchmarks/runtime.png)

Synthetic panel: 20,000 firm-year rows, two regressors, firm/year effects,
99 row-level permutations, and Stata compatibility mode. Bars show the median
of three subsequent calls; whiskers show their full range. First-call times are
reported separately. Each Python worker uses one computational thread.

| Model | Stata/MP, 8 cores | Python, 8 workers | Speedup |
|---|---:|---:|---:|
| OLS | 14.478 s | 3.878 s | 3.73x |
| PPML | 62.044 s | 8.137 s | 7.62x |

These are measured command runtimes on an AMD EPYC 7B13 host. The
[benchmark documentation](https://github.com/luzhiyu-econ/pybdiff/blob/v0.2.0/docs/benchmarks/README.md)
includes all five configurations per model, first-call times, all 40 timing
observations, data generation, software versions, and the exact timing scope.

## Validation and development

The [validation evidence](https://github.com/luzhiyu-econ/pybdiff/tree/v0.2.0/docs/validation)
records executed Stata comparisons. GitHub CI tests the installed wheel and a
wheel rebuilt from the source distribution. Stata comparisons run separately
with a licensed installation. Setup and reproduction commands are in the
[development guide](https://github.com/luzhiyu-econ/pybdiff/blob/v0.2.0/docs/development.md).

## Attribution and license

Original Stata command: Yujun Lian (连玉君), *bdiff: Coefficient Difference Test
across Two Groups*, version 1.04, 24 Nov 2020
([SSC record](https://ideas.repec.org/c/boc/bocode/s458202.html)).
Python implementation: Zhiyu Lu. Stata's implementation is not bundled.

MIT license. See [LICENSE](https://github.com/luzhiyu-econ/pybdiff/blob/v0.2.0/LICENSE).
