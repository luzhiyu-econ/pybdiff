# pybdiff

Python replication of Stata's `bdiff` command. Tests whether regression coefficients differ across two subgroups using permutation tests, bootstrap tests, or parametric Wald tests — with optional parallel execution.

## Attribution

This package is a Python reimplementation of the Stata command `bdiff` originally written by:

> Lian, Yujun (连玉君). *bdiff*: Coefficient Difference Test across Two Groups.
> Version 1.04, 24 Nov 2020. Sun Yat-sen University.
> Email: arlionn@163.com · Blog: https://www.lianxh.cn

All methodological credit belongs to the original author. The bootstrap and permutation procedures follow:

> Efron, B., Tibshirani, R., 1993. *An Introduction to the Bootstrap*. Chapman & Hall.

## Installation

```bash
pip install pybdiff
```

## Quick Start

```python
from pybdiff import bdiff

result = bdiff(
    df      = df,
    group   = "treated",          # column with values 0 and 1
    formula = "y ~ x1 + x2 | firm + year",
    vcov    = {"CRV1": "firm"},
    method  = "permutation",      # "permutation", "bootstrap", or "wald"
    reps    = 500,
    seed    = 42,
    n_jobs  = -1,                 # use all CPU cores
)
print(result)
```

## Methods

| Method | Type | Description |
|--------|------|-------------|
| `permutation` | Non-parametric | Randomly shuffles group labels (default) |
| `bootstrap` | Non-parametric | Resamples each group with replacement |
| `wald` | Parametric | Chi-squared test using block-diagonal VCV |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | Dataset with a 0/1 group column |
| `group` | `str` | — | Name of the grouping variable |
| `formula` | `str` | — | PyFixest formula, e.g. `"y ~ x1 + x2 \| fe"` |
| `vcov` | `str \| dict` | `"iid"` | Variance-covariance type |
| `method` | `str` | `"permutation"` | Test method |
| `reps` | `int` | `500` | Resampling iterations (ignored for Wald) |
| `seed` | `int \| None` | `42` | Master random seed |
| `n_jobs` | `int` | `1` | Parallel workers (`-1` = all cores) |
| `verbose` | `bool` | `True` | Print progress bar and results table |

## Returns

`pd.DataFrame` indexed by variable name with columns:

- `b_group0`, `b_group1` — estimated coefficients per group
- `diff` — coefficient difference (b0 − b1)
- `stat` — test statistic (frequency count or chi-squared)
- `p_value` — two-sided p-value
- `se_diff` — standard error of the difference (Wald only)
- `valid_reps` — successful resampling iterations (resampling only)

## Dependencies

- [pyfixest](https://github.com/py-econometrics/pyfixest)
- pandas, numpy, scipy, joblib, tqdm

## License

MIT
