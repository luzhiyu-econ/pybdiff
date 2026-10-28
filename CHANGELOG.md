# Changelog

## 0.2.0 - 2026-10-23

Rebuild coefficient-difference inference and add PPML, parallel draw execution,
Stata draw replay, and reproducible performance comparisons.

### Changes

- Add OLS, PPML/Poisson with fixed effects, Logit, and Probit adapters.
- Support PPML separation checks, offset, exposure, and weights.
- Add sampling units, strata, estimation-sample diagnostics, and saved draw plans.
- Preserve shared-cluster covariance in joint Wald inference.
- Add an optional adapter that executes licensed Stata commands directly.
- Keep draw results invariant across Python worker counts.
- Publish synthetic-data generators, all benchmark timings, and a speed chart.
- Use a `src/` layout and explicit wheel/source-distribution file selection.

### Migration from 0.1

- The difference remains group 0 minus group 1 and results remain a DataFrame.
- Modern two-sided permutation tests use the absolute tail with a Monte Carlo
  correction. Bootstrap uses centered draws and independent within-group draws.
- `compatibility="stata"` explicitly selects bdiff 1.04's pooled bootstrap and
  smaller-tail probability. This probability is labelled `legacy`.
- The default sampling pool now uses actual estimation samples. With clustered
  covariance, modern automatic resampling selects eligible whole clusters.
- Wald inference accounts for clusters shared by the two groups.
- Use exported Stata plans for identical draws; equal seeds alone are insufficient.
- Native likelihood solvers agree within numerical tolerance. Use `StataEstimator`
  when reproducing the exact original Stata estimation command is required.
- The supported dependency range starts at PyFixest 0.60 and Python 3.10.

See [the usage guide](docs/usage.md) for the complete options and definitions.

## 0.1.0

Initial release with permutation, bootstrap, and Wald coefficient comparisons.
