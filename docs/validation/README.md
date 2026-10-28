# Executed validation

`summary.json` records the completed suite: 61 tests, zero failures. The six
warnings are expected separation/omitted-regressor diagnostics in constructed
edge cases. The suite includes deterministic synthetic samples and Stata's
`nlsw88` help-file examples for OLS, Logit and instrumental variables.

`stata_comparisons.csv` contains 63 comparison records, including baseline
differences, joint robust/clustered standard errors and probabilities, original
bdiff tail counts, and individual fixed-effect OLS/PPML bootstrap draws.
`.local/tests.xml` and `.local/pytest-release.log` are local execution receipts.

`mcp_comparison.csv` is an additional Stata-MCP execution/readback comparison
on the built-in `auto` data. It compares coefficients, standard errors and
probabilities with the Python joint Wald implementation. Its original Stata
execution log is retained locally.

Reproduce the complete suite with:

```sh
mkdir -p .local
uv run pytest tests --audit-dir docs/validation --junitxml=.local/tests.xml
```

The Stata tests require licensed Stata plus `bdiff`, `reghdfe` and `ppmlhdfe`.
Native-only tests use `uv run pytest -m "not stata"`.
