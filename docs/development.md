# Development and release

The repository uses a `src/` layout with Hatchling. Keep the project environment
on local storage; generated benchmark data and execution receipts are ignored.

```sh
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/pybdiff"
uv sync --frozen --group dev --group benchmark
uv run ruff check src tests examples benchmarks scripts
uv run pytest -m "not stata"
```

For the full Stata comparison suite, install licensed Stata plus `bdiff`,
`reghdfe`, and `ppmlhdfe`; set `STATA_BIN` when needed:

```sh
mkdir -p .local
uv run pytest tests --audit-dir docs/validation --junitxml=.local/tests.xml
```

Performance data, timing scope, all measured configurations, and reproduction
commands are in [benchmarks/README.md](benchmarks/README.md).

## Package checks

```sh
uv build
uv run twine check --strict dist/*.whl dist/*.tar.gz
uv run python scripts/check_dist.py
```

The wheel contains runtime code and required metadata. The source distribution
also contains tests and the standalone quick-start example. Benchmarks, images,
local environments, logs, and datasets are excluded explicitly. `uv.lock` stays
in GitHub for development reproducibility and is not shipped to PyPI.

CI tests Python 3.10, 3.11, and 3.12. Each job installs the wheel, verifies its
import path, runs OLS/parallel PPML smoke checks and native tests outside the
checkout, rebuilds a wheel from the source archive, and checks that installation.

## Release

The only version source is `src/pybdiff/__init__.py`. Update it together with the
changelog and versioned documentation links. Push a branch, open a pull request,
and merge after CI passes. Then create a matching `vX.Y.Z` tag on main.

`publish.yml` calls the same CI workflow and uploads its validated Python 3.12
artifacts through PyPI Trusted Publishing. The publish job depends on all checks;
it does not rebuild the files before uploading. Keep the `publish.yml` workflow
filename and `pypi` environment consistent with the PyPI publisher configuration.
