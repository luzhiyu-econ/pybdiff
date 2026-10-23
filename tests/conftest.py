"""Small reproducible data with both nested and shared clusters."""

import json
import os
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(7181)
    n = 160
    group = np.repeat([0, 1], n // 2)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    firm = np.repeat(np.arange(32), 5)
    return pd.DataFrame({"y": 1 + (0.4 + 1.2 * group) * x + 0.3 * z
                         + rng.normal(scale=0.5, size=n),
                         "x": x, "z": z, "group": group, "firm": firm,
                         "year": np.tile(np.arange(5), 32),
                         "shared": np.tile(np.arange(20), 8)})


@pytest.fixture
def poisson_data(linear_data):
    rng = np.random.default_rng(1135)
    data = linear_data.copy()
    mu = np.exp(0.3 + 0.15 * data.x + 0.4 * data.group * data.x
                + 0.04 * data.year)
    data["y"] = rng.poisson(mu)
    return data


def pytest_addoption(parser):
    parser.addoption("--audit-dir", default=None, help="Write executed comparison evidence")


@pytest.fixture
def audit(request):
    if not hasattr(request.config, "_bdiff_audit"):
        request.config._bdiff_audit = []

    def record(scenario, metric, actual, reference):
        a, b = np.asarray(actual, dtype=float).ravel(), np.asarray(reference, dtype=float).ravel()
        request.config._bdiff_audit.append({"scenario": scenario, "metric": metric,
                                           "compared_values": len(a),
                                           "max_absolute_error": float(np.max(np.abs(a - b)))})
    return record


def pytest_sessionfinish(session, exitstatus):
    destination = session.config.getoption("--audit-dir")
    if destination is None:
        return
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    rows = getattr(session.config, "_bdiff_audit", [])
    pd.DataFrame(rows).to_csv(root / "stata_comparisons.csv", index=False)
    summary = {"exit_code": int(exitstatus), "collected_tests": session.testscollected,
               "failed_tests": session.testsfailed, "comparison_records": len(rows),
               "data_scope": "deterministic synthetic samples and Stata nlsw88 help examples",
               "versions": {name: version(name) for name in ("pybdiff", "pyfixest", "numpy", "pandas")}}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
