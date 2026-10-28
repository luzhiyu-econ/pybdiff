"""Measure Stata bdiff and serial/parallel Python on exactly the same panel."""

import argparse
import json
import os
import platform
import re
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from generate_data import make_panel

from pybdiff import bdiff
from pybdiff.stata import run_stata, stata_executable


def stata_timings(data, command, cores, config):
    with tempfile.TemporaryDirectory(prefix="pybdiff-benchmark-") as temporary:
        root = Path(temporary)
        data.to_stata(root / "input.dta", write_index=False, version=118)
        script = (f'local command `"{command}"\'\nlocal reps {config.reps}\n'
                  f'local trials {config.trials}\nlocal seed {config.seed}\n'
                  f'local cores {cores}\ninclude "{Path(__file__).with_suffix(".do")}"\n')
        start = time.perf_counter()
        log = run_stata(root, script, timeout=1800)
        process_seconds = time.perf_counter() - start
        table = pd.read_stata(root / "timings.dta", convert_categoricals=False)
        (config.output / f"stata_{config.case}_{cores}.log").write_text(log, encoding="utf-8")
    table["engine"], table["workers"], table["case"] = "Stata", cores, config.case
    table["process_seconds"] = process_seconds
    return table


def python_timings(data, estimator, jobs, config):
    records = []
    for trial in range(config.trials + 1):
        start = time.perf_counter()
        result = bdiff(data, "group", "y ~ x + z | firm + year", estimator=estimator,
                       compatibility="stata", method="permutation", reps=config.reps,
                       coefficients=["x", "z"], n_jobs=jobs, seed=config.seed + trial,
                       verbose=False)
        seconds = time.perf_counter() - start
        records.append({"trial": trial, "seconds": seconds,
                        "diff_x": result.loc["x", "diff"], "diff_z": result.loc["z", "diff"],
                        "engine": "Python", "workers": jobs, "case": config.case,
                        "process_seconds": np.nan})
        print(f"  Python {jobs} workers, trial {trial}: {seconds:.3f}s", flush=True)
    return pd.DataFrame(records)


def summarize(raw, config):
    steady = raw[raw.trial > 0]
    summary = steady.groupby(["case", "engine", "workers"], sort=False)["seconds"].agg(
        median_seconds="median", min_seconds="min", max_seconds="max").reset_index()
    first = raw[raw.trial == 0].rename(columns={"seconds": "first_call_seconds"})
    summary = summary.merge(first[["case", "engine", "workers", "first_call_seconds"]],
                            on=["case", "engine", "workers"])
    for case in summary.case.unique():
        selected = summary.case == case
        baseline = summary[selected & (summary.engine == "Stata")
                           & (summary.workers == max(config.stata_cores))].median_seconds.iloc[0]
        serial = summary[selected & (summary.engine == "Python")
                         & (summary.workers == 1)].median_seconds.iloc[0]
        summary.loc[selected, "speedup_vs_stata"] = baseline / summary.loc[selected, "median_seconds"]
        summary.loc[selected, "speedup_vs_python_serial"] = serial / summary.loc[selected, "median_seconds"]
    summary["observations"], summary["replications"] = config.firms * config.years, config.reps
    summary["timed_trials"] = config.trials
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firms", type=int, default=1000)
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--reps", type=int, default=99)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--stata-cores", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output", type=Path, default=Path(".local/benchmarks"))
    parser.add_argument("--reuse-stata", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=["OLS", "PPML"], default=["OLS", "PPML"])
    config = parser.parse_args()
    config.output.mkdir(parents=True, exist_ok=True)
    data = make_panel(config.firms, config.years, config.seed)
    data.to_stata(config.output / "input_panel.dta", write_index=False, version=118)
    tables = []
    if config.reuse_stata:
        previous = json.loads((config.output / "metadata.json").read_text())
        for key in ("firms", "years", "reps", "trials", "seed"):
            if previous[key] != getattr(config, key):
                raise ValueError(f"Cannot reuse Stata timings with a different {key}")
        prior = pd.read_csv(config.output / "raw_timings.csv")
        tables = [prior[(prior.engine == "Stata") | ~prior.case.isin(config.cases)]]
    for estimator, outcome, command in [
        ("ols", "y_linear", "reghdfe y x z, absorb(firm year) tolerance(1e-10)"),
        ("ppml", "y", "ppmlhdfe y x z, absorb(firm year) tolerance(1e-10)"),
    ]:
        config.case = estimator.upper()
        if config.case not in config.cases:
            continue
        sample = data.assign(y=data[outcome])
        print(f"{config.case}: {len(sample):,} rows, {config.reps} draws", flush=True)
        for cores in config.stata_cores:
            if config.reuse_stata:
                continue
            print(f"  Starting Stata MP {cores} cores", flush=True)
            tables.append(stata_timings(sample, command, cores, config))
        for jobs in config.jobs:
            tables.append(python_timings(sample, estimator, jobs, config))
        pd.concat(tables, ignore_index=True).to_csv(config.output / "raw_timings.csv", index=False)
    raw = pd.concat(tables, ignore_index=True)
    summary = summarize(raw, config)
    summary.to_csv(config.output / "summary.csv", index=False)
    _save_metadata(raw, config)
    print(summary.to_string(index=False), flush=True)


def _save_metadata(raw, config):
    errors = {}
    for case in raw.case.unique():
        rows = raw[raw.case == case]
        reference = rows[rows.engine == "Stata"][["diff_x", "diff_z"]].iloc[0].to_numpy()
        error = float(np.max(abs(rows[["diff_x", "diff_z"]].to_numpy() - reference)))
        errors[case] = error
        if error > 1e-6:
            raise AssertionError(f"{case}: observed coefficient differences disagree by {error}")
    cpuinfo = Path("/proc/cpuinfo")
    cpu_lines = cpuinfo.read_text().splitlines() if cpuinfo.exists() else []
    cpu = next((line.split(":", 1)[1].strip() for line in cpu_lines
                if line.startswith("model name")), platform.processor())
    metadata = {"observations": config.firms * config.years, "firms": config.firms,
                "years": config.years, "reps": config.reps, "trials": config.trials,
                "method": "permutation", "compatibility": "stata",
                "sampling_unit": "row", "fixed_effects": ["firm", "year"],
                "covariates": ["x", "z"], "seed": config.seed,
                "case_outcomes": {"OLS": "y_linear", "PPML": "y"},
                "cpu": cpu, "logical_cpus": os.cpu_count(),
                "load_average_at_finish": os.getloadavg() if hasattr(os, "getloadavg") else None,
                "platform": platform.system(),
                "versions": {name: version(name) for name in ("pybdiff", "pyfixest", "numpy", "pandas")},
                "stata_versions": _stata_environment(config.output),
                "max_observed_difference_error": errors,
                "timing": "Full command after data load; Python includes worker startup. "
                          "Trial 0 is first call; medians use subsequent trials in the same process.",
                "random_draws": "Engine-specific RNGs; identical data, model, method, and draw count. "
                                "Exact draw equivalence is tested separately with exported Stata plans.",
                "stata_baseline_cores": max(config.stata_cores),
                "python_threads_per_worker": 1}
    (config.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _stata_environment(output):
    installation = Path(stata_executable()).parent
    versions = {}
    logs = sorted(output.glob("stata_*.log"))
    if logs:
        match = re.search(r"Stata(?:Now)?\s+[0-9]+\.[0-9]+", logs[0].read_text())
        versions["stata"] = match.group(0) if match else None
    for name in ("bdiff", "reghdfe", "ppmlhdfe"):
        path = installation / "ado" / "plus" / name[0] / f"{name}.ado"
        versions[name] = path.read_text(errors="replace").splitlines()[0] if path.exists() else None
    return versions


if __name__ == "__main__":
    main()
