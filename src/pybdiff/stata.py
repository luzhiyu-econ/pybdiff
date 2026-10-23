"""Optional licensed-Stata adapter for estimators without a native Python equivalent."""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .estimators import Estimate


def stata_executable(explicit=None):
    candidates = [explicit, os.environ.get("STATA_BIN"), shutil.which("stata-mp"),
                  shutil.which("stata-se"), shutil.which("stata"),
                  "/usr/local/stata19/stata-mp"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("Set STATA_BIN or supply the path to a licensed Stata executable")


def run_stata(directory, script, executable=None, timeout=120):
    directory = Path(directory).resolve()
    if '"' in str(directory):
        raise ValueError("Stata working paths cannot contain double quotes")
    path = directory / "run.do"
    (directory / "success.txt").unlink(missing_ok=True)
    path.write_text(f'cd "{directory.as_posix()}"\n' + script, encoding="utf-8")
    completed = subprocess.run([stata_executable(executable), "-b", "do", str(path)],
                               cwd=directory, capture_output=True, text=True, timeout=timeout,
                               check=False)
    log_path = directory / "run.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    if completed.returncode or not (directory / "success.txt").exists():
        raise RuntimeError(f"Stata failed (exit {completed.returncode}):\n"
                           f"{log[-6000:]}\n{completed.stderr[-1000:]}")
    return log


@dataclass(frozen=True)
class StataEstimator:
    command: str
    executable: str | None = None
    timeout: int = 120

    def __call__(self, *, data, formula=None, vcov=None, **options):
        """Execute the supplied Stata command on exactly this draw's input frame.

        Include model options, weights and covariance in ``command``. The bdiff
        formula/vcov arguments do not rewrite Stata syntax. This adapter supplies
        e(b), e(V), and e(sample); custom score extraction remains estimator-specific.
        """
        if options:
            raise ValueError("Put Stata estimator options in StataEstimator.command")
        with tempfile.TemporaryDirectory(prefix="pybdiff-stata-") as temporary:
            root = Path(temporary)
            data.reset_index(drop=True).to_stata(root / "input.dta", write_index=False, version=118)
            run_stata(root, _estimation_script(self.command), self.executable, self.timeout)
            coefficient_data = pd.read_stata(root / "coefficients.dta", convert_categoricals=False)
            raw_names = coefficient_data["variable"].tolist()
            equations = {name.rsplit(":", 1)[0] for name in raw_names if ":" in name}
            if len(equations) <= 1:
                raw_names = [name.rsplit(":", 1)[-1] for name in raw_names]
            names = ["Intercept" if name == "_cons" else name for name in raw_names]
            b = pd.Series(coefficient_data["coefficient"].to_numpy(), index=names)
            v = pd.read_stata(root / "covariance.dta", convert_categoricals=False).to_numpy()
            sample = pd.read_stata(root / "sample.dta", convert_categoricals=False)["used"].to_numpy()
        return Estimate(b, pd.DataFrame(v, index=names, columns=names),
                        data.index.to_numpy()[sample.astype(bool)],
                        diagnostics={"estimator": self.command, "engine": "Stata",
                                     "dropped": int(np.sum(sample == 0))})


def _estimation_script(command):
    return '\n'.join([
        'version 16.0', 'set more off', 'set varabbrev off',
        'capture set processors 1', 'use "input.dta", clear',
        f'quietly {command}', 'matrix B = e(b)', 'matrix V = e(V)',
        'generate byte used = e(sample)', 'preserve', 'keep used',
        'save "sample.dta", replace', 'restore',
        'local names : colfullnames B', 'local k = colsof(B)', 'clear',
        'set obs `k\'', 'generate str244 variable = ""',
        'generate double coefficient = .', 'forvalues i = 1/`k\' {',
        '    local name : word `i\' of `names\'',
        '    replace variable = "`name\'" in `i\'',
        '    replace coefficient = B[1, `i\'] in `i\'', '}',
        'save "coefficients.dta", replace', 'clear', 'svmat double V, names(v)',
        'save "covariance.dta", replace',
        'file open receipt using "success.txt", write replace',
        'file write receipt "success"', 'file close receipt',
    ]) + '\n'
