"""Exercise an installed distribution from outside the source checkout."""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

import pybdiff
from pybdiff import bdiff

assert "site-packages" in str(Path(pybdiff.__file__).resolve()), pybdiff.__file__
assert pybdiff.__version__ == version("pybdiff")
rng = np.random.default_rng(20261023)
n = 160
group = np.repeat([0, 1], n // 2)
x, z = rng.normal(size=(2, n))
data = pd.DataFrame({"group": group, "x": x, "z": z})
data["y"] = 1 + (0.2 + 0.6 * group) * x + 0.3 * z + rng.normal(scale=0.1, size=n)
ols = bdiff(data, "group", "y ~ x + z", method="wald", coefficients=["x"], verbose=False)
assert abs(ols.loc["x", "diff"] + 0.6) < 0.1
data["y"] = rng.poisson(np.exp(0.3 + 0.15 * x + 0.2 * group * x))
ppml = bdiff(data, "group", "y ~ x + z", estimator="ppml", method="permutation",
             coefficients=["x"], reps=3, seed=42, n_jobs=2, verbose=False)
assert ppml.loc["x", "valid_reps"] == 3
assert 0 <= ppml.loc["x", "p_value"] <= 1
print(f"Installed pybdiff {pybdiff.__version__}: OLS and parallel PPML passed")
