"""Run OLS/PPML difference tests on a reproducible firm-year panel."""

import numpy as np
import pandas as pd

from pybdiff import bdiff


def make_panel(firms=400, years=10, seed=20261023):
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(firms), years)
    year = np.tile(np.arange(years), firms)
    group = (firm >= firms // 2).astype(int)
    x, z = rng.normal(size=(2, len(firm)))
    firm_effect = rng.normal(scale=0.2, size=firms)[firm]
    eta = 0.3 + firm_effect + 0.02 * year + (0.15 + 0.12 * group) * x + 0.2 * z
    return pd.DataFrame({"y": rng.poisson(np.exp(eta)),
                         "y_linear": eta + rng.normal(size=len(firm)),
                         "x": x, "z": z, "group": group, "firm": firm, "year": year})


if __name__ == "__main__":
    data = make_panel()
    result = bdiff(data, "group", "y ~ x + z | firm + year", estimator="ppml",
                   vcov={"CRV1": "firm"}, method="bootstrap", coefficients=["x"],
                   reps=199, n_jobs=4, seed=42, keep_draws=True)
    print(result.attrs["metadata"])
