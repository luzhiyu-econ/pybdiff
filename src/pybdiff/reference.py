"""Export Stata's actual draws for cross-language, seed-exact replay."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .sampling import Draw, SamplingPlan, fingerprint
from .stata import run_stata


def make_stata_plan(data, group, *, method="permutation", reps=99, seed=42,
                    n_group0=None, executable=None, timeout=120):
    """Reproduce bdiff 1.04's row draws using Stata's own RNG and sorting.

    The data must be in exactly the same order as the Stata input. Optional
    n_group0 supplies its observed e(N) when the baseline excludes rows.
    This row-level export deliberately does not emulate the original xt-branch
    implementation errors. Portable explicit cluster plans can be supplied too.
    """
    if method not in {"permutation", "bootstrap"}:
        raise ValueError("Stata plans require permutation or bootstrap")
    if not isinstance(reps, int) or reps < 2:
        raise ValueError("reps must be an integer of at least 2")
    if not data[group].isin([0, 1]).all():
        raise ValueError("The Stata plan input must contain only complete 0/1 groups")
    n0 = int((data[group] == 0).sum()) if n_group0 is None else int(n_group0)
    if not 0 < n0 < len(data):
        raise ValueError("n_group0 must leave at least one row in each group")
    with tempfile.TemporaryDirectory(prefix="pybdiff-plan-") as temporary:
        root = Path(temporary)
        frame = pd.DataFrame({"source": np.arange(len(data), dtype=int)})
        frame.to_stata(root / "input.dta", write_index=False, version=118)
        run_stata(root, _plan_script(method, reps, seed, n0), executable, timeout)
        rows = pd.read_stata(root / "draws.dta", convert_categoricals=False)
    draws = []
    for _, selected in rows.groupby("rep", sort=True):
        selected = selected.sort_values("position")
        draws.append(Draw(selected.loc[selected.drawgroup == 0, "source"].to_numpy(dtype=int),
                          selected.loc[selected.drawgroup == 1, "source"].to_numpy(dtype=int)))
    return SamplingPlan(draws, len(data), method, data_fingerprint=fingerprint(data),
                        source=f"Stata bdiff 1.04 row RNG; seed={seed}")


def _plan_script(method, reps, seed, n0):
    lines = [
        'version 13', 'set more off', 'set varabbrev off', 'capture set processors 1',
        'use "input.dta", clear', 'save "original.dta", replace', f'set seed {int(seed)}',
        'postfile handle long rep long position long source byte drawgroup using "draws.dta", replace',
        f'forvalues rep = 1/{reps} {{',
    ]
    if method == "bootstrap":
        lines += ['    use "original.dta", clear', '    bsample']
    lines += [
        '    generate float random = runiform()', '    sort random',
        '    local n = _N', '    forvalues i = 1/`n\' {',
        f'        post handle (`rep\') (`i\') (source[`i\']) (`i\' > {n0})',
        '    }', '    drop random', '}', 'postclose handle',
        'file open receipt using "success.txt", write replace',
        'file write receipt "success"', 'file close receipt',
    ]
    return '\n'.join(lines) + '\n'
