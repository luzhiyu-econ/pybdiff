"""Explicit sampling units and portable, estimator-independent draw plans."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def fingerprint(data: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes()
    schema = repr([(str(c), str(t)) for c, t in zip(data.columns, data.dtypes)]).encode()
    return hashlib.sha256(schema + values).hexdigest()


@dataclass
class Draw:
    index0: np.ndarray
    index1: np.ndarray
    labels0: np.ndarray | None = None
    labels1: np.ndarray | None = None

    def __post_init__(self):
        for name in ("index0", "index1", "labels0", "labels1"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value))


@dataclass
class SamplingPlan:
    """Draw indices refer to the frozen estimation sample, in its row order."""

    draws: list[Draw]
    n_rows: int
    method: str
    resample_unit: str | None = None
    strata: tuple[str, ...] = ()
    data_fingerprint: str | None = None
    source: str = "numpy"
    seeds: list[int] | None = None
    compatibility: str = "modern"
    n_group0: int | None = None

    def __len__(self):
        return len(self.draws) if self.seeds is None else len(self.seeds)

    def draw_at(self, index, data, group):
        if self.seeds is None:
            return self.draws[index]
        units, labels, blocks = _blocks(data, group, self.resample_unit, self.strata)
        return _draw_for_seed(self.seeds[index], units, labels, blocks,
                              self.method, self.compatibility, self.resample_unit, self.n_group0)

    def save(self, path):
        payload = dict(vars(self), draws=[{k: None if v is None else v.tolist()
                       for k, v in vars(d).items()} for d in self.draws])
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["draws"] = [Draw(**{k: None if v is None else np.asarray(v, dtype=int)
                            for k, v in d.items()}) for d in payload["draws"]]
        payload["strata"] = tuple(payload["strata"])
        return cls(**payload)


    def validate(self, data, method, resample_unit, strata):
        if self.n_rows != len(data) or self.method != method:
            raise ValueError("Sampling plan row count or method does not match")
        if self.resample_unit != resample_unit or self.strata != tuple(strata):
            raise ValueError("Sampling plan units or strata do not match")
        if self.data_fingerprint and self.data_fingerprint != fingerprint(data):
            raise ValueError("Sampling plan data/order fingerprint does not match")
        if not len(self):
            raise ValueError("Sampling plan contains no draws")
        if self.seeds is not None:
            if self.draws or any(not isinstance(s, int) or isinstance(s, bool) or s < 0
                                 for s in self.seeds):
                raise ValueError("Lazy plans require nonnegative integer seeds and no explicit draws")
        if self.n_group0 is not None and not 0 < self.n_group0 < len(data):
            raise ValueError("The plan group-0 size is out of range")
        for draw in self.draws:
            for index, labels in ((draw.index0, draw.labels0), (draw.index1, draw.labels1)):
                a = np.asarray(index)
                if a.ndim != 1 or len(a) == 0 or a.dtype.kind not in "iu":
                    raise ValueError("Draw indices must be nonempty integer vectors")
                if np.any(a < 0) or np.any(a >= len(data)):
                    raise ValueError("Draw indices are out of range")
                if labels is not None and len(labels) != len(a):
                    raise ValueError("Draw cluster labels must align with row indices")
            if method == "permutation":
                pooled = np.concatenate([draw.index0, draw.index1])
                if len(pooled) != len(data) or len(np.unique(pooled)) != len(data):
                    raise ValueError("A permutation must partition all rows exactly once")


def _blocks(data, group, unit, strata):
    if unit is None:
        units = None
        first = np.arange(len(data))
    else:
        grouped = data.groupby(unit, sort=False, observed=True, dropna=False).indices
        units = [np.asarray(index, dtype=int) for index in grouped.values()]
        first = np.array([index[0] for index in units])
        for column in [group, *strata]:
            if data.groupby(unit, observed=True)[column].nunique().gt(1).any():
                raise ValueError(f"{column!r} must be constant within resample_unit={unit!r}")
    labels = data.iloc[first][group].to_numpy()
    if strata:
        cells = pd.MultiIndex.from_frame(data.iloc[first][list(strata)])
        codes, _ = pd.factorize(cells, sort=False)
    else:
        codes = np.zeros(len(first), dtype=int)
    blocks = [np.flatnonzero(codes == code) for code in np.unique(codes)]
    return units, labels, blocks


def make_plan(data, group, *, method="permutation", reps=999, seed=42,
              resample_unit=None, strata=(), compatibility="modern",
              store_indices=False, n_group0=None) -> SamplingPlan:
    """Generate permutation, within-group bootstrap, or Stata pooled-bootstrap draws."""
    if method not in {"permutation", "bootstrap"}:
        raise ValueError("Sampling plans require permutation or bootstrap")
    if not isinstance(reps, int) or isinstance(reps, bool) or reps < 1:
        raise ValueError("reps must be a positive integer")
    strata = (strata,) if isinstance(strata, str) else tuple(strata)
    if n_group0 is not None and (strata or resample_unit is not None):
        raise ValueError("n_group0 overrides apply only to unstratified row draws")
    units, labels, blocks = _blocks(data, group, resample_unit, strata)
    seeds = [int(child.generate_state(1, dtype=np.uint64)[0])
             for child in np.random.SeedSequence(seed).spawn(reps)]
    plan = SamplingPlan([], len(data), method, resample_unit, strata,
                        fingerprint(data), "numpy", seeds, compatibility, n_group0)
    if store_indices:
        plan.draws = [_draw_for_seed(s, units, labels, blocks, method,
                      compatibility, resample_unit, n_group0) for s in seeds]
        plan.seeds = None
    return plan


def _draw_for_seed(seed, units, labels, blocks, method, compatibility, unit, n_group0=None):
    rng = np.random.default_rng(seed)
    selections = [[], []]
    for block in blocks:
        n0 = int(np.sum(labels[block] == 0)) if n_group0 is None else int(n_group0)
        if method == "permutation":
            sampled = rng.permutation(block)
            parts = [sampled[:n0], sampled[n0:]]
        elif compatibility == "stata":
            sampled = rng.choice(block, size=len(block), replace=True)
            sampled = rng.permutation(sampled)
            parts = [sampled[:n0], sampled[n0:]]
        else:
            parts = [rng.choice(block[labels[block] == g],
                     size=int(np.sum(labels[block] == g)), replace=True) for g in (0, 1)]
        for g in (0, 1):
            selections[g].extend(parts[g].tolist())
    if units is None:
        return Draw(*(np.asarray(selected, dtype=int) for selected in selections))
    indices, new_labels = [], []
    offset = 0
    for selected in selections:
        indices.append(np.concatenate([units[i] for i in selected]))
        new_labels.append(np.concatenate([np.full(len(units[i]), j + offset)
                          for j, i in enumerate(selected)]))
        offset += len(selected)
    relabel = method == "bootstrap" and unit is not None
    return Draw(*indices, *(new_labels if relabel else [None, None]))


def materialize(data, draw, unit):
    frames = []
    for index, labels in ((draw.index0, draw.labels0), (draw.index1, draw.labels1)):
        frame = data.iloc[index].copy().reset_index(drop=True)
        if labels is not None and unit:
            frame[unit] = labels  # 每次抽中的面板副本获得独立 ID。
        frames.append(frame)
    return frames
