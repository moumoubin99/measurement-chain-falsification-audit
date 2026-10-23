"""Shared helpers for run scripts: paths, split, donor pool, parallel map, JSON output."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))          # experiments/

from kincost import chain as C          # noqa: E402
from kincost import io_data as IO       # noqa: E402

PROJ = HERE.parents[3]
DEF_ZIP = os.environ.get("GBVR_ZIP", str(PROJ / "data/collected/gazebasevr/gazebasevr.zip"))
DEF_GB = os.environ.get("GB_ROOT", str(PROJ / "data/collected/gazebase"))
RES = Path(os.environ.get("KINCOST_RESULTS", str(PROJ / "experiments/results")))

E_SET, E_WORST = C.settling_exclusion_ms()


def gbvr_split(zip_path=DEF_ZIP):
    idx = IO.gbvr_index(zip_path)
    dev, conf = IO.make_split(list(idx), IO.N_D_GBVR, salt=1)
    return idx, dev, conf


def gb_split(root=DEF_GB):
    idx = IO.gb_index(root)
    if not idx:
        return idx, [], []
    dev, conf = IO.make_split(list(idx), IO.N_D_GB, salt=2)
    return idx, dev, conf


def donor_pool(zip_path=DEF_ZIP, n=None):
    idx, dev, _ = gbvr_split(zip_path)
    recs = [IO.load_gbvr(zip_path, idx[p]) for p in (dev if n is None else dev[:n])]
    return C.DonorPool.from_recordings(recs)


def pmap(fn, items, workers):
    if workers <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, items, chunksize=1))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


_T0 = time.perf_counter()


def _resources():
    """CPU time (self + reaped children) at save time; RSS fields are per-process maxima (largest single
    process), not whole-job concurrent memory (R004 records that separately as peak_tree_rss_mib)."""
    import resource
    su, ch = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(cpu_s_self=su.ru_utime + su.ru_stime, cpu_s_children=ch.ru_utime + ch.ru_stime,
                peak_rss_mib_self=su.ru_maxrss / 1024, peak_rss_mib_children_max=ch.ru_maxrss / 1024,
                wall_s=time.perf_counter() - _T0)


def save(run, name, obj, provisional=None):
    """Write obj as JSON with a metadata envelope.  provisional: derived by the caller from the actual data /
    configuration (synthetic or reduced inputs); None falls back to KINCOST_PROVISIONAL (default provisional).
    Non-dict payloads are wrapped as {"data": obj, ...} so they carry the same metadata."""
    d = RES / run
    d.mkdir(parents=True, exist_ok=True)
    if provisional is None:
        provisional = os.environ.get("KINCOST_PROVISIONAL", "1") == "1"
    obj = dict(obj) if isinstance(obj, dict) else dict(data=obj)
    obj.setdefault("_resources", _resources())
    obj.setdefault("_provisional", bool(provisional))
    with open(d / name, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    return d / name


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t
