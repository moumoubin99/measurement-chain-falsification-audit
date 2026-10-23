"""Data loading and participant splits for GazeBaseVR (GBVR) and GazeBase (GB).

GBVR file:  data/S_<R><SSS>_S<sess>_5_RAN.csv   columns n,x,y,lx,ly,rx,ry,xT,yT,zT,...  (n in ms, dva)
GB   file:  .../S_<R><SSS>_S<sess>_RAN.csv       columns n,x,y,val,(dP,lab,)xT,yT       (n in ms, dva, 1000 Hz)

A recording is returned as a dict with keys
  t  (ms, float64), x, y (deg; NaN where invalid), valid (bool), xT, yT, fs_nominal, pid, sess, task
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

GBVR_RE = re.compile(r"S_(\d)(\d{3})_S(\d)_(\d)_([A-Z]{3})\.csv$")
GB_RE = re.compile(r"S_(\d)(\d{3})_S(\d)_([A-Z]{3})\.csv$")

SPLIT_SEED = 20261020
N_D_GBVR = 50
N_D_GB = 100


def _rec(df, xcol, ycol, valid_extra=None, fs=250.0, pid=None, sess=None, task=None):
    t = df["n"].to_numpy(np.float64)
    x = df[xcol].to_numpy(np.float64)
    y = df[ycol].to_numpy(np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    if valid_extra is not None:
        valid &= valid_extra
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)
    return dict(t=t, x=x, y=y, valid=valid,
                xT=df["xT"].to_numpy(np.float64), yT=df["yT"].to_numpy(np.float64),
                fs_nominal=fs, pid=pid, sess=sess, task=task)


def gbvr_index(zip_path: str | Path, rnd: int = 1, sess: int = 1, task: str = "RAN"):
    """Map pid -> member name inside the GBVR zip."""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            m = GBVR_RE.search(name)
            if m and int(m.group(1)) == rnd and int(m.group(3)) == sess and m.group(5) == task:
                out[int(m.group(1) + m.group(2))] = name
    return dict(sorted(out.items()))


def load_gbvr(zip_path, member, eye="left"):
    m = GBVR_RE.search(member)
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(io.BytesIO(z.read(member)))
    xc, yc = ("lx", "ly") if eye == "left" else ("rx", "ry")
    return _rec(df, xc, yc, fs=250.0, pid=int(m.group(1) + m.group(2)), sess=int(m.group(3)), task=m.group(5))


def gb_index(root: str | Path, rnd: int = 1, sess: int = 1, task: str = "RAN"):
    """Map pid -> csv path (or zip member) for GazeBase. Accepts extracted trees or zip files under root."""
    root = Path(root)
    out = {}
    for p in root.rglob("*.csv"):
        m = GB_RE.search(p.name)
        if m and int(m.group(1)) == rnd and int(m.group(3)) == sess and m.group(4) == task:
            out[int(m.group(1) + m.group(2))] = str(p)
    if not out:
        for zp in root.rglob("*.zip"):
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    m = GB_RE.search(name)
                    if m and int(m.group(1)) == rnd and int(m.group(3)) == sess and m.group(4) == task:
                        out[int(m.group(1) + m.group(2))] = f"{zp}::{name}"
    return dict(sorted(out.items()))


def load_gb(path_or_member: str):
    if "::" in path_or_member:
        zp, name = path_or_member.split("::", 1)
        with zipfile.ZipFile(zp) as z:
            df = pd.read_csv(io.BytesIO(z.read(name)))
        fname = name
    else:
        df = pd.read_csv(path_or_member)
        fname = path_or_member
    m = GB_RE.search(fname)
    # GazeBase validity flag: 0 = valid (any non-zero value is treated as invalid; verify on arrival)
    val = (df["val"].to_numpy() == 0) if "val" in df.columns else None
    return _rec(df, "x", "y", valid_extra=val, fs=1000.0,
                pid=int(m.group(1) + m.group(2)) if m else None,
                sess=int(m.group(3)) if m else None, task=m.group(4) if m else None)


def make_split(pids, n_dev, seed=SPLIT_SEED, salt=0):
    """Seeded development / confirmatory split over sorted participant ids."""
    pids = np.array(sorted(pids))
    rng = np.random.default_rng([seed, salt])
    dev = np.sort(rng.choice(pids, size=min(n_dev, len(pids)), replace=False))
    conf = np.setdiff1d(pids, dev)
    return dev.tolist(), conf.tolist()


def transitions(rec):
    """Indices and times of target steps (rows where (xT, yT) changes); first row excluded."""
    xT, yT = rec["xT"], rec["yT"]
    ch = np.where((np.diff(xT) != 0) | (np.diff(yT) != 0))[0] + 1
    ch = ch[np.isfinite(xT[ch]) & np.isfinite(yT[ch])]
    return ch, rec["t"][ch]
