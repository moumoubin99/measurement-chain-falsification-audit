"""D-split calibration: GB hold caches for all 276 members, native GBVR holds, window-length rule,
profiled admissibility (B = 2000) and the operational non-rejection set A."""
from __future__ import annotations

import time

import numpy as np

from . import chain as C
from . import holds as H
from . import pipeline as PL

_G = {}


def gb_part_caches(rec, pool, e_set, R, stream, win_ms, bands, members):
    part = PL.GBParticipant(rec, pool, e_set, R, stream, win_ms=win_ms, bands=bands)
    nq = 2 + len(bands) + 3
    M = len(members)
    a = np.full((M, R, nq), np.nan)
    b = a.copy()
    c = a.copy()
    nwin = np.zeros((M, R), int)
    for i, m in enumerate(members):
        a[i], b[i], c[i], nwin[i] = part.hold_cache(m)
    return dict(pid=part.pid, z=part.z, a=a, b=b, c=c, nwin=nwin[0], gap=part.gaps,
                n_win_native=len(part.win_starts_ms))


def stack_gb(parts_out, member_index, nb):
    return dict(a=np.array([p["a"][member_index] for p in parts_out]),
                b=np.array([p["b"][member_index] for p in parts_out]),
                c=np.array([p["c"][member_index] for p in parts_out]),
                z=np.array([p["z"] for p in parts_out]),
                gap=np.array([p["gap"] for p in parts_out]), nb=nb)


def window_rule(n_gb, n_vr, min_win=H.MIN_WIN, frac=0.8):
    return bool(np.mean(np.asarray(n_gb) >= min_win) >= frac and np.mean(np.asarray(n_vr) >= min_win) >= frac)


def _init(nat, idx_gb, idx_vr, sd_fit, nat_boot):
    _G.update(nat=nat, idx_gb=idx_gb, idx_vr=idx_vr, sd_fit=sd_fit, nat_boot=nat_boot)


def _adm_worker(args):
    mid, cache = args
    t0, c0 = time.perf_counter(), time.process_time()
    res = H.admissibility(cache, _G["nat"], _G["idx_gb"], _G["idx_vr"], _G["sd_fit"], nat_boot=_G["nat_boot"],
                          seed=mid)
    res["mid"] = mid
    res["t_s"] = time.perf_counter() - t0
    res["cpu_s"] = time.process_time() - c0
    return res


def native_boot(nat, idx_vr):
    return np.array([H.coords_native(nat, i) for i in idx_vr])


def boot_setup(caches, nat, B, seed):
    """Shared admissibility bootstrap set-up (run_admissibility, R004).  Resampling population = participants
    eligible after the member-independent exclusions (GB: hold-eligible with >= 1 realisation of >= MIN_WIN
    windows, read from the first member's cache; GBVR: native Q defined).  Returns idx_gb, idx_vr, raw SDs of
    coordinates 1-4 (floored only inside holds.fit_eta), native bootstrap coordinates, metadata."""
    a0 = caches[0][1]["a"]
    elig_gb = np.flatnonzero(np.isfinite(a0.reshape(a0.shape[0], -1)).any(1))
    Q = nat["Q"]
    elig_vr = np.flatnonzero(np.isfinite(Q.reshape(Q.shape[0], -1)).all(1))
    idx_gb = elig_gb[H.bootstrap_indices(len(elig_gb), B, seed, 1)]
    idx_vr = elig_vr[H.bootstrap_indices(len(elig_vr), B, seed, 2)]
    nb_ = native_boot(nat, idx_vr)
    with np.errstate(invalid="ignore"):
        sd_raw = nb_.std(0, ddof=1)
    meta = dict(n_gb=int(a0.shape[0]), n_gb_eligible=int(len(elig_gb)), n_vr=int(Q.shape[0]),
                n_vr_eligible=int(len(elig_vr)), sd_fit_raw=sd_raw[:4].tolist(),
                sd_fit=H.fit_scales(sd_raw[:4], H.coords_native(nat)).tolist(),
                native_coords=H.coords_native(nat).tolist(), native_boot_sd=sd_raw.tolist(),
                native_boot_nonfinite=int((~np.isfinite(nb_).all(1)).sum()))
    return idx_gb, idx_vr, sd_raw[:4], nb_, meta


def run_admissibility(caches, nat, B, seed, workers, executor_cls=None):
    """caches: list of (mid, cache).  Shared bootstrap index vectors across members."""
    idx_gb, idx_vr, sd_raw, nb_, meta = boot_setup(caches, nat, B, seed)
    if workers <= 1 or executor_cls is None:
        _init(nat, idx_gb, idx_vr, sd_raw, nb_)
        out = [_adm_worker(x) for x in caches]
    else:
        with executor_cls(workers, initializer=_init, initargs=(nat, idx_gb, idx_vr, sd_raw, nb_)) as ex:
            out = list(ex.map(_adm_worker, caches, chunksize=1))
    return out, meta
