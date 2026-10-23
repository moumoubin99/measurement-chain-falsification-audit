"""Per-participant glue: native GBVR processing, GB hold caches, degraded re-detection -> endpoint caches."""
from __future__ import annotations

import math

import numpy as np

from . import chain as C
from . import detector as DET
from . import endpoint as EP
from . import holds as H
from . import io_data as IO


def z_of(pid):
    return float(np.random.default_rng([C.SEED_BASE, 4242, int(pid)]).standard_normal())


def trans_info(rec):
    idx, tt = IO.transitions(rec)
    targets = np.column_stack([rec["xT"][idx], rec["yT"][idx]])
    return tt, targets


# --------------------------------------------------------------------------------------- native GBVR
def grid_native(t, x, y, valid_g, usable, tt, targets, pid, win_ms=H.WIN_MS, bands=H.BANDS, want_events=True):
    """Native-style processing of grid data (native GBVR, or a benchmark pseudo-GBVR realisation)."""
    x = np.where(usable, x, 0.0)
    y = np.where(usable, y, 0.0)
    rng = np.random.default_rng([C.SEED_BASE, 31, int(pid)])
    starts_ms, winfo = H.hold_windows(t, x, y, usable, H.GRID_FS, tt, rng, win_ms=win_ms)
    st, n = H.grid_window_index(t, starts_ms, win_ms)
    qf = H.QuadForms(n, bands)
    st = H.window_ok(usable, st, n)
    Q = qf.window_mean(x, x, y, y, st)
    if len(st) < H.MIN_WIN:            # hold-ineligible participant (fewer than MIN_WIN usable windows)
        Q = np.full(np.shape(Q), np.nan)
    gap = H.gap_stats(valid_g)
    out = dict(pid=pid, Q=Q, n_win=len(st), hold_eligible=len(st) >= H.MIN_WIN, gap=gap, usable=usable)
    if want_events:
        ev, thr, info = DET.detect(x, y, usable, H.GRID_FS)
        prim = DET.primary_saccades(ev, t, x, y, tt, targets)
        out.update(events=ev, primary=prim, det_info=info, thr=thr)
    return out


def gbvr_native(rec, e_set, win_ms=H.WIN_MS, bands=H.BANDS, want_events=True):
    g = C.regularise(rec["t"], rec["x"], rec["y"], rec["valid"])
    usable = g["valid"] & ~C.settle_mask(g["valid"], e_set)
    tt, targets = trans_info(rec)
    out = grid_native(g["t"], g["x"], g["y"], g["valid"], usable, tt, targets, rec["pid"], win_ms, bands,
                      want_events)
    out["grid"] = g
    return out


# --------------------------------------------------------------------------------------- GB participant
def native_gb_usable(rec, e_set):
    """Native GB usable mask: valid and outside the E_set settling exclusion after each valid-run start (1000 Hz)."""
    dt = 1000.0 / rec.get("fs_nominal", 1000.0)
    return rec["valid"] & ~C.settle_mask(rec["valid"], e_set, dt=dt)


def native_gb_windows(rec, tt, e_set, win_ms=H.WIN_MS):
    """Fixed hold-window start times on the native GB recording (settling-excluded usable mask)."""
    usable = native_gb_usable(rec, e_set)
    x0 = np.where(usable, rec["x"], 0.0)
    y0 = np.where(usable, rec["y"], 0.0)
    rng = np.random.default_rng([C.SEED_BASE, 32, int(rec["pid"])])
    return H.hold_windows(rec["t"], x0, y0, usable, 1000.0, tt, rng, win_ms=win_ms)



class GBParticipant:
    """Holds the native GB recording, its fixed hold windows, z_p and member-independent geometries."""

    def __init__(self, rec, pool, e_set, R, stream, win_ms=H.WIN_MS, bands=H.BANDS):
        self.rec = rec
        self.pid = int(rec["pid"])
        self.z = z_of(self.pid)
        self.e_set = e_set
        self.win_ms = win_ms
        self.bands = bands
        self.tt, self.targets = trans_info(rec)
        self.win_starts_ms, self.win_info = native_gb_windows(rec, self.tt, e_set, win_ms)
        self.hold_eligible = len(self.win_starts_ms) >= H.MIN_WIN
        self.geos = [C.make_geometry(rec, pool, self.pid, r, stream, e_set) for r in range(R)]
        self.win_grid = []
        for g in self.geos:
            st, n = H.grid_window_index(g.grid_t, self.win_starts_ms, win_ms)
            self.win_grid.append((st, n))
        self.gaps = np.array([H.gap_stats(g.valid_g) for g in self.geos])

    def hold_cache(self, member):
        ch = C.Chain(member)
        R = len(self.geos)
        nq = 2 + len(self.bands) + 3
        a = np.full((R, nq), np.nan)
        b = a.copy()
        c = a.copy()
        nwin = np.zeros(R, int)
        for r, g in enumerate(self.geos):
            st, n = self.win_grid[r]
            st = H.window_ok(g.usable_g, st, n)
            nwin[r] = len(st)
            # policy: a participant with < MIN_WIN native windows is hold-ineligible (all NaN); a transformed
            # realisation that keeps < MIN_WIN of the fixed windows is excluded for that participant (NaN).
            # Geometry is member-independent, so the eligible set is the same for every member.
            if not self.hold_eligible or len(st) < H.MIN_WIN:
                continue
            qf = H.QuadForms(n, self.bands)
            sH, sV, eH, eV = ch.signal_and_noise(g, self.rec)
            a[r], b[r], c[r] = qf.abc(sH, sV, eH, eV, st)
        return a, b, c, nwin

    def degraded_events(self, member, sigH, sigV, r_list=None, geos=None):
        """Exact noise-scaled re-detection on s + sigma*eps for each realisation; returns list of primary saccades."""
        ch = C.Chain(member)
        out = []
        for r, g in enumerate(geos if geos is not None else self.geos):
            if r_list is not None and r not in r_list:
                continue
            xg, yg = ch.degraded(g, self.rec, sigH, sigV)
            x = np.where(g.usable_g, xg, 0.0)
            y = np.where(g.usable_g, yg, 0.0)
            ev, thr, info = DET.detect(x, y, g.usable_g, H.GRID_FS)
            out.append(DET.primary_saccades(ev, g.grid_t, x, y, self.tt, self.targets))
        return out

    def sigma(self, eta):
        sH, sV = H.sigma_of(eta, np.array([self.z]))
        return float(sH[0]), float(sV[0])


def build_cache(parts, member):
    """Stack hold caches over participants -> dict for holds.coords_gb / fit_eta."""
    A, B, Cc, NW = [], [], [], []
    for p in parts:
        a, b, c, nw = p.hold_cache(member)
        A.append(a)
        B.append(b)
        Cc.append(c)
        NW.append(nw)
    return dict(a=np.array(A), b=np.array(B), c=np.array(Cc), z=np.array([p.z for p in parts]),
                gap=np.array([p.gaps for p in parts]), nb=len(parts[0].bands), nwin=np.array(NW))


def native_holds(nats, bands=H.BANDS):
    Q = np.array([n["Q"] for n in nats])[:, None, :]
    gap = np.array([n["gap"] for n in nats])[:, None, :]
    lH, lV, lb, ac1, ac2 = H.participant_coords(Q[:, 0], len(bands))
    pp = np.column_stack([lH, lV, lH, lV, lb, ac1, ac2, gap[:, 0, 0], gap[:, 0, 1] / np.maximum(gap[:, 0, 2], 1)])   # scale refs for zero-SE rule
    return dict(Q=Q, gap=gap, nb=len(bands), pp_sd=np.nanstd(pp, 0))
