"""Hold windows, quadratic-coefficient hold cache, cohort coordinates, noise fit and profiled admissibility."""
from __future__ import annotations

import math

import numpy as np
from . import chain as C
from . import detector as DET
from . import fastfit as FF

WIN_MS = 512.0
WIN_MS_FALLBACK = 256.0
K_WIN = 40
MIN_WIN = 10
BANDS = ((4, 10), (10, 30), (30, 60), (60, 125))
BANDS_FALLBACK = ((8, 30), (30, 60), (60, 125))
POST_EVENT_MS = 100.0
PRE_NEXT_MS = 20.0
MIN_EVENT_AMP = 0.5
FIT_STARTS = 5
GRID_FS = 1000.0 / C.GRID_DT


# --------------------------------------------------------------------------------------- windows
def hold_windows(t, x, y, usable, fs, trans_t, rng, win_ms=WIN_MS, k=K_WIN):
    """Window start *times* (ms) on native data, one candidate per trial, then K seeded draws.
    Window = first win_ms span starting in [last event>=0.5 deg offset + 100 ms, next transition - 20 ms],
    sliding in 4 ms steps, with every native sample usable."""
    events, thr, info = DET.detect(x, y, usable, fs)
    on_t = np.array([t[e["on"]] for e in events]) if events else np.zeros(0)
    off_t = np.array([t[e["off"]] for e in events]) if events else np.zeros(0)
    amp = np.array([e["amp"] for e in events]) if events else np.zeros(0)
    bad_cum = np.concatenate([[0], np.cumsum(~usable)])
    cands = []
    for i, tt in enumerate(trans_t):
        nxt = trans_t[i + 1] if i + 1 < len(trans_t) else t[-1]
        sel = np.nonzero((on_t >= tt) & (on_t < nxt) & (amp >= MIN_EVENT_AMP))[0]
        if len(sel) == 0:
            continue
        lo = off_t[sel[-1]] + POST_EVENT_MS
        hi = nxt - PRE_NEXT_MS
        s = lo
        while s + win_ms <= hi:
            a = np.searchsorted(t, s, side="left")
            b = np.searchsorted(t, s + win_ms, side="left")
            if b - a >= int(0.9 * win_ms * fs / 1000.0) and bad_cum[b] - bad_cum[a] == 0:
                cands.append(s)
                break
            s += C.GRID_DT
    cands = np.array(cands)
    if len(cands) > k:
        cands = np.sort(rng.choice(cands, size=k, replace=False))
    return cands, info


def grid_window_index(grid_t, starts_ms, win_ms=WIN_MS):
    """Grid index start for each window: first grid point >= start time; length n = win_ms / 4."""
    n = int(round(win_ms / C.GRID_DT))
    i0 = np.searchsorted(grid_t, starts_ms, side="left")
    ok = i0 + n <= len(grid_t)
    return i0[ok], n


# --------------------------------------------------------------------------------------- quadratic quantities
class QuadForms:
    """Bilinear forms B(u, w) for the 9 hold quantities over a set of grid windows (same length n)."""

    def __init__(self, n, bands=BANDS, fs=GRID_FS):
        self.n = n
        self.hann = np.hanning(n)
        self.U = (self.hann ** 2).sum()
        f = np.fft.rfftfreq(n, 1.0 / fs)
        self.df = fs / n
        w = np.full(len(f), 2.0)
        w[0] = 1.0
        if n % 2 == 0:
            w[-1] = 1.0
        self.side_w = w / (fs * self.U)
        self.bandsel = []
        for j, (lo, hi) in enumerate(bands):
            last = j == len(bands) - 1
            self.bandsel.append((f >= lo) & ((f <= hi) if last else (f < hi)))
        self.nq = 2 + len(bands) + 3

    def per_window(self, uH, wH, uV, wV, starts):
        n = self.n
        idx = starts[:, None] + np.arange(n)[None, :]
        UH, WH, UV, WV = uH[idx], wH[idx], uV[idx], wV[idx]
        out = np.empty((len(starts), self.nq))
        dUH, dWH = np.diff(UH, axis=1), np.diff(WH, axis=1)
        out[:, 0] = (dUH * dWH).mean(1)
        out[:, 1] = (np.diff(UV, axis=1) * np.diff(WV, axis=1)).mean(1)
        FU = np.fft.rfft(self.hann * (UH - UH.mean(1, keepdims=True)), axis=1)
        FW = np.fft.rfft(self.hann * (WH - WH.mean(1, keepdims=True)), axis=1)
        cross = (FU * np.conj(FW)).real * self.side_w * self.df
        for j, sel in enumerate(self.bandsel):
            out[:, 2 + j] = cross[:, sel].sum(1)
        vu = dUH - dUH.mean(1, keepdims=True)
        vw = dWH - dWH.mean(1, keepdims=True)
        m = vu.shape[1]
        for lag in range(3):
            if lag == 0:
                s = (vu * vw).sum(1)
            else:
                s = 0.5 * ((vu[:, :-lag] * vw[:, lag:]).sum(1) + (vw[:, :-lag] * vu[:, lag:]).sum(1))
            out[:, 2 + len(self.bandsel) + lag] = s
        out[:, 2 + len(self.bandsel):] /= m      # pooled: identical length windows -> mean of per-window means
        return out

    def window_mean(self, uH, wH, uV, wV, starts):
        if len(starts) == 0:
            return np.full(self.nq, np.nan)
        return self.per_window(uH, wH, uV, wV, starts).mean(0)

    def abc(self, sH, sV, eH, eV, starts):
        a = self.window_mean(sH, sH, sV, sV, starts)
        b = self.window_mean(sH, eH, sV, eV, starts)
        c = self.window_mean(eH, eH, eV, eV, starts)
        return a, b, c


def window_ok(usable_g, starts, n):
    if len(starts) == 0:
        return starts
    cs = np.concatenate([[0], np.cumsum(~usable_g)])
    ok = cs[starts + n] - cs[starts] == 0
    return starts[ok]


def gap_stats(valid_g, dt=C.GRID_DT):
    """Gap fraction and (sum log gap length ms, count) on the R4 grid."""
    gaps = C.runs_of(~valid_g)
    L = (gaps[:, 1] - gaps[:, 0]) * dt if len(gaps) else np.zeros(0)
    return float((~valid_g).mean()), float(np.log(L).sum()) if len(L) else 0.0, float(len(L))


# --------------------------------------------------------------------------------------- coordinates
def sigma_of(eta, z):
    muH, sH, muV, sV = eta
    return np.exp(muH + sH * z), np.exp(muV + sV * z)


def _iqr(v, axis):
    q = np.nanpercentile(v, [25, 75], axis=axis)
    return q[1] - q[0]


def participant_coords(Q, nb):
    """Q: (..., 9 or 8) quadratic quantities -> per-participant coordinate quantities (..., 2+2... )
    returns (logrms_H, logrms_V, logband[nb], ac1, ac2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lH = 0.5 * np.log(Q[..., 0])
        lV = 0.5 * np.log(Q[..., 1])
        lb = np.log(Q[..., 2:2 + nb])
        q0 = Q[..., 2 + nb]
        ac1 = Q[..., 3 + nb] / q0
        ac2 = Q[..., 4 + nb] / q0
    return lH, lV, lb, ac1, ac2


def coords_from_Q(Q, gap, nb, idx=None):
    """Cohort coordinates (n_coord,) from Q (P, R, nq) and gap (P, R, 3); cohort stats within r, then mean over r.
    idx: participant index vector (bootstrap).  NaN rows (no surviving windows) are ignored."""
    if idx is not None:
        Q = Q[idx]
        gap = gap[idx]
    lH, lV, lb, ac1, ac2 = participant_coords(Q, nb)
    co = [np.nanmedian(lH, 0), np.nanmedian(lV, 0), _iqr(lH, 0), _iqr(lV, 0)]
    co += [np.nanmedian(lb[..., j], 0) for j in range(nb)]
    co += [np.nanmedian(ac1, 0), np.nanmedian(ac2, 0)]
    co += [np.nanmedian(gap[..., 0], 0)]
    co += [gap[..., 1].sum(0) / np.maximum(gap[..., 2].sum(0), 1e-12)]
    return np.array([np.mean(c) for c in co])


def eval_Q(cache, eta, idx=None):
    a, b, c, z = cache["a"], cache["b"], cache["c"], cache["z"]
    if idx is not None:
        a, b, c, z = a[idx], b[idx], c[idx], z[idx]
    sH, sV = sigma_of(eta, z)
    sig = np.repeat(sH[:, None], a.shape[2], axis=1)
    sig[:, 1] = sV
    sig = sig[:, None, :]
    return a + 2.0 * sig * b + sig ** 2 * c


def coords_gb(cache, eta, idx=None):
    Q = eval_Q(cache, eta, idx)
    gap = cache["gap"] if idx is None else cache["gap"][idx]
    return coords_from_Q(Q, gap, cache["nb"])


def coords_native(nat, idx=None):
    """Native GBVR: nat['Q'] (P, 1, nq) observed quantities, nat['gap'] (P, 1, 3)."""
    Q = nat["Q"] if idx is None else nat["Q"][idx]
    gap = nat["gap"] if idx is None else nat["gap"][idx]
    return coords_from_Q(Q, gap, nat["nb"])


# --------------------------------------------------------------------------------------- noise fit
def fit_axis(a, b, c, z, tmed, tiqr, sd_med, sd_iqr, seed):
    """Fit (mu, s) for one axis to (median, IQR) of log RMS-S2S. a,b,c: (P, R).
    5 seeded starts (start 0 = moment guess), box-constrained LM (fastfit), deterministic tie-break."""
    (lo0, hi0), (lo1, hi1) = C.MU_BOUNDS, C.S_BOUNDS
    a = np.ascontiguousarray(a, float)
    b = np.ascontiguousarray(b, float)
    c = np.ascontiguousarray(c, float)
    z = np.ascontiguousarray(z, float)
    rng = np.random.default_rng([C.SEED_BASE, 77, seed])
    starts = np.column_stack([rng.uniform(lo0, hi0, FIT_STARTS), rng.uniform(lo1, hi1, FIT_STARTS)])
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = 0.5 * np.nanmedian(np.log(c))
    starts[0] = [np.clip(tmed - lc, lo0, hi0), 0.3]
    sols = []
    for k, (x0, x1) in enumerate(starts):
        y0, y1, f = FF.lm_box(a, b, c, z, tmed, tiqr, sd_med, sd_iqr, float(x0), float(x1), lo0, hi0, lo1, hi1)
        sols.append((float(f), float(y1), k, np.array([y0, y1])))
    best = min(sols, key=lambda q: (round(q[0], 10), q[1], q[2]))
    x = best[3]
    eps = 1e-6
    flags = dict(boundary=bool(abs(x[0] - lo0) < eps or abs(x[0] - hi0) < eps or abs(x[1] - lo1) < eps
                               or abs(x[1] - hi1) < eps))
    flags["distinct_optima"] = bool(any(
        f <= best[0] * 1.01 + 1e-12 and np.max(np.abs(xx - x) / np.maximum(np.abs(x), 1e-3)) > 0.10
        for f, _, _, xx in sols))
    h = 1e-4
    J = np.zeros((2, 2))
    for j in range(2):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (np.array(FF.axis_stats(a, b, c, z, *xp)) - np.array(FF.axis_stats(a, b, c, z, *xm))) / (2 * h)
    cond = float(np.linalg.cond(J)) if np.isfinite(J).all() else np.inf
    flags["ill_conditioned"] = bool(not np.isfinite(cond) or cond > 1e6)
    return x, best[0], flags


def fit_scales(sd, target):
    """Fitting weights for the (median, IQR) residuals of coordinates 1-4: the supplied SDs floored at
    max(1e-3 |target|, 1e-9) so that a zero / non-finite SD cannot produce infinite residuals.  Applied at the
    fitting boundary (fit_eta), so every caller is covered; the admissibility statistic itself is unchanged."""
    sd = np.asarray(sd, float)[:4]
    t = np.abs(np.asarray(target, float)[:4])
    floor = np.maximum(1e-3 * np.where(np.isfinite(t), t, 0.0), 1e-9)
    return np.where(np.isfinite(sd), np.maximum(sd, floor), floor)


def fit_eta(cache, target, sd, idx=None, seed=0):
    sd = fit_scales(sd, target)
    a, b, c, z = cache["a"], cache["b"], cache["c"], cache["z"]
    if idx is not None:
        a, b, c, z = a[idx], b[idx], c[idx], z[idx]
    xH, fH, flH = fit_axis(a[..., 0], b[..., 0], c[..., 0], z, target[0], target[2], sd[0], sd[2], seed)
    xV, fV, flV = fit_axis(a[..., 1], b[..., 1], c[..., 1], z, target[1], target[3], sd[1], sd[3], seed + 1)
    eta = np.array([xH[0], xH[1], xV[0], xV[1]])
    flags = {k: bool(flH[k] or flV[k]) for k in flH}
    return eta, fH + fV, flags


# --------------------------------------------------------------------------------------- admissibility
def bootstrap_indices(n, B, seed, salt):
    rng = np.random.default_rng([C.SEED_BASE, 555, seed, salt])
    return rng.integers(0, n, size=(B, n))


def native_sd(nat, idx_vr):
    return np.array([coords_native(nat, i) for i in idx_vr]).std(0, ddof=1)


def admissibility(cache, nat, idx_gb, idx_vr, sd_fit, nat_boot=None, seed=0):
    """Profiled admissibility for one member.  idx_gb (B, P_gb), idx_vr (B, P_vr) shared across members (drawn
    from the eligible participants, see calibrate.boot_setup).  sd_fit: GBVR-bootstrap SDs of coordinates 1-4
    (floored inside fit_eta).  Returns dict.  An undefined test (non-finite native target, point discrepancy or
    any bootstrap discrepancy) is reported as estimable=False -- it is neither a statistical nor a deterministic
    rejection, and the member is not placed in A."""
    T = coords_native(nat)
    Bn = len(idx_gb)

    def _nonest(reason, eta=None, f=np.nan, flags=None, D_hat=None, s=None, n_bad=0):
        return dict(eta=(np.full(4, np.nan) if eta is None else eta).tolist(), fit_obj=f,
                    flags=flags or dict(boundary=False, nonconv=False, multimodal=False),
                    D_hat=(np.full(len(T), np.nan) if D_hat is None else D_hat).tolist(),
                    s=(np.full(len(T), np.nan) if s is None else s).tolist(), Lambda=np.nan, c=np.nan,
                    admissible=False, det_reject=False, dropped=[], estimable=False, reason=reason,
                    n_boot_nonfinite=n_bad)
    if not np.isfinite(T).all():
        return _nonest("nonfinite_native_target")
    eta, f, flags = fit_eta(cache, T, sd_fit, seed=seed)
    D_hat = coords_gb(cache, eta) - T
    if not np.isfinite(D_hat).all():
        return _nonest("nonfinite_point_discrepancy", eta, f, flags, D_hat)
    Db = np.empty((Bn, len(T)))
    for bb in range(Bn):
        Tb = nat_boot[bb] if nat_boot is not None else coords_native(nat, idx_vr[bb])
        if not np.isfinite(Tb).all():
            Db[bb] = np.nan
            continue
        eb, _, _ = fit_eta(cache, Tb, sd_fit, idx=idx_gb[bb], seed=seed)
        Db[bb] = coords_gb(cache, eb, idx_gb[bb]) - Tb
    n_bad = int((~np.isfinite(Db).all(1)).sum())
    if n_bad:
        return _nonest("nonfinite_bootstrap", eta, f, flags, D_hat, n_bad=n_bad)
    s = Db.std(0, ddof=1)
    scale = np.maximum.reduce([np.abs(T), nat.get("pp_sd", np.zeros_like(T)), np.full_like(T, 1e-12)])
    keep = np.ones(len(T), bool)
    det_reject = False
    dropped = []
    for j in range(len(T)):
        if s[j] < 1e-6 * scale[j]:
            if abs(D_hat[j]) < 1e-6 * scale[j]:
                keep[j] = False
                dropped.append(j)
            else:
                det_reject = True
    Lam = float(np.max(np.abs(D_hat[keep]) / s[keep])) if keep.any() else 0.0
    Lstar = np.max(np.abs(Db[:, keep] - D_hat[keep]) / s[keep], axis=1) if keep.any() else np.zeros(Bn)
    c = float(np.percentile(Lstar, 95))
    adm = (not det_reject) and Lam <= c
    return dict(eta=eta.tolist(), fit_obj=f, flags=flags, D_hat=D_hat.tolist(), s=s.tolist(),
                Lambda=Lam, c=c, admissible=bool(adm), det_reject=det_reject, dropped=dropped, estimable=True,
                reason="", n_boot_nonfinite=0)


# --------------------------------------------------------------------------------------- native GBVR quantities
def native_quantities(grid, usable, starts, n, bands=BANDS):
    """Observed (noise-free) per-participant quantities on native GBVR grid data."""
    qf = QuadForms(n, bands)
    st = window_ok(usable, starts, n)
    x = np.where(grid["valid"], grid["x"], 0.0)
    y = np.where(grid["valid"], grid["y"], 0.0)
    Q = qf.window_mean(x, x, y, y, st)
    return Q, len(st)
