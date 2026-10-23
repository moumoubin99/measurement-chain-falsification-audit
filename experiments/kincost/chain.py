"""Measurement-chain manifest and the 9-step normative chain (FINAL_PROPOSAL §Measurement chain).

The chain is exactly linear in its inputs once a realisation (phase, donor, missingness, noise draw) is
fixed: interpolation, per-run lfilter with steady-state initialisation (zi * x[run_start]) and noise
addition are all linear maps without offsets.  Hence

    out(s + sigma * eps) == out(s, 0) + sigma * out(0, eps)

which is what the quadratic hold cache and the exact noise-scaled re-detection rely on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.signal as ss

GRID_DT = 4.0            # ms, shared regularisation R4
E_SET_BASE = 100.0       # ms, settling exclusion before the settling assertion
BUTTER_ORDERS = (2, 4)
BUTTER_FC = (15, 20, 25, 30, 40, 50, 60, 80, 100)
MA_B1 = (2, 3, 4, 5)          # samples at 250 Hz (donor index)
MA_B2 = (4, 8, 12, 16, 20)    # samples at 1000 Hz
ALPHAS = (0.0, 0.5, 1.0, 1.5)  # 0 = white
FLAT_BELOW_HZ = 0.5
MU_BOUNDS = (math.log(1e-4), math.log(1.0))
S_BOUNDS = (0.0, 1.5)
SEED_BASE = 8_2026_1018


# --------------------------------------------------------------------------------------- manifest
@dataclass(frozen=True)
class Member:
    mid: int
    branch: str        # 'B1' | 'B2'
    ftype: str         # 'id' | 'butter' | 'ma'
    fpar: tuple        # (order, fc) | (k,) | ()
    noise_loc: str     # 'pre' | 'post'
    alpha: float

    @property
    def name(self):
        if self.ftype == "id":
            f = "id"
        elif self.ftype == "butter":
            f = "bw{}_{}".format(*self.fpar)
        else:
            f = "ma{}".format(*self.fpar)
        return f"{self.branch}_{f}_{self.noise_loc}_a{self.alpha:g}"


def _filters(branch):
    fl = [("butter", (o, fc)) for o in BUTTER_ORDERS for fc in BUTTER_FC]
    if branch == "B1":
        return [("id", ())] + fl + [("ma", (k,)) for k in MA_B1]
    return fl + [("ma", (k,)) for k in MA_B2]


def manifest():
    out = []
    for ft, fp in _filters("B1"):
        for loc in ("pre", "post"):
            for a in ALPHAS:
                out.append(Member(len(out), "B1", ft, fp, loc, a))
    for ft, fp in _filters("B2"):
        for a in ALPHAS:
            out.append(Member(len(out), "B2", ft, fp, "post", a))
    assert len(out) == 276, len(out)
    return out


def filter_ba(ftype, fpar, fs):
    if ftype == "id":
        return None
    if ftype == "butter":
        order, fc = fpar
        return ss.butter(order, fc / (fs / 2.0))
    k = fpar[0]
    return np.ones(k) / k, np.array([1.0])


def step_settling_ms(ftype, fpar, fs, tol=0.02, n=4000):
    ba = filter_ba(ftype, fpar, fs)
    if ba is None:
        return 0.0
    b, a = ba
    y = ss.lfilter(b, a, np.ones(n), zi=ss.lfilter_zi(b, a) * 0.0)[0]
    bad = np.nonzero(np.abs(y - 1.0) > tol)[0]
    return 0.0 if len(bad) == 0 else (bad[-1] + 1) * 1000.0 / fs


def settling_exclusion_ms(members=None, extra_ops=()):
    """E_set = max(100 ms, 2% step-settling time of the slowest implemented operator), rounded up to GRID_DT."""
    members = members or manifest()
    worst = 0.0
    for m in members:
        fs = 250.0 if m.branch == "B1" else 1000.0
        worst = max(worst, step_settling_ms(m.ftype, m.fpar, fs))
    for ft, fp, fs in extra_ops:
        worst = max(worst, step_settling_ms(ft, fp, fs))
    e = max(E_SET_BASE, math.ceil(worst / GRID_DT) * GRID_DT)
    return e, worst


# --------------------------------------------------------------------------------------- utilities
def runs_of(mask):
    """(start, stop) half-open index pairs of contiguous True segments."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return np.stack([np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]], axis=1)


def runs_split(valid, t, max_step):
    """Valid runs, additionally split wherever consecutive timestamps differ by more than max_step."""
    out = []
    for s, e in runs_of(valid):
        cut = np.nonzero(np.diff(t[s:e]) > max_step)[0] + 1 + s
        b = [s, *cut.tolist(), e]
        out.extend(zip(b[:-1], b[1:]))
    return np.array(out, int).reshape(-1, 2)


def filt_runs(x, runs, ba):
    if ba is None:
        return x
    b, a = ba
    zi = ss.lfilter_zi(b, a)
    out = np.zeros_like(x)
    for s, e in runs:
        out[s:e] = ss.lfilter(b, a, x[s:e], zi=zi * x[s])[0]
    return out


def shaped_noise(white_full, alpha, fs):
    """Shape a white N(0,1) vector (length 2N, first N kept) to a 1/f^alpha spectrum, flat below 0.5 Hz,
    scaled to unit *expected* RMS sample-to-sample difference at its native rate fs."""
    n2 = len(white_full)
    k = np.arange(n2)
    f = np.minimum(k, n2 - k) * fs / n2
    H = np.maximum(f, FLAT_BELOW_HZ) ** (-alpha / 2.0) if alpha > 0 else np.ones(n2)
    x = np.fft.ifft(np.fft.fft(white_full) * H).real
    var_diff = np.mean(H ** 2 * np.abs(1.0 - np.exp(2j * np.pi * k / n2)) ** 2)
    return x[: n2 // 2] / math.sqrt(var_diff)


def interp_op(t_src, valid_src, t_q, max_step):
    """Linear interpolation operator src -> query restricted to pairs of adjacent valid source samples
    (same valid run).  Returns (i0, w, ok): value = (1-w) src[i0] + w src[i0+1]."""
    n = len(t_src)
    i0 = np.searchsorted(t_src, t_q, side="right") - 1
    ok = (i0 >= 0) & (i0 < n - 1)
    i0c = np.clip(i0, 0, n - 2)
    dt = t_src[i0c + 1] - t_src[i0c]
    ok &= valid_src[i0c] & valid_src[i0c + 1] & (dt > 0) & (dt <= max_step)
    w = np.where(ok, (t_q - t_src[i0c]) / np.where(dt > 0, dt, 1.0), 0.0)
    # exact timestamp match on a valid sample that ends a run (or is the final sample): use the left bracket
    # [i-1, i] with w = 1, provided that bracket is itself an admissible adjacent valid pair
    i0r = np.clip(i0, 0, n - 1)
    exact = (~ok) & (i0 >= 1) & (t_q == t_src[i0r]) & valid_src[i0r]
    if exact.any():
        il = np.clip(i0r - 1, 0, n - 2)
        dtl = t_src[il + 1] - t_src[il]
        okl = exact & valid_src[il] & (dtl > 0) & (dtl <= max_step)
        i0c = np.where(okl, il, i0c)
        w = np.where(okl, 1.0, w)
        ok = ok | okl
    return i0c, w, ok


def apply_interp(x, op):
    i0, w, ok = op
    xs = np.where(np.isfinite(x), x, 0.0)
    return np.where(ok, (1.0 - w) * xs[i0] + w * xs[i0 + 1], 0.0)


def settle_mask(valid_g, e_set_ms, dt=GRID_DT):
    """True for grid samples within e_set_ms after a valid-run start."""
    ex = np.zeros(len(valid_g), bool)
    nset = int(round(e_set_ms / dt))
    for s, e in runs_of(valid_g):
        ex[s:min(e, s + nset)] = True
    return ex


RUN_MAX_STEP = 12.0   # ms: two consecutive valid samples further apart than this belong to different runs


def regularise(t, x, y, valid, dt=GRID_DT, origin=None, max_step=RUN_MAX_STEP):
    """R4 regularisation of native GBVR (identical rule to step 8 of the chain). Returns grid dict."""
    if origin is None:
        origin = math.ceil(t[0] / dt) * dt
    g = origin + dt * np.arange(int((t[-1] - origin) // dt) + 1)
    op = interp_op(t, valid, g, max_step)
    return dict(t=g, x=apply_interp(x, op), y=apply_interp(y, op), valid=op[2], dt=dt)


# --------------------------------------------------------------------------------------- donors / missingness
@dataclass
class DonorPool:
    """D-split GBVR RAN timing donors and the empirical valid-run / gap length distributions (donor samples)."""
    dts: list                 # list of arrays of inter-sample differences (ms)
    run_lengths: np.ndarray
    gap_lengths: np.ndarray

    @classmethod
    def from_recordings(cls, recs):
        dts, rl, gl = [], [], []
        for r in recs:
            d = np.diff(r["t"])
            dts.append(d[np.isfinite(d) & (d > 0)])
            v = r["valid"]
            rl.extend((e - s) for s, e in runs_of(v))
            gl.extend((e - s) for s, e in runs_of(~v))
        return cls(dts, np.asarray(rl, int), np.asarray(gl, int))


def renewal_mask(n, pool: DonorPool, rng):
    """Two-state renewal missingness on n donor samples; starts in the valid state.
    Run and gap lengths are i.i.d. draws from the D-split GBVR empirical distributions."""
    m = np.ones(n, bool)
    if len(pool.gap_lengths) == 0:
        return m
    i = 0
    state_valid = True
    while i < n:
        L = int(rng.choice(pool.run_lengths if state_valid else pool.gap_lengths))
        if not state_valid:
            m[i:i + L] = False
        i += max(L, 1)
        state_valid = not state_valid
    return m


# --------------------------------------------------------------------------------------- realisation geometry
@dataclass
class Geometry:
    """Member-independent geometry of one (participant, realisation): phase, donor, masks, operators."""
    t1000: np.ndarray
    valid1000: np.ndarray
    runs1000: np.ndarray
    t_d: np.ndarray
    op_d: tuple
    valid_d: np.ndarray
    runs_d: np.ndarray
    grid_t: np.ndarray
    op_g: tuple
    valid_g: np.ndarray
    usable_g: np.ndarray
    white_pre: np.ndarray = field(repr=False)    # (2, 2*N1000) white N(0,1), H and V
    white_post: np.ndarray = field(repr=False)   # (2, 2*Nd)
    phi: float = 0.0
    donor: int = -1
    noise_cache: dict = field(default_factory=dict, repr=False)


def rng_for(pid, r, stream):
    return np.random.default_rng([SEED_BASE, int(pid), int(r), int(stream)])


def make_geometry(rec, pool: DonorPool, pid, r, stream, e_set, use_missing=True):
    rng = rng_for(pid, r, stream)
    t, v = rec["t"], rec["valid"]
    phi = rng.uniform(0.0, 4.0)
    donor = int(rng.integers(len(pool.dts)))
    d = pool.dts[donor]
    span = t[-1] - t[0]
    nrep = int(math.ceil(span / max(d.sum(), 1e-9))) + 1
    td = t[0] + phi + np.concatenate([[0.0], np.cumsum(np.tile(d, nrep))])
    td = td[td <= t[-1]]
    op_d = interp_op(t, v, td, max_step=1.5)
    vd = op_d[2].copy()
    if use_missing:
        vd &= renewal_mask(len(td), pool, rng)
    origin = math.ceil(t[0] / GRID_DT) * GRID_DT
    g = origin + GRID_DT * np.arange(int((td[-1] - origin) // GRID_DT) + 1)
    op_g = interp_op(td, vd, g, max_step=RUN_MAX_STEP)
    vg = op_g[2]
    usable = vg & ~settle_mask(vg, e_set)
    white_pre = rng.standard_normal((2, 2 * len(t)))
    white_post = rng.standard_normal((2, 2 * len(td)))
    return Geometry(t, v, runs_split(v, t, RUN_MAX_STEP), td, op_d, vd, runs_split(vd, td, RUN_MAX_STEP),
                    g, op_g, vg, usable,
                    white_pre, white_post, phi, donor)


class Chain:
    """Applies one manifest member on a fixed Geometry. All methods are linear in their inputs."""

    def __init__(self, member: Member):
        self.m = member
        self.ba_pre = filter_ba(member.ftype, member.fpar, 1000.0) if member.branch == "B2" else None
        self.ba_post = filter_ba(member.ftype, member.fpar, 250.0) if member.branch == "B1" else None

    def noise(self, geo: Geometry, axis):
        """Unit-scale noise draw eps for this member's location and spectrum (H axis=0, V axis=1)."""
        key = (self.m.noise_loc, self.m.alpha, axis)
        if key not in geo.noise_cache:
            if self.m.noise_loc == "pre":
                geo.noise_cache[key] = shaped_noise(geo.white_pre[axis], self.m.alpha, 1000.0)
            else:
                geo.noise_cache[key] = shaped_noise(geo.white_post[axis], self.m.alpha, 250.0)
        return geo.noise_cache[key]

    def run(self, geo: Geometry, x1000, pre=None, post=None):
        """Steps 2-8 for one axis. x1000: native 1000 Hz signal (NaN where invalid), pre/post: added noise."""
        x = np.where(geo.valid1000, np.nan_to_num(x1000), 0.0)
        if self.ba_pre is not None:
            x = filt_runs(x, geo.runs1000, self.ba_pre)
        if pre is not None:
            x = x + np.where(geo.valid1000, pre, 0.0)
        xd = apply_interp(x, geo.op_d)
        xd = np.where(geo.valid_d, xd, 0.0)
        if self.ba_post is not None:
            xd = filt_runs(xd, geo.runs_d, self.ba_post)
        if post is not None:
            xd = xd + np.where(geo.valid_d, post, 0.0)
        return apply_interp(xd, geo.op_g)

    def signal_and_noise(self, geo, rec):
        """Grid outputs (s_H, s_V, e_H, e_V) with out(s + sigma e) = s + sigma e exactly."""
        sH = self.run(geo, rec["x"])
        sV = self.run(geo, rec["y"])
        outs = []
        for ax in (0, 1):
            eps = self.noise(geo, ax)
            z = np.zeros_like(rec["x"])
            if self.m.noise_loc == "pre":
                outs.append(self.run(geo, z, pre=eps))
            else:
                outs.append(self.run(geo, z, post=eps))
        return sH, sV, outs[0], outs[1]

    def degraded(self, geo, rec, sigH, sigV):
        """Direct chain on s + sigma eps (used for the paired identity check)."""
        epsH, epsV = self.noise(geo, 0), self.noise(geo, 1)
        if self.m.noise_loc == "pre":
            return (self.run(geo, rec["x"], pre=sigH * epsH), self.run(geo, rec["y"], pre=sigV * epsV))
        return (self.run(geo, rec["x"], post=sigH * epsH), self.run(geo, rec["y"], post=sigV * epsV))
