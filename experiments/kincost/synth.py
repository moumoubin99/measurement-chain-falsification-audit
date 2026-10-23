"""Synthetic GB-like (1000 Hz) RAN recordings with known saccade kinematics.

Used only for smoke tests, profiling (R004 before GB arrives), the analytic existence example (R005) and
synthetic attribution (B6).  "True PV" language is valid only for these data.

Saccades: minimum-jerk position profile, duration D = 2.2 A + 21 ms, true PV = 1.875 A / D (deg/s).
Fixation: random-walk drift + 80 Hz tremor + white measurement noise.  Blinks: 150-300 ms gaps (val = 1).
"""
from __future__ import annotations

import math

import numpy as np


def min_jerk(tau):
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def sacc_duration_ms(A):
    return 2.2 * A + 21.0


def true_pv(A):
    return 1.875 * A / (sacc_duration_ms(A) / 1000.0)


def ran_targets(rng, n=100, xr=15.0, yr=9.0, min_jump=2.0):
    pts = [(0.0, 0.0)]
    while len(pts) < n:
        p = (rng.uniform(-xr, xr), rng.uniform(-yr, yr))
        if math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) >= min_jump:
            pts.append(p)
    return np.array(pts)


def gb_like(seed, n_targets=100, dwell_ms=1000.0, fs=1000.0, drift_sd=0.02, tremor_amp=0.003,
            white_sd=0.01, blink_rate_hz=0.15, gain=0.95, lat_mu=200.0, lat_sd=40.0, pid=None):
    rng = np.random.default_rng([2026, 1018, seed])
    tg = ran_targets(rng, n_targets)
    N = int(n_targets * dwell_ms * fs / 1000.0)
    t = np.arange(N) * 1000.0 / fs
    xT = np.repeat(tg[:, 0], int(dwell_ms * fs / 1000.0))[:N]
    yT = np.repeat(tg[:, 1], int(dwell_ms * fs / 1000.0))[:N]
    x = np.zeros(N)
    y = np.zeros(N)
    cur = np.array([0.0, 0.0])
    sacc = []
    step = int(dwell_ms * fs / 1000.0)
    for k in range(1, n_targets):
        i0 = k * step
        lat = max(100.0, rng.normal(lat_mu, lat_sd))
        on = i0 + int(lat * fs / 1000.0)
        tgt = tg[k]
        land = cur + gain * (tgt - cur) + rng.normal(0, 0.3, 2)
        A = float(np.hypot(*(land - cur)))
        D = sacc_duration_ms(A)
        nd = max(2, int(round(D * fs / 1000.0)))
        tau = np.arange(nd + 1) / nd
        prof = min_jerk(tau)
        seg_end = min(on + nd + 1, N)
        x[on:seg_end] = cur[0] + (land[0] - cur[0]) * prof[:seg_end - on]
        y[on:seg_end] = cur[1] + (land[1] - cur[1]) * prof[:seg_end - on]
        x[seg_end:] = land[0]      # held until the next saccade overwrites it
        y[seg_end:] = land[1]
        # truth uses the realised (sample-rounded) duration nd / fs, not the unrounded main-sequence D
        dur_s = nd / fs
        sacc.append(dict(trial=k, on_ms=t[on], amp=A, dur_ms=1000.0 * dur_s, pv_true=1.875 * A / dur_s,
                         theta=math.atan2(land[1] - cur[1], land[0] - cur[0])))
        cur = land
    # fixational components
    drift = np.cumsum(rng.normal(0, drift_sd / math.sqrt(fs), (2, N)), axis=1)
    # re-centre the walk at every target step so the trace stays near the landing point
    for k in range(n_targets):
        sl = slice(k * step, min((k + 1) * step, N))
        drift[:, sl] -= drift[:, sl.start:sl.start + 1]
    ph = rng.uniform(0, 2 * np.pi, 2)
    trem = tremor_amp * np.sin(2 * np.pi * 80.0 * t[None, :] / 1000.0 + ph[:, None])
    xo = x + drift[0] + trem[0] + rng.normal(0, white_sd, N)
    yo = y + drift[1] + trem[1] + rng.normal(0, white_sd, N)
    val = np.zeros(N, int)
    nb = rng.poisson(blink_rate_hz * N / fs)
    for _ in range(nb):
        s = rng.integers(0, N)
        L = int(rng.uniform(150, 300) * fs / 1000.0)
        val[s:s + L] = 1
    valid = val == 0
    return dict(t=t, x=np.where(valid, xo, np.nan), y=np.where(valid, yo, np.nan), valid=valid,
                xT=xT, yT=yT, fs_nominal=fs, pid=pid if pid is not None else 9000 + seed, sess=1, task="RAN",
                truth=sacc, latent_x=x, latent_y=y)


def saccade_only(A, fs=1000.0, pad_ms=300.0, theta=0.0):
    """Single noiseless saccade embedded in constant fixation (for the existence example)."""
    D = sacc_duration_ms(A)
    npad = int(pad_ms * fs / 1000.0)
    nd = int(round(D * fs / 1000.0))
    tau = np.arange(nd + 1) / nd
    p = np.concatenate([np.zeros(npad), A * min_jerk(tau), np.full(npad, A)])
    t = np.arange(len(p)) * 1000.0 / fs
    return t, p * math.cos(theta), p * math.sin(theta)
