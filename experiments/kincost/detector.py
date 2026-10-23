"""Fixed adaptive-threshold saccade detector (Nystrom & Holmqvist 2010 style), identical on both datasets.

Input: positions (deg) on a regular grid with a boolean `usable` mask.  SG order-2 derivative,
window 5 samples at 250 Hz / 17 at 1000 Hz; speed undefined wherever the SG support touches an
unusable sample.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.signal as ss

PT0 = 100.0
MAX_ITER = 50
MIN_DUR_MS = 12.0
GUARD_MS = 20.0
LOCAL_MS = 40.0


def sg_window(fs):
    return 5 if fs <= 300 else 17


def sg_velocity(x, y, usable, fs):
    w = sg_window(fs)
    xs = np.where(usable, x, 0.0)
    ys = np.where(usable, y, 0.0)
    vx = ss.savgol_filter(xs, w, 2, deriv=1, mode="constant") * fs
    vy = ss.savgol_filter(ys, w, 2, deriv=1, mode="constant") * fs
    bad = np.convolve((~usable).astype(np.int32), np.ones(w, np.int32), mode="same") > 0
    half = w // 2
    bad[:half] = True
    bad[-half:] = True
    speed = np.hypot(vx, vy)
    speed[bad] = np.nan
    return speed, vx, vy, ~bad


def thresholds(speed):
    v = speed[np.isfinite(speed)]
    if len(v) < 50:
        return None
    pt = PT0
    for it in range(MAX_ITER):
        s = v[v < pt]
        if len(s) < 10:
            return None
        new = s.mean() + 6.0 * s.std()
        if abs(new - pt) < 1.0:
            pt = new
            s = v[v < pt]
            return dict(PT=pt, OT=s.mean() + 3.0 * s.std(), mu=s.mean(), sd=s.std(), iters=it + 1)
        pt = new
    return None   # recording excluded


def detect(x, y, usable, fs, thr=None):
    """Return (events, thr, info).  events: list of dicts with on, off (indices, inclusive),
    amp, dur_ms, pv, theta."""
    speed, vx, vy, defined = sg_velocity(x, y, usable, fs)
    if thr is None:
        thr = thresholds(speed)
    info = dict(n_peaks=0, lost_onset=0, lost_offset=0, lost_merge=0, lost_dur=0, lost_guard=0)
    if thr is None:
        return [], None, dict(info, excluded=True)
    PT, OT = thr["PT"], thr["OT"]
    n = len(speed)
    sp = np.where(defined, speed, np.nan)
    above = np.nan_to_num(sp, nan=-1.0) > PT
    # peak segments
    segs = []
    d = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        segs.append((s, e - 1))
    info["n_peaks"] = len(segs)
    nloc = max(1, int(round(LOCAL_MS * fs / 1000.0)))
    guard = int(round(GUARD_MS * fs / 1000.0))
    raw = []
    for s, e in segs:
        # onset: walk back to first sample below OT that is a local minimum
        i = s
        ok = True
        while True:
            i -= 1
            if i < 1 or not np.isfinite(sp[i]) or not np.isfinite(sp[i - 1]):
                ok = False
                break
            if sp[i] < OT and sp[i] <= sp[i - 1] and sp[i] <= sp[i + 1]:
                break
        if not ok:
            info["lost_onset"] += 1
            continue
        on = i
        loc = sp[max(0, on - nloc):on]
        loc = loc[np.isfinite(loc)]
        offt = 0.7 * OT + 0.3 * (loc.mean() + 3.0 * loc.std()) if len(loc) >= 5 else OT
        j = e
        ok = True
        while True:
            j += 1
            if j >= n - 1 or not np.isfinite(sp[j]) or not np.isfinite(sp[j + 1]):
                ok = False
                break
            if sp[j] < offt and sp[j] <= sp[j - 1] and sp[j] <= sp[j + 1]:
                break
        if not ok:
            info["lost_offset"] += 1
            continue
        raw.append([on, j])
    # merge overlapping
    merged = []
    for ev in raw:
        if merged and ev[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], ev[1])
            info["lost_merge"] += 1
        else:
            merged.append(list(ev))
    events = []
    for on, off in merged:
        dur = (off - on) * 1000.0 / fs
        if dur < MIN_DUR_MS:
            info["lost_dur"] += 1
            continue
        a, b = on - guard, off + guard
        if a < 0 or b >= n or not defined[a:b + 1].all():
            info["lost_guard"] += 1
            continue
        dx, dy = x[off] - x[on], y[off] - y[on]
        events.append(dict(on=int(on), off=int(off), amp=float(math.hypot(dx, dy)), dur_ms=float(dur),
                           pv=float(np.nanmax(sp[on:off + 1])), theta=float(math.atan2(dy, dx))))
    return events, thr, dict(info, excluded=False)


def primary_saccades(events, t, x, y, trans_t, targets, lat=(80.0, 500.0), amp=(5.0, 15.0),
                     land_tol=3.0, next_gap=400.0):
    """RAN primary-saccade selection.  trans_t: transition times (ms); targets: (xT, yT) after each.
    Returns list of event dicts with 'trial' index added."""
    if not events:
        return []
    on_t = np.array([t[e["on"]] for e in events])
    out = []
    for k, tt in enumerate(trans_t):
        nxt = trans_t[k + 1] if k + 1 < len(trans_t) else np.inf
        idx = np.nonzero((on_t >= tt + lat[0]) & (on_t <= tt + lat[1]))[0]
        if len(idx) == 0:
            continue
        e = events[idx[0]]
        if t[e["on"]] >= nxt:
            continue
        if not (amp[0] <= e["amp"] <= amp[1]):
            continue
        tx, ty = targets[k]
        if math.hypot(x[e["off"]] - tx, y[e["off"]] - ty) > land_tol:
            continue
        if nxt - t[e["off"]] <= next_gap:
            continue
        out.append(dict(e, trial=k, latency=float(t[e["on"]] - tt)))
    return out
