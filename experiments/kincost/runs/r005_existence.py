"""R005 - Block 5 analytic existence example.

Constant latent fixation with minimum-jerk main-sequence saccades (5/10/15 deg, rightward and leftward),
recorded through B1 members with unit DC gain (identity, BW2-60, BW4-30, BW4-15, MA5) and identical shaped post
noise.  With a constant latent hold and post-filter noise, the hold quadratic quantities (and therefore
coordinates 1-10) agree across members up to post-saccadic filter transients (same draws); PV does not.
Reports: per-member hold coordinates (cohort of P synthetic participants), max relative coordinate spread,
noise-free filtered PV (deterministic) and detected PV at sigma, per amplitude.
"""
from __future__ import annotations

import argparse

import numpy as np

import common as CM
from kincost import chain as C
from kincost import detector as DET
from kincost import holds as H
from kincost import pipeline as PL
from kincost import synth as S

MEMBERS = ["B1_id_post_a1", "B1_bw2_60_post_a1", "B1_bw4_30_post_a1", "B1_bw4_15_post_a1", "B1_ma5_post_a1"]
AMPS = (5.0, 10.0, 15.0)
ETA = np.array([np.log(0.02), 0.3, np.log(0.02), 0.3])   # GBVR-like S2S noise scale (deg)


def constant_fixation_rec(seed, reps=8, dwell_ms=1500.0, lat_ms=200.0, fs=1000.0):
    """Targets alternate 0 -> +A -> 0 -> -A ... for A in AMPS; latent gaze is exactly constant between saccades."""
    rng = np.random.default_rng([2026, 5, seed])
    seq = []
    for _ in range(reps):
        for A in rng.permutation(AMPS):
            seq += [A, 0.0, -A, 0.0]
    step = int(dwell_ms * fs / 1000)
    N = step * (len(seq) + 1)
    t = np.arange(N) * 1000.0 / fs
    x = np.zeros(N)
    xT = np.zeros(N)
    truth = []
    cur = 0.0
    for k, tgt in enumerate(seq, start=1):
        i0 = k * step
        xT[i0:] = tgt
        A = abs(tgt - cur)
        D = S.sacc_duration_ms(A)
        nd = int(round(D * fs / 1000))
        on = i0 + int(lat_ms * fs / 1000)
        prof = S.min_jerk(np.arange(nd + 1) / nd)
        x[on:on + nd + 1] = cur + (tgt - cur) * prof
        x[on + nd + 1:] = tgt
        dur_s = nd / fs          # realised duration (integral at 1000 Hz for these amplitudes)
        truth.append(dict(trial=k - 1, amp=A, pv_true=1.875 * A / dur_s, on_ms=t[on], dur_ms=1000.0 * dur_s))
        cur = tgt
    y = np.zeros(N)
    return dict(t=t, x=x, y=y, valid=np.ones(N, bool), xT=xT, yT=np.zeros(N), fs_nominal=fs, pid=30000 + seed,
                sess=1, task="RAN", truth=truth)


def _filt(name):
    m = next(mm for mm in C.manifest() if mm.name == name)
    return None if m.ftype == "id" else (m.ftype, m.fpar)


def regular_pool():
    """Regular 250 Hz timing (4 ms), no missingness: isolates filter + noise from sampling irregularity."""
    return C.DonorPool([np.full(200000, 4.0)], np.array([10 ** 6]), np.array([], int))


def one(args):
    seed, pool, R = args
    rec = constant_fixation_rec(seed)
    tt, targets = PL.trans_info(rec)
    # window placement only: tiny native jitter so the native detector has a noise floor (member-independent)
    rng = np.random.default_rng([2026, 6, seed])
    xj = rec["x"] + rng.normal(0, 0.003, len(rec["x"]))
    yj = rec["y"] + rng.normal(0, 0.003, len(rec["x"]))
    starts, _ = H.hold_windows(rec["t"], xj, yj, rec["valid"], 1000.0, tt,
                               np.random.default_rng([C.SEED_BASE, 32, rec["pid"]]))
    z = PL.z_of(rec["pid"])
    sH, sV = H.sigma_of(ETA, np.array([z]))
    out = {}
    geos = [C.make_geometry(rec, pool, rec["pid"], r, 50, CM.E_SET, use_missing=False) for r in range(R)]
    for name in MEMBERS:
        m = next(mm for mm in C.manifest() if mm.name == name)
        ch = C.Chain(m)
        A_, B_, C_ = [], [], []
        pv_nf = {A: [] for A in AMPS}
        pv_det = {A: [] for A in AMPS}
        for g in geos:
            st, n = H.grid_window_index(g.grid_t, starts)
            st = H.window_ok(g.usable_g, st, n)
            qf = H.QuadForms(n, H.BANDS)
            s_h, s_v, e_h, e_v = ch.signal_and_noise(g, rec)
            a, b, c = qf.abc(s_h, s_v, e_h, e_v, st)
            A_.append(a)
            B_.append(b)
            C_.append(c)
            # noise-free filtered PV (deterministic; SG velocity on the R4 grid) within each true saccade window
            sp = DET.sg_velocity(s_h, s_v, g.usable_g, H.GRID_FS)[0]
            for tr in rec["truth"]:
                i0 = np.searchsorted(g.grid_t, tr["on_ms"] - 20)
                i1 = np.searchsorted(g.grid_t, tr["on_ms"] + tr["dur_ms"] + 40)
                pv_nf[tr["amp"]].append(float(np.nanmax(sp[i0:i1])) / tr["pv_true"])
            xg, yg = ch.degraded(g, rec, float(sH[0]), float(sV[0]))
            x = np.where(g.usable_g, xg, 0.0)
            y = np.where(g.usable_g, yg, 0.0)
            ev, thr, info = DET.detect(x, y, g.usable_g, H.GRID_FS)
            prim = DET.primary_saccades(ev, g.grid_t, x, y, tt, targets)
            truth_by_trial = {tr["trial"]: tr for tr in rec["truth"]}
            for e in prim:
                tr = truth_by_trial[e["trial"]]
                pv_det[tr["amp"]].append(e["pv"] / tr["pv_true"])
        out[name] = dict(a=np.array(A_), b=np.array(B_), c=np.array(C_), pv_nf=pv_nf, pv_det=pv_det)
    return dict(pid=rec["pid"], z=z, gap=[H.gap_stats(g.valid_g) for g in geos], members=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--P", type=int, default=40)
    ap.add_argument("--R", type=int, default=4)
    a = ap.parse_args()
    pool = regular_pool()
    parts = CM.pmap(one, [(s, pool, a.R) for s in range(a.P)], a.workers)
    res = {}
    coords = {}
    for name in MEMBERS:
        cache = dict(a=np.array([p["members"][name]["a"] for p in parts]),
                     b=np.array([p["members"][name]["b"] for p in parts]),
                     c=np.array([p["members"][name]["c"] for p in parts]),
                     z=np.array([p["z"] for p in parts]), gap=np.array([p["gap"] for p in parts]), nb=len(H.BANDS))
        co = H.coords_gb(cache, ETA)
        coords[name] = co
        r = dict(coords_1_10=co[:10].tolist(), coords_11_12=co[10:].tolist())
        for A in AMPS:
            nf = [v for p in parts for v in p["members"][name]["pv_nf"][A]]
            dt = [v for p in parts for v in p["members"][name]["pv_det"][A]]
            r[f"pv_nf_ratio_{A:g}"] = float(np.exp(np.mean(np.log(nf))))
            r[f"pv_det_ratio_{A:g}"] = float(np.exp(np.mean(np.log(dt)))) if dt else None
            r[f"n_det_{A:g}"] = len(dt)
        res[name] = r
    ref = coords[MEMBERS[0]][:10]
    for n in MEMBERS:
        res[n]["max_rel_coord_dev_1_10"] = float(np.max(np.abs(coords[n][:10] - ref) / np.maximum(np.abs(ref), 1e-12)))
        res[n]["step_settling_ms"] = float(C.step_settling_ms(*_filt(n), 250.0)) if _filt(n) else 0.0
    # core set: members whose step settling is shorter than the 100 ms post-event hold margin; a slower member
    # (BW4-15) leaks post-saccadic ringing into the holds and is reported as the boundary case
    core = [n for n in MEMBERS if res[n]["step_settling_ms"] < H.POST_EVENT_MS]
    spread = max(res[n]["max_rel_coord_dev_1_10"] for n in core)
    pv10 = [res[n]["pv_det_ratio_10"] for n in core if res[n]["pv_det_ratio_10"]]
    pv_gap = float(max(pv10) / min(pv10) - 1) if pv10 else None
    summ = dict(P=a.P, R=a.R, eta=ETA.tolist(), members=res, core_members=core, max_rel_coord_spread_1_10=spread,
                pv10_detected_max_over_min_minus1=pv_gap, success=bool(pv_gap is not None and pv_gap > 0.05
                                                                     and spread < 1e-3))
    p = CM.save("R005", "existence.json", summ, provisional=False)
    print(p)
    print("coord spread", spread, "PV10 gap", pv_gap)
    for n in MEMBERS:
        print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[n].items() if k.startswith("pv")})


if __name__ == "__main__":
    main()
