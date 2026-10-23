"""R006 - Block 6 component-addition ladder and synthetic attribution (synthetic GB-like RAN, known true PV).

Rungs (component-addition ordering A, noise added before the filter):
  0  native 1000 Hz detection on the observed recording (settling exclusion E_set applied, as in the proposal)
  1  + GBVR donor sampling, missingness and R4 grid (identity member, sigma = 0), native-reference windows
     [window availability: a native-reference window counts only if every grid sample in it is usable]
  2  + re-detection on the grid (boundaries from the grid detector)
  3  + post noise at sigma_p (identity member)
  4  + the filter (FILTER member, post noise at sigma_p)
Component-addition ordering B adds the filter before the noise: 3' = filter without noise, 4' = filter + noise.

Synthetic attribution, all as mean log PV ratios over true amplitudes in the 8-12 deg bin (the "10 deg" summary is
this bin average, not exact 10 deg saccades), participant-level means with participant-bootstrap CIs:
  sparse-sampling loss       latent noiseless trajectory through donor sampling + R4 grid + SG velocity, PV in the
                             true saccade window, vs true PV (separately: the SG estimator on the latent at 1000 Hz)
  filter attenuation         log PV(filter, sigma=0) - log PV(identity, sigma=0)   [same native-ref window]
  noise-maximum bias         log PV(filter, sigma)   - log PV(filter, sigma=0)     [same native-ref window]
  detector selection effect  mean over re-detected primaries - mean over available native-ref windows (unpaired;
                             filter + noise); paired (trial, realisation) difference reported alongside, not
                             assumed zero
  amplitude truncation       log(detected amplitude / true amplitude) for re-detected primaries
Paired quantities join on (trial, realisation).  Retention = detected primaries / (true 8-12 deg saccades x R).
On real D-GB data (when available) rung 4 uses the median-delta admissible member and true PV is replaced by
the native GB PV; FILTER here is a fixed placeholder (B1 BW4-30) - see EXPERIMENT_RESULTS.
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

ETA = np.array([np.log(0.02), 0.3, np.log(0.02), 0.3])
BIN10 = (8.0, 12.0)


def _member(name):
    return next(m for m in C.manifest() if m.name == name)


def _grid_speed(xg, yg, usable):
    x = np.where(usable, xg, 0.0)
    y = np.where(usable, yg, 0.0)
    return x, y, DET.sg_velocity(x, y, usable, H.GRID_FS)[0]


def _window_pv(speed, grid_t, on_ms, off_ms):
    """Max grid speed within a native-reference window; NaN unless every grid sample in it is usable."""
    i0 = np.searchsorted(grid_t, on_ms, side="left")
    i1 = np.searchsorted(grid_t, off_ms, side="right")
    if i1 <= i0:
        return np.nan
    v = speed[i0:i1]
    return float(v.max()) if np.isfinite(v).all() else np.nan


def _latent_pv(sp, grid_t, usable, tr, pad=C.GRID_DT):
    """PV of the latent noiseless trajectory in the true saccade window (all samples usable, else NaN)."""
    i0 = np.searchsorted(grid_t, tr["on_ms"] - pad, side="left")
    i1 = np.searchsorted(grid_t, tr["on_ms"] + tr["dur_ms"] + pad, side="right")
    if i1 <= i0 or not usable[i0:i1].all():
        return np.nan
    return float(np.max(sp[i0:i1]))


def one(args):
    seed, pool, R, filt_name, eta = args
    rec = S.gb_like(seed, pid=40000 + seed)
    tt, targets = PL.trans_info(rec)
    truth = {tr["trial"]: tr for tr in rec["truth"]}
    z = PL.z_of(rec["pid"])
    sH, sV = (float(v[0]) for v in H.sigma_of(eta, np.array([z])))
    rec_lat = dict(rec, x=rec["latent_x"], y=rec["latent_y"], valid=rec["valid"])
    rows = []     # (trial, amp_true, key, value, realisation, detected_amp)
    # native SG estimator on the latent trajectory at 1000 Hz, true window
    ok_n = np.ones(len(rec["t"]), bool)
    sp_lat_n = DET.sg_velocity(rec["latent_x"], rec["latent_y"], ok_n, 1000.0)[0]
    for k, tr in truth.items():
        rows.append((k, tr["amp"], "lat_native", _latent_pv(sp_lat_n, rec["t"], ok_n, tr, pad=1.0), -1, np.nan))
    # rung 0: native detection with the proposal's settling exclusion (valid & not within E_set of a gap edge)
    usable0 = PL.native_gb_usable(rec, CM.E_SET)
    x0 = np.where(usable0, rec["x"], 0.0)
    y0 = np.where(usable0, rec["y"], 0.0)
    ev0, _, _ = DET.detect(x0, y0, usable0, 1000.0)
    prim0 = DET.primary_saccades(ev0, rec["t"], x0, y0, tt, targets)
    nat = {e["trial"] + 1: (rec["t"][e["on"]], rec["t"][e["off"]], e["pv"], e["amp"]) for e in prim0}
    for k, (on, off, pv, am) in nat.items():
        rows.append((k, truth[k]["amp"], "r0", pv, -1, am))
    ch_id = C.Chain(_member("B1_id_post_a1"))
    ch_f = C.Chain(_member(filt_name))
    for r in range(R):
        g = C.make_geometry(rec, pool, rec["pid"], r, 60, CM.E_SET)
        # sparse sampling on the latent noiseless trajectory (identity member, sigma = 0), true windows
        xl, yl = ch_id.degraded(g, rec_lat, 0.0, 0.0)
        _, _, sp_l = _grid_speed(xl, yl, g.usable_g)
        for k, tr in truth.items():
            rows.append((k, tr["amp"], "lat_grid", _latent_pv(sp_l, g.grid_t, g.usable_g, tr), r, np.nan))
        variants = {"id0": ch_id.degraded(g, rec, 0.0, 0.0), "idn": ch_id.degraded(g, rec, sH, sV),
                    "f0": ch_f.degraded(g, rec, 0.0, 0.0), "fn": ch_f.degraded(g, rec, sH, sV)}
        for vk, (xg, yg) in variants.items():
            x, y, sp = _grid_speed(xg, yg, g.usable_g)
            # native-reference window (native boundaries padded by one grid step)
            for k, (on, off, _, _) in nat.items():
                rows.append((k, truth[k]["amp"], f"{vk}_ref",
                             _window_pv(sp, g.grid_t, on - C.GRID_DT, off + C.GRID_DT), r, np.nan))
            ev, _, _ = DET.detect(x, y, g.usable_g, H.GRID_FS)
            for e in DET.primary_saccades(ev, g.grid_t, x, y, tt, targets):
                k = e["trial"] + 1
                rows.append((k, truth[k]["amp"], f"{vk}_det", e["pv"], r, e["amp"]))
    out = {}
    for k_, a_, key, pv, r, am in rows:
        if not np.isfinite(pv) or pv <= 0:
            continue
        out.setdefault(key, []).append((k_, a_, np.log(pv / truth[k_]["pv_true"]), r,
                                        np.log(am / a_) if np.isfinite(am) and am > 0 else np.nan))
    n_bin = sum(BIN10[0] <= tr["amp"] <= BIN10[1] for tr in truth.values())
    return dict(pid=rec["pid"], n_true=len(truth), n_true_bin=int(n_bin), n_native_bin=int(
        sum(BIN10[0] <= truth[k]["amp"] <= BIN10[1] for k in nat)), rows={k: np.array(v) for k, v in out.items()})


# ladder quantities (component-addition orderings): key -> rung label
LADDER_A = [("r0", "0 native"), ("id0_ref", "1 +sampling/R4 (native-ref)"), ("id0_det", "2 +re-detection"),
            ("idn_det", "3 +noise"), ("fn_det", "4 +filter")]
LADDER_B = [("r0", "0 native"), ("id0_ref", "1 +sampling/R4 (native-ref)"), ("id0_det", "2 +re-detection"),
            ("f0_det", "3' +filter"), ("fn_det", "4' +noise")]


def _part_mean(parts, key, amp_bin, col=2):
    """Per-participant mean of column col (2 = log PV ratio, 4 = log amplitude ratio) within the bin (NaN if none)."""
    v = []
    for p in parts:
        a = p["rows"].get(key)
        if a is None or len(a) == 0:
            v.append(np.nan)
            continue
        sel = (a[:, 1] >= amp_bin[0]) & (a[:, 1] <= amp_bin[1]) & np.isfinite(a[:, col])
        v.append(a[sel, col].mean() if sel.any() else np.nan)
    return np.array(v)


def _paired_trials(parts, k1, k2, amp_bin):
    """Per-participant mean of (log ratio k1 - log ratio k2) over (trial, realisation) pairs present in both.
    Native-only rows (realisation -1) pair with every realisation of the other key."""
    v = []
    for p in parts:
        a, b = p["rows"].get(k1), p["rows"].get(k2)
        if a is None or b is None:
            v.append(np.nan)
            continue
        def _d(arr):
            d = {}
            for tr, amp, lr, r, _ in arr:
                if amp_bin[0] <= amp <= amp_bin[1]:
                    d[(int(tr), int(r))] = lr
            return d
        da, db = _d(a), _d(b)
        if all(r == -1 for _, r in db) and not all(r == -1 for _, r in da):
            db = {(t, r): db[(t, -1)] for (t, r) in da if (t, -1) in db}
        common = sorted(set(da) & set(db))
        v.append(np.mean([da[c] - db[c] for c in common]) if common else np.nan)
    return np.array(v)


def _boot(v, B, seed):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(est=None, lo=None, hi=None, n=0)
    rng = np.random.default_rng([2026, 66, seed])
    bs = np.array([np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(B)])
    return dict(est=float(np.mean(v)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                n=int(len(v)), ratio=float(np.exp(np.mean(v))), ratio_lo=float(np.exp(np.percentile(bs, 2.5))),
                ratio_hi=float(np.exp(np.percentile(bs, 97.5))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--P", type=int, default=100)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--filter", default="B1_bw4_30_post_a1")
    ap.add_argument("--n_donor", type=int, default=None)
    a = ap.parse_args()
    pool = CM.donor_pool(n=a.n_donor)
    parts = CM.pmap(one, [(s, pool, a.R, a.filter, ETA) for s in range(a.P)], a.workers)
    res = dict(P=a.P, R=a.R, B=a.B, filter=a.filter, eta=ETA.tolist(), amp_bin=BIN10, provisional=True)
    for name, lad in (("ladder_A", LADDER_A), ("ladder_B", LADDER_B)):
        res[name] = [dict(rung=lab, key=k, **_boot(_part_mean(parts, k, BIN10), a.B, i)) for i, (k, lab) in
                     enumerate(lad)]
    # retention of true 8-12 deg saccades per detection rung: detected primaries / (true bin count x R);
    # native-reference rows are window availability (usable native-ref windows / native-detected bin saccades x R)
    nb = sum(p["n_true_bin"] for p in parts)
    nn = sum(p["n_native_bin"] for p in parts)
    def _count(k):
        return sum(int(((p["rows"].get(k, np.zeros((0, 5)))[:, 1] >= BIN10[0]) &
                        (p["rows"].get(k, np.zeros((0, 5)))[:, 1] <= BIN10[1])).sum()) for p in parts)
    res["n_true_bin"] = nb
    def _rate(num, den):
        return dict(rate=(num / den if den else None), num=int(num), den=int(den))
    res["retention"] = {k: _rate(_count(k), nb * (1 if k == "r0" else a.R)) for k in
                        ("r0", "id0_det", "idn_det", "f0_det", "fn_det")}
    res["window_availability"] = {k: _rate(_count(k), nn * a.R) for k in ("id0_ref", "idn_ref", "f0_ref", "fn_ref")}
    res["attribution"] = {
        "sparse_sampling_loss_latent": _boot(_part_mean(parts, "lat_grid", BIN10), a.B, 10),
        "sg_estimator_latent_native_1000hz": _boot(_part_mean(parts, "lat_native", BIN10), a.B, 18),
        "sparse_sampling_observed_native_ref": _boot(_part_mean(parts, "id0_ref", BIN10), a.B, 19),
        "filter_attenuation": _boot(_paired_trials(parts, "f0_ref", "id0_ref", BIN10), a.B, 11),
        "noise_maximum_bias": _boot(_paired_trials(parts, "fn_ref", "f0_ref", BIN10), a.B, 12),
        "detector_selection_effect_unpaired": _boot(_part_mean(parts, "fn_det", BIN10) -
                                                    _part_mean(parts, "fn_ref", BIN10), a.B, 13),
        "detector_paired_det_minus_ref": _boot(_paired_trials(parts, "fn_det", "fn_ref", BIN10), a.B, 17),
        "amplitude_truncation_fn_det": _boot(_part_mean(parts, "fn_det", BIN10, col=4), a.B, 20),
        "amplitude_truncation_id0_det": _boot(_part_mean(parts, "id0_det", BIN10, col=4), a.B, 21),
        "amplitude_truncation_native": _boot(_part_mean(parts, "r0", BIN10, col=4), a.B, 22),
        "native_detection": _boot(_part_mean(parts, "r0", BIN10), a.B, 14),
        "total_fn_det": _boot(_part_mean(parts, "fn_det", BIN10), a.B, 15),
    }
    # interaction check: noise effect without filter (ordering dependence)
    res["attribution"]["noise_maximum_bias_identity"] = _boot(_paired_trials(parts, "idn_ref", "id0_ref", BIN10),
                                                              a.B, 16)
    res["notes"] = dict(amp_bin="8-12 deg bin average of true amplitudes (not exact 10 deg saccades)",
                        orderings="component-addition orderings A (noise then filter) and B (filter then noise)",
                        provisional="synthetic GB-like recordings; real D-GB run deferred (GazeBase unavailable)")
    p = CM.save("R006", "ladder.json", res, provisional=True)
    print(p)
    for name in ("ladder_A", "ladder_B"):
        print(name)
        for row in res[name]:
            print(f"  {row['rung']:<32} ratio={row.get('ratio')}  [{row.get('ratio_lo')}, {row.get('ratio_hi')}]")
    for k, v in res["attribution"].items():
        print(f"{k:<30} {v.get('ratio')} [{v.get('ratio_lo')}, {v.get('ratio_hi')}]")


if __name__ == "__main__":
    main()
