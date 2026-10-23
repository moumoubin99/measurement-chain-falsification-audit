"""R104 - Block 6 component-addition ladder on real D-GB (100 participants); D split only.

As R006, but the reference is the native GB recording itself: there is no true PV, so every row is
log(PV / native-GB PV) matched on trial (rung 0 = 1 by definition) and binned 8-12 deg on the native amplitude.
Rung 4 uses the median-delta member of A, where delta is the D-only endpoint delta (D-GBVR native, R = 1, vs D-GB
under each member of A at its calibration eta, R = 4, stream ST_DELTA); non-finite deltas are excluded and listed,
and if none is finite the rung-4 ladder is reported unavailable.  The identity counterpart of that member is
B1_id_<noise location>_a<alpha> (same noise location and spectrum, no filter) at the same eta.  If A is empty the
fixed placeholder B1_bw4_30_post_a1 is used and flagged.  All settings (R, R_delta, B, streams, bin) and A / eta
come from the R105 lock, which is verified first; only --workers is free.

Rungs (ordering A): 0 native 1000 Hz detection | 1 +donor sampling/missingness/R4 (identity, sigma = 0; native-ref
windows) | 2 +re-detection | 3 +noise | 4 +filter.  Ordering B: 3' filter without noise | 4' filter + noise.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as CM
import r105_lock as LOCK
from kincost import benchmark as BM
from kincost import chain as C
from kincost import detector as DET
from kincost import endpoint as EP
from kincost import holds as H
from kincost import io_data as IO
from kincost import pipeline as PL
from r006_ladder import LADDER_A, LADDER_B, _boot, _grid_speed, _paired_trials, _part_mean, _window_pv

BIN10 = (8.0, 12.0)
ST_DELTA, ST_LADDER = 6000, 60
PLACEHOLDER = "B1_bw4_30_post_a1"
_W = {}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fork(workers):
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"))


def _vr_suff(member):
    rec = IO.load_gbvr(CM.DEF_ZIP, member)
    return EP.suff(PL.gbvr_native(rec, CM.E_SET, win_ms=_W["win_ms"], bands=_W["bands"])["primary"], _W["knots"])


def _gb_suffs(i):
    rec = _W["recs"][i]
    return BM.endpoint_suffs(rec, _W["pool"], CM.E_SET, _W["chains"], _W["etas"], PL.z_of(rec["pid"]), _W["knots"],
                             geo_rs=list(range(_W["R_delta"])), stream=ST_DELTA)


def _ladder(i):
    rec = _W["recs"][i]
    R, ch_id, ch_f, eta = _W["R"], _W["ch_id"], _W["ch_f"], _W["eta"]
    tt, targets = PL.trans_info(rec)
    sH, sV = BM.sigma(eta, PL.z_of(rec["pid"]))
    usable0 = PL.native_gb_usable(rec, CM.E_SET)
    x0 = np.where(usable0, rec["x"], 0.0)
    y0 = np.where(usable0, rec["y"], 0.0)
    ev0, _, _ = DET.detect(x0, y0, usable0, 1000.0)
    prim0 = DET.primary_saccades(ev0, rec["t"], x0, y0, tt, targets)
    nat = {e["trial"]: (rec["t"][e["on"]], rec["t"][e["off"]], e["pv"], e["amp"]) for e in prim0}
    rows = [(k, am, "r0", pv, -1, am) for k, (_, _, pv, am) in nat.items()]
    for r in range(R):
        g = C.make_geometry(rec, _W["pool"], int(rec["pid"]), r, ST_LADDER, CM.E_SET)
        variants = {"id0": ch_id.degraded(g, rec, 0.0, 0.0), "idn": ch_id.degraded(g, rec, sH, sV),
                    "f0": ch_f.degraded(g, rec, 0.0, 0.0), "fn": ch_f.degraded(g, rec, sH, sV)}
        for vk, (xg, yg) in variants.items():
            x, y, sp = _grid_speed(xg, yg, g.usable_g)
            for k, (on, off, _, am) in nat.items():
                rows.append((k, am, f"{vk}_ref", _window_pv(sp, g.grid_t, on - C.GRID_DT, off + C.GRID_DT), r,
                             np.nan))
            ev, _, _ = DET.detect(x, y, g.usable_g, H.GRID_FS)
            for e in DET.primary_saccades(ev, g.grid_t, x, y, tt, targets):
                if e["trial"] in nat:              # matched on trial; natively undetected trials have no reference
                    rows.append((e["trial"], nat[e["trial"]][3], f"{vk}_det", e["pv"], r, e["amp"]))
    out = {}
    for k, am_nat, key, pv, r, am in rows:
        if not np.isfinite(pv) or pv <= 0:
            continue
        out.setdefault(key, []).append((k, am_nat, np.log(pv / nat[k][2]), r,
                                        np.log(am / am_nat) if np.isfinite(am) and am > 0 else np.nan))
    n_bin = sum(BIN10[0] <= v[3] <= BIN10[1] for v in nat.values())
    return dict(pid=int(rec["pid"]), n_native=len(nat), n_native_bin=int(n_bin),
                rows={k: np.array(v) for k, v in out.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=100)
    a = ap.parse_args()
    lock = LOCK.verify()
    L = lock["ladder"]
    assert (L["stream_delta"], L["stream_ladder"], tuple(L["amp_bin"])) == (ST_DELTA, ST_LADDER, BIN10)
    a.R, a.R_delta, a.B = L["R"], L["R_delta"], L["B"]
    k = LOCK.load_ctx(lock)
    members = C.manifest()
    mbn = {m.name: m for m in members}
    idx, dev, gidx, gdev = LOCK.d_inputs(lock)
    pool = C.DonorPool.from_recordings([IO.load_gbvr(CM.DEF_ZIP, idx[p]) for p in dev])
    assert [int(p) for p in gdev[:IO.N_D_GB]] == k["gb_pids"], "D-GB split differs from the R004 calibration"
    recs = [IO.load_gb(gidx[p]) for p in gdev[:IO.N_D_GB]]
    A = [int(m) for m in lock["A"]]
    eta_full = {members[int(m)].name: np.asarray(e) for m, e in lock["eta"].items()}
    _W.update(recs=recs, pool=pool, knots=k["knots"], win_ms=k["win_ms"], bands=k["bands"], R_delta=a.R_delta)
    res = dict(P=len(recs), R=a.R, B=a.B, amp_bin=BIN10, reference="native GB PV, matched on trial",
               bin_on="native amplitude", A_size=len(A))
    t0 = time.perf_counter()
    if A:
        # D-only delta per member of A -> median-delta member
        _W.update(chains=[C.Chain(members[m]) for m in A], etas=[np.asarray(eta_full[members[m].name]) for m in A])
        with fork(a.workers) as ex:
            vr = list(ex.map(_vr_suff, [idx[p] for p in dev], chunksize=1))
            gb = list(ex.map(_gb_suffs, range(len(recs)), chunksize=1))
        red_vr = EP.reduce_participants(EP.stack([[s] for s in vr]))
        E_vr = EP.estimate(red_vr, k["xref"])[0]
        dhat = {}
        for i, m in enumerate(A):
            red = EP.reduce_participants(EP.stack([g[i] for g in gb]))
            dhat[m] = E_vr - EP.estimate(red, k["xref"])[0]
        fin = sorted((d, m) for m, d in dhat.items() if np.isfinite(d))
        res.update(delta_D={members[m].name: float(d) for m, d in dhat.items()},
                   delta_D_excluded_nonfinite=[members[m].name for m, d in dhat.items() if not np.isfinite(d)])
        if not fin:
            res.update(rung4_member=None, rung4_basis="no finite D-only delta over A: rung-4 ladder unavailable",
                       placeholder=False, delta_s=time.perf_counter() - t0)
            p = CM.save("R104", "ladder.json", res, provisional=False)
            log(f"R104: no finite D-only delta; ladder unavailable ({p})")
            return
        med = np.median([d for d, _ in fin])
        filt = members[min(fin, key=lambda t: (abs(t[0] - med), t[1]))[1]]
        res.update(delta_D_median=float(med), rung4_member=filt.name,
                   rung4_basis="median finite D-only delta over A", placeholder=False)
    else:
        filt = mbn[PLACEHOLDER]
        res.update(rung4_member=filt.name, rung4_basis="A empty: fixed placeholder", placeholder=True)
    res["delta_s"] = time.perf_counter() - t0
    ident = mbn[f"B1_id_{filt.noise_loc}_a{filt.alpha:g}"]
    eta = np.asarray(eta_full[filt.name])
    res.update(identity_member=ident.name, eta=eta.tolist())
    log(f"R104 rung-4 member {filt.name} (identity {ident.name}); |A|={len(A)}")
    _W.update(R=a.R, ch_id=C.Chain(ident), ch_f=C.Chain(filt), eta=eta)
    t0 = time.perf_counter()
    with fork(a.workers) as ex:
        parts = list(ex.map(_ladder, range(len(recs)), chunksize=1))
    res["ladder_s"] = time.perf_counter() - t0
    for name, lad in (("ladder_A", LADDER_A), ("ladder_B", LADDER_B)):
        res[name] = [dict(rung=lab, key=kk, **_boot(_part_mean(parts, kk, BIN10), a.B, i)) for i, (kk, lab) in
                     enumerate(lad)]
    nn = sum(p["n_native_bin"] for p in parts)

    def _count(kk):
        return sum(int(((p["rows"].get(kk, np.zeros((0, 5)))[:, 1] >= BIN10[0]) &
                        (p["rows"].get(kk, np.zeros((0, 5)))[:, 1] <= BIN10[1])).sum()) for p in parts)

    def _rate(num, den):
        return dict(rate=(num / den if den else None), num=int(num), den=int(den))
    res["n_native_bin"] = nn
    res["retention_vs_native"] = {kk: _rate(_count(kk), nn * a.R) for kk in ("id0_det", "idn_det", "f0_det", "fn_det")}
    res["window_availability"] = {kk: _rate(_count(kk), nn * a.R) for kk in ("id0_ref", "idn_ref", "f0_ref", "fn_ref")}
    res["attribution"] = {
        "sampling_R4_native_ref": _boot(_part_mean(parts, "id0_ref", BIN10), a.B, 19),
        "filter_attenuation": _boot(_paired_trials(parts, "f0_ref", "id0_ref", BIN10), a.B, 11),
        "noise_maximum_bias": _boot(_paired_trials(parts, "fn_ref", "f0_ref", BIN10), a.B, 12),
        "noise_maximum_bias_identity": _boot(_paired_trials(parts, "idn_ref", "id0_ref", BIN10), a.B, 16),
        "detector_selection_effect_unpaired": _boot(_part_mean(parts, "fn_det", BIN10) -
                                                    _part_mean(parts, "fn_ref", BIN10), a.B, 13),
        "detector_paired_det_minus_ref": _boot(_paired_trials(parts, "fn_det", "fn_ref", BIN10), a.B, 17),
        "amplitude_ratio_fn_det": _boot(_part_mean(parts, "fn_det", BIN10, col=4), a.B, 20),
        "amplitude_ratio_id0_det": _boot(_part_mean(parts, "id0_det", BIN10, col=4), a.B, 21),
        "total_fn_det": _boot(_part_mean(parts, "fn_det", BIN10), a.B, 15),
    }
    res["notes"] = dict(rung0="identically 1 (native PV is the reference)",
                        no_truth="no latent / true-PV rows on real data; SG-estimator and latent sparse-sampling "
                                 "terms exist only in the synthetic R006")
    p = CM.save("R104", "ladder.json", res, provisional=bool(res["placeholder"]))
    log(f"saved {p}")
    for name in ("ladder_A", "ladder_B"):
        for row in res[name]:
            log(f"  {name} {row['rung']:<30} ratio={row.get('ratio')} [{row.get('ratio_lo')}, {row.get('ratio_hi')}]")


if __name__ == "__main__":
    main()
