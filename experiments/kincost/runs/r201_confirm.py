"""R201 / R202 / R203 - locked confirmatory C-split RAN decision, 8-12 deg sensitivity, C-hold diagnostic.

Refuses to run unless the R105 lock (results/R105/lock.json) exists and the current code hash equals the locked
hash.  Every constant comes from the lock: A and eta-hat (D calibration), knots, x_ref, window length, seeds.

R201  delta_m = E_GBVR - E_m for every m in A (GB-C degraded by m at eta-hat, R = 20; GBVR-C native, R = 1),
      shared participant bootstrap B = 2000, max-T 90 % bands, preregistered decision, m*, +-8 % (descriptive),
      support per dataset x member (unscaled thresholds), native GB vs GBVR delta.
R202  the same estimator restricted to 8-12 deg events with a linear term in log A (4 harmonics kept).
R203  the admissibility statistic Lambda / c recomputed on C-split holds (diagnostic only) for A u the 10 members
      with the lowest D Lambda / c.
If A is empty, R201 / R202 record CALIBRATION FAILURE and no C endpoint quantity is computed (GBVR holds only, for
R203).  The C cohort is the locked one: the regenerated split and the recording names / sizes must equal the lock.
R302 diagnostics (--eye right, or --gb_session 2) write to R302_RE / R302_S2 (never to R201 / R202) and skip R203;
for session 2 the locked C-GB participants without a session-2 recording are listed and left out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as CM
import r105_lock as LOCK
from kincost import benchmark as BM
from kincost import calibrate as CAL
from kincost import chain as C
from kincost import detector as DET
from kincost import endpoint as EP
from kincost import holds as H
from kincost import io_data as IO
from kincost import pipeline as PL

_W = {}
MARGIN_8 = (math.log(0.92), math.log(1.08))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _vr(member):
    rec = IO.load_gbvr(CM.DEF_ZIP, member, eye=_W["eye"])
    o = PL.gbvr_native(rec, CM.E_SET, win_ms=_W["win_ms"], bands=_W["bands"], want_events=_W["events"])
    out = dict(pid=rec["pid"], Q=o["Q"], gap=o["gap"])
    if _W["events"]:
        out.update(suff=EP.suff(o["primary"], _W["knots"]), lin=EP.suff_lin(o["primary"]))
    return out


def _native_gb_events(rec):
    usable = PL.native_gb_usable(rec, CM.E_SET)
    x = np.where(usable, rec["x"], 0.0)
    y = np.where(usable, rec["y"], 0.0)
    ev, _, _ = DET.detect(x, y, usable, 1000.0)
    tt, targets = PL.trans_info(rec)
    return DET.primary_saccades(ev, rec["t"], x, y, tt, targets)


def _gb(path):
    """One C-GB participant: native rung-0 suff (R = 1) and, per member of A, R = 20 degraded realisations."""
    rec = IO.load_gb(path)
    pid = int(rec["pid"])
    z = PL.z_of(pid)
    nat = _native_gb_events(rec)
    tt, targets = PL.trans_info(rec)
    s = [[None] * _W["R"] for _ in _W["A"]]
    sl = [[None] * _W["R"] for _ in _W["A"]]
    for r in range(_W["R"]):
        geo = C.make_geometry(rec, _W["pool"], pid, r, _W["stream"], CM.E_SET)
        for i, (ch, eta) in enumerate(zip(_W["chains"], _W["etas"])):
            xg, yg = ch.degraded(geo, rec, *BM.sigma(eta, z))
            ev = BM.detect_primary(geo, xg, yg, tt, targets)
            s[i][r] = EP.suff(ev, _W["knots"])
            sl[i][r] = EP.suff_lin(ev)
    return dict(pid=pid, nat=EP.suff(nat, _W["knots"]), nat_lin=EP.suff_lin(nat), s=s, sl=sl)


def _hold_cache(path):
    rec = IO.load_gb(path)
    return CAL.gb_part_caches(rec, _W["pool"], CM.E_SET, _W["R_cal"], 0, _W["win_ms"], _W["bands"],
                              _W["hold_members"])


def fork(workers, **kw):
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), **kw)


def decide_block(red_vr, reds, xref, B, seed, support_vr, support):
    support_ok = support_vr["ok"] and all(s["ok"] for s in support.values())
    bd = EP.bands(red_vr, reds, xref, B, seed)
    dec, tag = EP.decide(bd["lo"], bd["hi"], support_ok, estimable=bd["estimable"])
    out = dict(decision=dec, tag=tag, members=[int(m) for m in bd["members"]], dhat=bd["dhat"], se=bd["se"],
               c=bd["c"], lo=bd["lo"], hi=bd["hi"], ratio=np.exp(bd["dhat"]), ratio_lo=np.exp(bd["lo"]),
               ratio_hi=np.exp(bd["hi"]), E_vr=bd["E_vr"], estimable=bd["estimable"], reason=bd["reason"],
               support_ok=support_ok, support_vr=support_vr, support={int(k): v for k, v in support.items()})
    if support_ok and bd["estimable"]:
        out["m_star"] = EP.m_star(bd["dhat"], bd["se"], bd["c"])
        out["m_star_ratio"] = math.exp(out["m_star"])
        out["within_8pct"] = bool(np.all((bd["lo"] >= MARGIN_8[0]) & (bd["hi"] <= MARGIN_8[1])))
    else:
        out.update(m_star=None, m_star_ratio=None, within_8pct=None)
    out["ambiguity"] = float(np.ptp(bd["dhat"])) if np.isfinite(bd["dhat"]).all() else None
    return out


def native_delta(red_vr, red_nat, xref, B, seed):
    bd = EP.bands(red_vr, {"native": red_nat}, xref, B, seed)
    return dict(dhat=float(bd["dhat"][0]), se=float(bd["se"][0]), ratio=float(np.exp(bd["dhat"][0])),
                ci90=[float(np.exp(bd["lo"][0])), float(np.exp(bd["hi"][0]))] if bd["estimable"] else None,
                estimable=bd["estimable"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--eye", default="left", choices=["left", "right"], help="GBVR eye (R302 diagnostic: right)")
    ap.add_argument("--gb_session", type=int, default=1, choices=[1, 2], help="GB session (R302 diagnostic: 2)")
    ap.add_argument("--skip_r203", action="store_true")
    a = ap.parse_args()
    diag = (a.eye != "left") or (a.gb_session != 1)
    tag = ("_RE" if a.eye == "right" else "") + ("_S2" if a.gb_session == 2 else "")
    run201, run202 = ("R302" + tag, "R302" + tag) if diag else ("R201", "R202")
    lock = LOCK.verify()                           # raises unless the lock exists and the code hash matches
    K = lock["constants"]
    A = [int(m) for m in lock["A"]]
    eta = {int(k): np.array(v) for k, v in lock["eta"].items()}
    members = C.manifest()
    knots, xref = np.array(lock["knots"]), np.array(lock["xref"])
    win_ms = lock["window_ms"]
    bands = H.BANDS if win_ms == H.WIN_MS else H.BANDS_FALLBACK
    R, B, stream = K["R_confirm"], K["B_confirm"], K["stream_confirm"]
    # the locked cohort and inputs
    sp = lock["split"]
    idx, dev, conf = CM.gbvr_split()
    gidx1, gdev, gconf = CM.gb_split()
    assert [int(p) for p in dev] == sp["gbvr_D"] and [int(p) for p in conf] == sp["gbvr_C"], "GBVR split != lock"
    assert [int(p) for p in gdev] == sp["gb_D"] and [int(p) for p in gconf] == sp["gb_C"], "GB split != lock"
    assert os.path.getsize(CM.DEF_ZIP) == lock["gbvr_zip_size"], "GBVR zip differs from the lock"
    assert all(idx[p] == lock["gbvr_members"][str(p)] for p in sp["gbvr_D"] + sp["gbvr_C"]), "GBVR members != lock"
    for p in sp["gb_D"] + sp["gb_C"]:
        f = lock["gb_files"][str(p)]
        assert os.path.basename(gidx1[p]) == f["name"] and os.path.getsize(gidx1[p]) == f["size"], f"GB file {p}"
    gidx_all = IO.gb_index(CM.DEF_GB, sess=a.gb_session)
    missing = [p for p in sp["gb_C"] if p not in gidx_all]
    assert a.gb_session == 2 or not missing, f"locked C-GB recordings missing: {missing}"
    conf = sp["gbvr_C"]
    gconf = [p for p in sp["gb_C"] if p in gidx_all]
    pool = C.DonorPool.from_recordings([IO.load_gbvr(CM.DEF_ZIP, idx[p]) for p in dev])
    _W.update(eye=a.eye, win_ms=win_ms, bands=bands, knots=knots, pool=pool, R=R, stream=stream, A=A,
              chains=[C.Chain(members[m]) for m in A], etas=[eta[m] for m in A], events=bool(A))
    log(f"C split: GBVR {len(conf)}  GB {len(gconf)} (session {a.gb_session}, missing {len(missing)})  |A|={len(A)}  "
        f"R={R}  B={B}  -> {run201}")
    T = {}
    t0 = time.perf_counter()
    with fork(a.workers) as ex:
        vr = list(ex.map(_vr, [idx[p] for p in conf], chunksize=1))
    T["vr_s"] = time.perf_counter() - t0
    res201 = dict(n_gbvr=len(vr), n_gb=len(gconf), R=R, B=B, stream=stream, A=A, eye=a.eye, gb_session=a.gb_session,
                  gb_missing_session=missing, diagnostic=diag)
    res202 = dict(res201)
    if not A:
        res201.update(decision="CALIBRATION FAILURE", tag="", note="A empty: no C-split endpoint test is run")
        res202.update(decision="CALIBRATION FAILURE", tag="")
    else:
        red_vr = EP.reduce_participants(EP.stack([[v["suff"]] for v in vr]))
        red_vr_lin = EP.reduce_participants(EP.stack([[v["lin"]] for v in vr]))
        sup_vr = BM._support(red_vr, EP.estimate(red_vr, xref)[2], 1.0)
        sup_vr_lin = BM._support(red_vr_lin, EP.estimate(red_vr_lin, EP.xref_lin(xref))[2], 1.0)
        t0 = time.perf_counter()
        with fork(a.workers) as ex:
            gb = list(ex.map(_gb, [gidx_all[p] for p in gconf], chunksize=1))
        T["gb_s"] = time.perf_counter() - t0
        reds, reds_lin, sup, sup_lin = {}, {}, {}, {}
        for i, m in enumerate(A):
            reds[m] = EP.reduce_participants(EP.stack([g["s"][i] for g in gb]))
            reds_lin[m] = EP.reduce_participants(EP.stack([g["sl"][i] for g in gb]))
            sup[m] = BM._support(reds[m], EP.estimate(reds[m], xref)[2], 1.0)
            sup_lin[m] = BM._support(reds_lin[m], EP.estimate(reds_lin[m], EP.xref_lin(xref))[2], 1.0)
        red_nat = EP.reduce_participants(EP.stack([[g["nat"]] for g in gb]))
        red_nat_lin = EP.reduce_participants(EP.stack([[g["nat_lin"]] for g in gb]))
        t0 = time.perf_counter()
        res201.update(decide_block(red_vr, reds, xref, B, K["seed_band_confirm"], sup_vr, sup))
        res201["native_gb"] = native_delta(red_vr, red_nat, xref, B, K["seed_band_confirm"] + 1)
        res201["native_gb_support"] = BM._support(red_nat, EP.estimate(red_nat, xref)[2], 1.0)
        res201["member_names"] = [members[m].name for m in A]
        res202.update(decide_block(red_vr_lin, reds_lin, EP.xref_lin(xref), B, K["seed_band_confirm"] + 2,
                                   sup_vr_lin, sup_lin))
        res202["native_gb"] = native_delta(red_vr_lin, red_nat_lin, EP.xref_lin(xref), B,
                                           K["seed_band_confirm"] + 3)
        res202["member_names"] = res201["member_names"]
        T["bands_s"] = time.perf_counter() - t0
    res201["timings"] = T
    CM.save(run201, "confirm.json", res201, provisional=False)
    CM.save(run202, "sensitivity_812_linear.json", res202, provisional=False)
    log(f"{run201} primary: {res201['decision']} {res201.get('tag', '')}  m*={res201.get('m_star_ratio')}  "
        f"native={res201.get('native_gb', {}).get('ratio')}")
    log(f"{run202} 8-12 linear: {res202['decision']} {res202.get('tag', '')}")
    if a.skip_r203 or diag:
        return
    # R203: C-hold admissibility statistic (diagnostic)
    calib = lock["calibration_members"]
    low = sorted((r for r in calib if r["c"] > 0 and np.isfinite(r["Lambda"])), key=lambda r: (r["Lambda"] / r["c"],
                                                                                             r["mid"]))
    hm = sorted(set(A) | {r["mid"] for r in low[:K["n_low_lambda_r203"]]})
    _W.update(hold_members=[members[m] for m in hm], R_cal=K["R_cal"])
    t0 = time.perf_counter()
    with fork(a.workers) as ex:
        parts = list(ex.map(_hold_cache, [gidx_all[p] for p in gconf], chunksize=1))
    nb = len(bands)
    caches = [(m, CAL.stack_gb(parts, i, nb)) for i, m in enumerate(hm)]
    nat = dict(Q=np.array([v["Q"] for v in vr])[:, None, :], gap=np.array([v["gap"] for v in vr])[:, None, :], nb=nb)
    lH, lV, lb, ac1, ac2 = H.participant_coords(nat["Q"][:, 0], nb)
    pp = np.column_stack([lH, lV, lH, lV, lb, ac1, ac2, nat["gap"][:, 0, 0],
                          nat["gap"][:, 0, 1] / np.maximum(nat["gap"][:, 0, 2], 1)])
    nat["pp_sd"] = np.nanstd(pp, 0)
    adm, meta = CAL.run_admissibility(caches, nat, K["B_cal"], seed=K["seed_r203"], workers=min(a.workers, len(hm)),
                                      executor_cls=lambda w, **kw: fork(w, **kw))
    d_by = {r["mid"]: r for r in calib}
    res203 = dict(members=[dict(mid=r["mid"], name=members[r["mid"]].name, Lambda=r["Lambda"], c=r["c"],
                                Lambda_over_c=(r["Lambda"] / r["c"] if r["c"] > 0 else None),
                                admissible_C=r["admissible"], eta_C=r["eta"], flags=r["flags"],
                                D_Lambda_over_c=(d_by[r["mid"]]["Lambda"] / d_by[r["mid"]]["c"]
                                                 if d_by[r["mid"]]["c"] > 0 else None)) for r in adm],
                   basis="A u the 10 lowest D Lambda/c members", in_A=[m in A for m in hm], n_gbvr=len(vr), n_gb=len(gconf),
                   B=K["B_cal"], wall_s=time.perf_counter() - t0, bootstrap_setup=meta)
    CM.save("R203", "c_hold_lambda.json", res203, provisional=False)
    log(f"R203: {sum(r['admissible_C'] for r in res203['members'])}/{len(hm)} non-rejected on C holds")


if __name__ == "__main__":
    main()
