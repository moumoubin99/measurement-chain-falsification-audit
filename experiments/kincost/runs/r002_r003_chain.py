"""R002 (chain + detector end to end, per-rule losses, runtime) and R003 (paired cache identity check).

GB data are not yet available, so the GB side uses synthetic GB-like 1000 Hz recordings (synth.gb_like)
with the real D-split GBVR donor pool for timing and missingness.  The identity check is a pure
implementation test and is valid on any input; R002 per-rule losses on synthetic GB are smoke values only.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import common as CM
from kincost import chain as C
from kincost import holds as H
from kincost import pipeline as PL
from kincost import synth as S

ID_MEMBERS = ["B1_id_post_a0", "B1_id_pre_a1", "B1_bw4_15_post_a1", "B1_bw2_60_pre_a0.5", "B1_ma5_post_a1.5",
              "B1_bw4_30_pre_a1.5", "B2_bw4_40_post_a1", "B2_ma20_post_a0", "B2_bw2_100_post_a0.5",
              "B2_bw4_15_post_a1.5"]
ETAS = [np.array([np.log(0.01), 0.3, np.log(0.01), 0.3]), np.array([np.log(0.05), 0.0, np.log(0.03), 0.8]),
        np.array([np.log(0.2), 1.2, np.log(0.1), 0.5])]


def members_by_name():
    return {m.name: m for m in C.manifest()}


def r003_one(args):
    mname, P, R, pool = args
    m = members_by_name()[mname]
    parts = [PL.GBParticipant(S.gb_like(i), pool, CM.E_SET, R, stream=0) for i in range(P)]
    cache = PL.build_cache(parts, m)
    ch = C.Chain(m)
    res = []
    for eta in ETAS:
        c_cache = H.coords_gb(cache, eta)
        Qd = np.full_like(cache["a"], np.nan)
        for p, part in enumerate(parts):
            sH, sV = part.sigma(eta)
            for r, g in enumerate(part.geos):
                st, n = part.win_grid[r]
                st = H.window_ok(g.usable_g, st, n)
                if len(st) == 0:
                    continue
                xg, yg = ch.degraded(g, part.rec, sH, sV)
                Qd[p, r] = H.QuadForms(n, part.bands).window_mean(xg, xg, yg, yg, st)
        c_dir = H.coords_from_Q(Qd, cache["gap"], cache["nb"])
        rel = np.abs(c_cache - c_dir) / np.maximum(np.abs(c_dir), 1e-12)
        res.append(dict(eta=eta.tolist(), coords_cache=c_cache.tolist(), coords_direct=c_dir.tolist(),
                        max_rel=float(rel.max())))
    return dict(member=mname, results=res, max_rel=max(r["max_rel"] for r in res))


def r002_one(args):
    mname, pool = args
    m = members_by_name()[mname]
    rec = S.gb_like(0)
    t0 = time.perf_counter()
    part = PL.GBParticipant(rec, pool, CM.E_SET, 1, stream=0)
    t_geo = time.perf_counter() - t0
    t0 = time.perf_counter()
    a, b, c, nwin = part.hold_cache(m)
    t_cache = time.perf_counter() - t0
    sH, sV = part.sigma(ETAS[0])
    t0 = time.perf_counter()
    ch = C.Chain(m)
    g = part.geos[0]
    xg, yg = ch.degraded(g, rec, sH, sV)
    from kincost import detector as DET
    x = np.where(g.usable_g, xg, 0.0)
    y = np.where(g.usable_g, yg, 0.0)
    ev, thr, info = DET.detect(x, y, g.usable_g, H.GRID_FS)
    prim = DET.primary_saccades(ev, g.grid_t, x, y, part.tt, part.targets)
    t_det = time.perf_counter() - t0
    truth = {s["trial"]: s for s in rec["truth"]}
    ratios = [e["pv"] / truth[e["trial"]]["pv_true"] for e in prim if e["trial"] in truth]
    return dict(member=mname, n_win_native=len(part.win_starts_ms), n_win_r0=int(nwin[0]), det_info=info,
                thr=thr, n_events=len(ev), n_primary=len(prim), n_trans=len(part.tt),
                pv_ratio_median=float(np.median(ratios)) if ratios else None,
                t_geometry_s=t_geo, t_cache_s=t_cache, t_degrade_detect_s=t_det,
                valid_grid_frac=float(g.valid_g.mean()), usable_grid_frac=float(g.usable_g.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--P", type=int, default=20)
    ap.add_argument("--R", type=int, default=10)
    a = ap.parse_args()
    pool = CM.donor_pool()
    r2 = CM.pmap(r002_one, [(m, pool) for m in ID_MEMBERS], a.workers)
    CM.save("R002", "chain_detector.json", r2, provisional=False)
    print("R002 done", [(x["member"], x["n_primary"], round(x["t_degrade_detect_s"], 3)) for x in r2])
    r3 = CM.pmap(r003_one, [(m, a.P, a.R, pool) for m in ID_MEMBERS], a.workers)
    worst = max(x["max_rel"] for x in r3)
    CM.save("R003", "identity.json", dict(P=a.P, R=a.R, worst_max_rel=worst, passed=bool(worst <= 1e-8),
                                          members=r3, note="synthetic GB-like input, real GBVR-D donors"))
    print("R003 worst rel err", worst, "PASS" if worst <= 1e-8 else "FAIL")


if __name__ == "__main__":
    main()
