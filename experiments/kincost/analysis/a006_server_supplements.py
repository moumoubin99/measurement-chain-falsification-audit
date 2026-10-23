"""Post-hoc server supplements requested by auto-review round 1 (AGY) and the result-to-claim gate.  Locked code is
only imported, never modified; nothing here changes a preregistered decision.

Parts (each writes results/A006/<part>.json):
  d_adm   D-split admissibility for all 276 members re-run on the identical inputs (R004 ctx caches, native GBVR D
          holds, boot_setup seed 2026, B = 2000, fit seed = mid), with the bootstrap discrepancies Db kept.  Checks
          Lambda and c against R004/calibration.json, then adds
            (i)  a recalibrated clean-window test: coordinates 1-10 only, Lambda_10 against its own max-T cutoff c_10;
            (ii) a consistent-population variant: point estimate on the bootstrap-eligible participants only (the
                 resampling population), bootstrap re-centred on it (affects the two gap coordinates only);
            (iii) per-coordinate standardised discrepancies z_j = D_hat_j / s_j and the binding coordinate.
  c_adm   The same on C-split holds (diagnostic, as R203) for the R203 members plus the 10 lowest D members by the
          recalibrated clean-window ratio; B_cal, seed_r203, R_cal from the lock (LOCK.verify()).
  gap_pv  Native-GB D missingness through donor timing / R4 only vs. + renewal transplant; native GB D PV@10 deg
          (frozen spline and 8-12 deg linear) with participant bootstrap; native GBVR / native GB ratio on D.
  window  Window-placement diagnostic for the in-family benchmark truths: member hold coordinates with the fixed
          native-GB windows (as in calibration) vs. pseudo-target windows placed on the degraded grid, on the same
          geometries and noise (benchmark pseudo-target streams, reps 0..R-1, all 100 D-GB participants).

Usage: python a006_server_supplements.py {d_adm|c_adm|gap_pv|window} [--workers 16]"""
import argparse
import json
import math
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "runs"))
sys.path.insert(0, str(HERE.parent))
import common as CM  # noqa: E402
from kincost import benchmark as BM  # noqa: E402
from kincost import calibrate as CAL  # noqa: E402
from kincost import chain as C  # noqa: E402
from kincost import endpoint as EP  # noqa: E402
from kincost import holds as H  # noqa: E402
from kincost import io_data as IO  # noqa: E402
from kincost import pipeline as PL  # noqa: E402

NAMES = ["med_lH", "med_lV", "iqr_lH", "iqr_lV", "bp_4_10", "bp_10_30", "bp_30_60", "bp_60_125", "ac1", "ac2",
         "gap_frac_median", "gap_loglen_mean"]
N_CLEAN = 10
_W = {}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fork(workers, **kw):
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), **kw)


def load_ctx():
    with open(CM.RES / "R004" / "ctx.pkl", "rb") as f:
        return pickle.load(f)


def bands_of(win_ms):
    return H.BANDS if win_ms == H.WIN_MS else H.BANDS_FALLBACK


def nat_from(vr, nb):
    """Native GBVR hold dict exactly as r004_profile / r201_confirm build it."""
    nat = dict(Q=np.array([v["Q"] for v in vr])[:, None, :], gap=np.array([v["gap"] for v in vr])[:, None, :], nb=nb)
    lH, lV, lb, ac1, ac2 = H.participant_coords(nat["Q"][:, 0], nb)
    pp = np.column_stack([lH, lV, lH, lV, lb, ac1, ac2, nat["gap"][:, 0, 0],
                          nat["gap"][:, 0, 1] / np.maximum(nat["gap"][:, 0, 2], 1)])
    nat["pp_sd"] = np.nanstd(pp, 0)
    return nat


def _vr(member):
    rec = IO.load_gbvr(CM.DEF_ZIP, member)
    o = PL.gbvr_native(rec, CM.E_SET, win_ms=_W["win_ms"], bands=_W["bands"], want_events=_W.get("events", False))
    out = dict(pid=int(rec["pid"]), Q=o["Q"], gap=o["gap"])
    if _W.get("events"):
        out.update(suff=EP.suff(o["primary"], _W["knots"]), lin=EP.suff_lin(o["primary"]),
                   rec=dict(t=rec["t"], valid=rec["valid"]))
    return out


# --------------------------------------------------------------------------------------- admissibility variants
def _maxT(D_hat, Db, s, T, pp_sd, cols):
    """holds.admissibility's zero-SE rule and max-T statistic restricted to coordinate subset cols."""
    scale = np.maximum.reduce([np.abs(T), pp_sd, np.full_like(T, 1e-12)])
    keep = np.zeros(len(T), bool)
    keep[cols] = True
    det_reject = False
    for j in cols:
        if s[j] < 1e-6 * scale[j]:
            if abs(D_hat[j]) < 1e-6 * scale[j]:
                keep[j] = False
            else:
                det_reject = True
    if not keep.any():
        return dict(Lambda=0.0, c=0.0, ratio=None, admissible=not det_reject, det_reject=det_reject, binding=None)
    zz = np.abs(D_hat[keep]) / s[keep]
    Lam = float(zz.max())
    Lstar = np.max(np.abs(Db[:, keep] - D_hat[keep]) / s[keep], axis=1)
    c = float(np.percentile(Lstar, 95))
    return dict(Lambda=Lam, c=c, ratio=(Lam / c if c > 0 else None), admissible=bool((not det_reject) and Lam <= c),
                det_reject=det_reject, binding=NAMES[int(np.flatnonzero(keep)[int(np.argmax(zz))])])


def _adm(args):
    mid, cache = args
    t0 = time.perf_counter()
    nat, idx_gb, sd_fit, nb_ = _W["nat"], _W["idx_gb"], _W["sd_fit"], _W["nat_boot"]
    elig_gb, elig_vr = _W["elig_gb"], _W["elig_vr"]
    pp_sd = nat.get("pp_sd", np.zeros(12))
    T = H.coords_native(nat)
    out = dict(mid=mid)
    if not np.isfinite(T).all():
        return dict(out, estimable=False, reason="nonfinite_native_target")
    eta, _, flags = H.fit_eta(cache, T, sd_fit, seed=mid)
    D_hat = H.coords_gb(cache, eta) - T
    Bn = len(idx_gb)
    Db = np.full((Bn, len(T)), np.nan)
    for bb in range(Bn):
        Tb = nb_[bb]
        if not np.isfinite(Tb).all():
            continue
        eb, _, _ = H.fit_eta(cache, Tb, sd_fit, idx=idx_gb[bb], seed=mid)
        Db[bb] = H.coords_gb(cache, eb, idx_gb[bb]) - Tb
    n_bad = int((~np.isfinite(Db).all(1)).sum())
    out.update(eta=eta.tolist(), flags=flags, D_hat=D_hat.tolist(), n_boot_nonfinite=n_bad,
               Db=Db.astype(np.float32))
    if n_bad or not np.isfinite(D_hat).all():
        return dict(out, estimable=False, reason="nonfinite")
    s = Db.std(0, ddof=1)
    allc = list(range(len(T)))
    full = _maxT(D_hat, Db, s, T, pp_sd, allc)
    clean = _maxT(D_hat, Db, s, T, pp_sd, list(range(N_CLEAN)))
    # (ii) consistent population: point estimate on the eligible participants that the bootstrap resamples
    T_c = H.coords_native(nat, elig_vr)
    eta_c, _, _ = H.fit_eta(cache, T_c, sd_fit, idx=elig_gb, seed=mid)
    D_c = H.coords_gb(cache, eta_c, elig_gb) - T_c
    cons = _maxT(D_c, Db, s, T_c, pp_sd, allc)
    # exploratory a004 profile kept for comparison: Lambda_1-10 over the 12-coordinate c
    out.update(estimable=True, s=s.tolist(), z=(D_hat / s).tolist(), z_cons=(D_c / s).tolist(),
               eta_cons=eta_c.tolist(), D_cons=D_c.tolist(), full=full, clean10=clean, consistent=cons,
               clean10_over_c12=(clean["Lambda"] / full["c"] if full["c"] > 0 else None),
               t_s=time.perf_counter() - t0)
    return out


def run_adm(caches, nat, B, seed, workers):
    idx_gb, idx_vr, sd_raw, nb_, meta = CAL.boot_setup(caches, nat, B, seed)
    a0 = caches[0][1]["a"]
    Q = nat["Q"]
    _W.update(nat=nat, idx_gb=idx_gb, sd_fit=sd_raw, nat_boot=nb_,
              elig_gb=np.flatnonzero(np.isfinite(a0.reshape(a0.shape[0], -1)).any(1)),
              elig_vr=np.flatnonzero(np.isfinite(Q.reshape(Q.shape[0], -1)).all(1)))
    with fork(workers) as ex:
        res = list(ex.map(_adm, caches, chunksize=1))
    return res, meta


def summarise(res, ref, members):
    """ref: {mid: (Lambda, c)} from the original run.  Returns per-member rows, summary, Db stack."""
    rows, Db = [], []
    for r in res:
        m = members[r["mid"]]
        Db.append(r.pop("Db", None))
        row = dict(r, name=m.name, branch=m.branch, filtered=(m.ftype != "id"))
        if r["mid"] in ref and r.get("estimable"):
            L0, c0 = ref[r["mid"]]
            row["repro_abs_err"] = [abs(r["full"]["Lambda"] - L0), abs(r["full"]["c"] - c0)]
        rows.append(row)
    est = [r for r in rows if r.get("estimable")]

    def block(key):
        rat = np.array([r[key]["ratio"] for r in est if r[key]["ratio"] is not None])
        order = sorted((r for r in est if r[key]["ratio"] is not None), key=lambda r: r[key]["ratio"])
        bind = {}
        for r in est:
            if r[key]["binding"] is not None:
                bind[r[key]["binding"]] = bind.get(r[key]["binding"], 0) + 1
        return dict(n_admissible=int(sum(r[key]["admissible"] for r in est)),
                    admissible=[r["name"] for r in est if r[key]["admissible"]],
                    ratio_min=float(rat.min()) if len(rat) else None,
                    ratio_q=np.quantile(rat, [0.1, 0.5, 0.9]).tolist() if len(rat) else None,
                    ratio_min_filtered=float(min((r[key]["ratio"] for r in est if r["filtered"]
                                                  and r[key]["ratio"] is not None), default=np.nan)),
                    lowest10=[(r["name"], r[key]["ratio"], r[key]["binding"]) for r in order[:10]],
                    binding_counts=bind,
                    c_q=np.quantile([r[key]["c"] for r in est], [0.1, 0.5, 0.9]).tolist())
    errs = np.array([r["repro_abs_err"] for r in rows if "repro_abs_err" in r])
    summ = dict(n=len(rows), n_estimable=len(est), full=block("full"), clean10=block("clean10"),
                consistent=block("consistent"),
                repro_max_abs_err=(errs.max(0).tolist() if len(errs) else None),
                clean10_over_c12_min=float(min(r["clean10_over_c12"] for r in est)),
                z_median_abs=np.median(np.abs([r["z"] for r in est]), 0).tolist(),
                z_names=NAMES)
    return rows, summ, Db


def save_db(name, rows, Db):
    d = CM.RES / "A006"
    d.mkdir(parents=True, exist_ok=True)
    ok = [i for i, x in enumerate(Db) if x is not None]
    np.savez_compressed(d / name, mid=np.array([rows[i]["mid"] for i in ok]), Db=np.stack([Db[i] for i in ok]),
                        names=np.array(NAMES))


def part_d_adm(a):
    t0 = time.perf_counter()
    ctx = load_ctx()
    cal = json.loads((CM.RES / "R004" / "calibration.json").read_text())
    members = C.manifest()
    win_ms = ctx["win_ms"]
    bands = ctx["bands"]
    idx, dev, _ = CM.gbvr_split()
    _W.update(win_ms=win_ms, bands=bands, events=False)
    with fork(a.workers) as ex:
        vr = list(ex.map(_vr, [idx[p] for p in dev], chunksize=1))
    nat = nat_from(vr, len(bands))
    assert np.allclose(H.coords_native(nat), cal["native_coords"], equal_nan=True), "native D holds != R004"
    log(f"d_adm: native D holds reproduce R004; {len(ctx['caches'])} members, B={a.B}")
    res, meta = run_adm(ctx["caches"], nat, a.B, 2026, a.workers)
    ref = {r["mid"]: (r["Lambda"], r["c"]) for r in cal["members"]}
    rows, summ, Db = summarise(res, ref, members)
    save_db("d_adm_Db.npz", rows, Db)
    out = dict(note=__doc__.split("\n\n")[0], part="d_adm", B=a.B, boot_seed=2026, bootstrap_setup=meta,
               summary=summ, members=rows, wall_s=time.perf_counter() - t0)
    CM.save("A006", "d_adm.json", out, provisional=False)
    log(f"d_adm: full {summ['full']['n_admissible']}  clean10 {summ['clean10']['n_admissible']}  consistent "
        f"{summ['consistent']['n_admissible']}  repro err {summ['repro_max_abs_err']}")


def _hold_cache(path):
    rec = IO.load_gb(path)
    return CAL.gb_part_caches(rec, _W["pool"], CM.E_SET, _W["R_cal"], 0, _W["win_ms"], _W["bands"],
                              _W["hold_members"])


def part_c_adm(a):
    import r105_lock as LOCK
    t0 = time.perf_counter()
    lock = LOCK.verify()
    K = lock["constants"]
    members = C.manifest()
    win_ms = lock["window_ms"]
    bands = bands_of(win_ms)
    sp = lock["split"]
    idx, dev, conf = CM.gbvr_split()
    gidx, gdev, gconf = CM.gb_split()
    assert [int(p) for p in conf] == sp["gbvr_C"] and [int(p) for p in gconf] == sp["gb_C"], "split != lock"
    gidx_all = IO.gb_index(CM.DEF_GB, sess=1)
    gconf = [p for p in sp["gb_C"] if p in gidx_all]
    calib = lock["calibration_members"]
    low = sorted((r for r in calib if r["c"] > 0 and np.isfinite(r["Lambda"])),
                 key=lambda r: (r["Lambda"] / r["c"], r["mid"]))
    A = [int(m) for m in lock["A"]]
    hm203 = sorted(set(A) | {r["mid"] for r in low[:K["n_low_lambda_r203"]]})
    d = json.loads((CM.RES / "A006" / "d_adm.json").read_text())
    est = [r for r in d["members"] if r.get("estimable") and r["clean10"]["ratio"] is not None]
    low10 = [r["mid"] for r in sorted(est, key=lambda r: (r["clean10"]["ratio"], r["mid"]))[:10]]
    hm = sorted(set(hm203) | set(low10))
    pool = C.DonorPool.from_recordings([IO.load_gbvr(CM.DEF_ZIP, idx[p]) for p in dev])
    _W.update(win_ms=win_ms, bands=bands, events=False, pool=pool, R_cal=K["R_cal"],
              hold_members=[members[m] for m in hm])
    log(f"c_adm: GBVR C {len(conf)}  GB C {len(gconf)}  members {len(hm)} (R203 {len(hm203)} + low clean10)")
    with fork(a.workers) as ex:
        vr = list(ex.map(_vr, [idx[p] for p in conf], chunksize=1))
        parts = list(ex.map(_hold_cache, [gidx_all[p] for p in gconf], chunksize=1))
    nat = nat_from(vr, len(bands))
    caches = [(m, CAL.stack_gb(parts, i, len(bands))) for i, m in enumerate(hm)]
    log(f"c_adm: caches built {time.perf_counter() - t0:.0f}s")
    res, meta = run_adm(caches, nat, K["B_cal"], K["seed_r203"], min(a.workers, len(hm)))
    r203 = json.loads((CM.RES / "R203" / "c_hold_lambda.json").read_text())
    ref = {r["mid"]: (r["Lambda"], r["c"]) for r in r203["members"]}
    rows, summ, Db = summarise(res, ref, members)
    dby = {r["mid"]: r for r in d["members"]}
    for r in rows:
        r["in_R203"] = r["mid"] in hm203
        r["in_low_clean10_D"] = r["mid"] in low10
        r["D_clean10_ratio"] = dby[r["mid"]]["clean10"]["ratio"] if dby[r["mid"]].get("estimable") else None
        r["D_full_ratio"] = dby[r["mid"]]["full"]["ratio"] if dby[r["mid"]].get("estimable") else None
    save_db("c_adm_Db.npz", rows, Db)
    out = dict(note="C-split hold diagnostic (as R203); not a decision.", part="c_adm", B=K["B_cal"],
               boot_seed=K["seed_r203"], bootstrap_setup=meta, members_R203=hm203, members_low_clean10_D=low10,
               n_gbvr=len(vr), n_gb=len(gconf), summary=summ, members=rows, wall_s=time.perf_counter() - t0)
    CM.save("A006", "c_adm.json", out, provisional=False)
    log(f"c_adm: full {summ['full']['n_admissible']}  clean10 {summ['clean10']['n_admissible']}  consistent "
        f"{summ['consistent']['n_admissible']}  repro err {summ['repro_max_abs_err']}")


# --------------------------------------------------------------------------------------- gap / PV anchors
def _gb_gap_pv(path):
    rec = IO.load_gb(path)
    pid = int(rec["pid"])
    gaps = {}
    for lab, miss in (("timing_R4_only", False), ("timing_R4_plus_renewal", True)):
        gaps[lab] = [H.gap_stats(C.make_geometry(rec, _W["pool"], pid, r, 0, CM.E_SET, use_missing=miss).valid_g)
                     for r in range(_W["R"])]
    usable = PL.native_gb_usable(rec, CM.E_SET)
    x = np.where(usable, rec["x"], 0.0)
    y = np.where(usable, rec["y"], 0.0)
    from kincost import detector as DET
    ev, _, _ = DET.detect(x, y, usable, 1000.0)
    tt, targets = PL.trans_info(rec)
    prim = DET.primary_saccades(ev, rec["t"], x, y, tt, targets)
    return dict(pid=pid, raw_invalid=float((~rec["valid"]).mean()), gaps=gaps, suff=EP.suff(prim, _W["knots"]),
                lin=EP.suff_lin(prim), n_primary=len(prim))


def part_gap_pv(a):
    import a005_gap_pv_diagnostics as A5
    t0 = time.perf_counter()
    ctx = load_ctx()
    knots, xref = np.asarray(ctx["knots"]), np.asarray(ctx["xref"])
    idx, dev, _ = CM.gbvr_split()
    gidx, gdev, _ = CM.gb_split()
    assert [int(p) for p in gdev[:IO.N_D_GB]] == [int(p) for p in ctx["gb_pids"]], "D-GB order != R004"
    _W.update(win_ms=ctx["win_ms"], bands=ctx["bands"], events=True, knots=knots, R=a.R)
    with fork(a.workers) as ex:
        vr = list(ex.map(_vr, [idx[p] for p in dev], chunksize=1))
    _W["pool"] = C.DonorPool.from_recordings([v["rec"] for v in vr])
    with fork(a.workers) as ex:
        gb = list(ex.map(_gb_gap_pv, [gidx[p] for p in gdev[:IO.N_D_GB]], chunksize=1))
    gaps = {lab: A5.cohort([g for x in gb for g in x["gaps"][lab]]) for lab in gb[0]["gaps"]}
    raw = np.array([g["raw_invalid"] for g in gb])
    red_vr = EP.reduce_participants(EP.stack([[v["suff"]] for v in vr]))
    red_gb = EP.reduce_participants(EP.stack([[g["suff"]] for g in gb]))
    lr_vr = EP.reduce_participants(EP.stack([[v["lin"]] for v in vr]))
    lr_gb = EP.reduce_participants(EP.stack([[g["lin"]] for g in gb]))

    def ratio(rv, rg, xr, seed):
        bd = EP.bands(rv, {"native": rg}, xr, a.B, seed)
        return dict(dhat=float(bd["dhat"][0]), se=float(bd["se"][0]), ratio_vr_over_gb=float(np.exp(bd["dhat"][0])),
                    band=[float(np.exp(bd["lo"][0])), float(np.exp(bd["hi"][0]))] if bd["estimable"] else None,
                    band_c=float(bd["c"]), estimable=bd["estimable"])
    out = dict(note=("Post-hoc descriptive, D split only (C endpoint not computed under the preregistered A = empty "
                     "stopping rule)."), part="gap_pv",
               native_gb_raw_invalid=dict(median=float(np.median(raw)), mean=float(raw.mean()),
                                          q90=float(np.quantile(raw, 0.9))),
               native_gb_through_R4=gaps, R=a.R,
               pv10=dict(native_gb_spline=A5.boot_E(red_gb, xref, a.B, 7201),
                         native_gb_linear_8_12=A5.boot_E(lr_gb, EP.xref_lin(xref), a.B, 7202),
                         native_gbvr_spline=A5.boot_E(red_vr, xref, a.B, 7101),
                         native_gbvr_linear_8_12=A5.boot_E(lr_vr, EP.xref_lin(xref), a.B, 7102)),
               native_ratio_spline=ratio(red_vr, red_gb, xref, 7301),
               native_ratio_linear_8_12=ratio(lr_vr, lr_gb, EP.xref_lin(xref), 7302),
               n_primary_gb_median=float(np.median([g["n_primary"] for g in gb])),
               wall_s=time.perf_counter() - t0)
    CM.save("A006", "gap_pv.json", out, provisional=False)
    log(json.dumps({k: out[k] for k in ("native_gb_raw_invalid", "native_gb_through_R4", "native_ratio_spline")},
                   default=CM._default))


# --------------------------------------------------------------------------------------- window placement
def _window(i):
    rec = _W["gb"][i]
    out = {}
    for tn in BM.IN_FAMILY:
        m = _W["mbn"][BM.IN_FAMILY[tn]]
        eta = _W["eta_full"][m.name]
        ti = BM.TRUTHS.index(tn)
        part = PL.GBParticipant(rec, _W["pool"], CM.E_SET, _W["R"], BM.ST_PSEUDO + ti, win_ms=_W["win_ms"],
                                bands=_W["bands"])
        a_, b_, c_, nwin = part.hold_cache(m)
        sH, sV = part.sigma(eta)
        ch = C.Chain(m)
        tt, targets = PL.trans_info(rec)
        nq = a_.shape[1]
        Qd, Qc, gd, nd, ovl = np.full((_W["R"], nq), np.nan), np.full((_W["R"], nq), np.nan), [], [], []
        for r, g in enumerate(part.geos):
            xg, yg = ch.degraded(g, rec, sH, sV)
            o = PL.grid_native(g.grid_t, xg, yg, g.valid_g, g.usable_g, tt, targets, part.pid, _W["win_ms"],
                               _W["bands"], want_events=False)
            Qd[r] = o["Q"]
            gd.append(o["gap"])
            nd.append(o["n_win"])
            # control: the fixed native-GB windows applied to the same degraded signal (must equal the cache)
            st, n = part.win_grid[r]
            st = H.window_ok(g.usable_g, st, n)
            if part.hold_eligible and len(st) >= H.MIN_WIN:
                x = np.where(g.usable_g, xg, 0.0)
                y = np.where(g.usable_g, yg, 0.0)
                Qc[r] = H.QuadForms(n, _W["bands"]).window_mean(x, x, y, y, st)
            rng = np.random.default_rng([C.SEED_BASE, 31, int(part.pid)])
            ds, _ = H.hold_windows(g.grid_t, np.where(g.usable_g, xg, 0.0), np.where(g.usable_g, yg, 0.0),
                                   g.usable_g, H.GRID_FS, tt, rng, win_ms=_W["win_ms"])
            ns = np.asarray(part.win_starts_ms)
            ovl.append(float(np.mean([np.min(np.abs(ns - s)) <= C.GRID_DT for s in ds])) if len(ds) and len(ns)
                       else np.nan)
        out[tn] = dict(a=a_, b=b_, c=c_, z=part.z, gap=part.gaps, Qd=Qd, Qc=Qc, gap_d=np.array(gd),
                       n_native=nwin, n_native_placed=len(part.win_starts_ms), n_deg=np.array(nd), overlap=np.array(ovl), sig=(sH, sV))
    return out


def part_window(a):
    t0 = time.perf_counter()
    ctx = load_ctx()
    cal = json.loads((CM.RES / "R004" / "calibration.json").read_text())
    members = C.manifest()
    mbn = {m.name: m for m in members}
    idx, dev, _ = CM.gbvr_split()
    gidx, gdev, _ = CM.gb_split()
    assert [int(p) for p in gdev[:IO.N_D_GB]] == [int(p) for p in ctx["gb_pids"]], "D-GB order != R004"
    pool = C.DonorPool.from_recordings([IO.load_gbvr(CM.DEF_ZIP, idx[p]) for p in dev])
    gb = [IO.load_gb(gidx[p]) for p in gdev[:IO.N_D_GB]]
    _W.update(gb=gb, pool=pool, mbn=mbn, eta_full=ctx["eta_full"], R=a.R, win_ms=ctx["win_ms"], bands=ctx["bands"])
    with fork(a.workers) as ex:
        per = list(ex.map(_window, range(len(gb)), chunksize=1))
    nb = len(ctx["bands"])
    s_by = {r["name"]: np.array(r["s"]) for r in cal["members"]}
    sd_fit = np.asarray(cal["bootstrap_setup"]["sd_fit_raw"])
    # benchmark SE scale: quarters of 25 vs 25 instead of 98 (GB eligible) vs 50 (GBVR)
    k_bench = math.sqrt((1 / 25 + 1 / 25) / (1 / 98 + 1 / 50))
    res = {}
    for tn in BM.IN_FAMILY:
        name = BM.IN_FAMILY[tn]
        eta = np.asarray(ctx["eta_full"][name])
        cache = dict(a=np.array([p[tn]["a"] for p in per]), b=np.array([p[tn]["b"] for p in per]),
                     c=np.array([p[tn]["c"] for p in per]), z=np.array([p[tn]["z"] for p in per]),
                     gap=np.array([p[tn]["gap"] for p in per]), nb=nb)
        co_nat = H.coords_gb(cache, eta)
        Qn = H.eval_Q(cache, eta)
        Qc = np.array([p[tn]["Qc"] for p in per])
        Qd = np.array([p[tn]["Qd"] for p in per])
        gd = np.array([p[tn]["gap_d"] for p in per])
        co_ctl = H.coords_from_Q(Qc, cache["gap"], nb)
        co_deg = H.coords_from_Q(Qd, gd, nb)
        s = s_by[name]
        off = co_deg - co_nat
        # as in admissibility: the member is profiled (eta refit) on the pseudo-target coordinates; coordinates 1-4
        # are then matched exactly (zero-SE drop), so the residual discrepancy lives in coordinates 5-10
        eta_d, _, _ = H.fit_eta(cache, co_deg, sd_fit, seed=mbn[name].mid)
        D_ref = H.coords_gb(cache, eta_d) - co_deg
        zc = D_ref[4:N_CLEAN] / (s[4:N_CLEAN] * k_bench)
        fin = np.isfinite(Qn) & np.isfinite(Qc)
        res[tn] = dict(member=name, eta=eta.tolist(), coords_native_windows=co_nat.tolist(),
                       coords_degraded_windows=co_deg.tolist(), offset=off.tolist(), offset_over_sD=(off / s).tolist(),
                       eta_refit=eta_d.tolist(), D_refit=D_ref.tolist(), D_refit_over_sD=(D_ref / s).tolist(),
                       D_refit_over_s_bench_5_10=zc.tolist(),
                       max_abs_D_refit_over_s_bench_5_10=float(np.max(np.abs(zc))),
                       argmax_coord=NAMES[4 + int(np.argmax(np.abs(zc)))],
                       control_max_rel_err=float(np.nanmax(np.abs(Qc[fin] - Qn[fin]) / np.maximum(np.abs(Qn[fin]),
                                                                                                   1e-12))),
                       control_coord_max_abs_err=float(np.nanmax(np.abs(co_ctl - co_nat))),
                       n_win_native_median=float(np.median([p[tn]["n_native"] for p in per])),
                       n_win_native_placed_median=float(np.median([p[tn]["n_native_placed"] for p in per])),
                       n_win_degraded_median=float(np.median([p[tn]["n_deg"] for p in per])),
                       start_overlap_median=float(np.nanmedian([p[tn]["overlap"] for p in per])),
                       eligible_native=int(np.isfinite(Qn[..., 0]).any(1).sum()),
                       eligible_degraded=int(np.isfinite(Qd[..., 0]).any(1).sum()))
    out = dict(note=("Window-placement diagnostic: identical geometries / noise (benchmark pseudo-target streams), "
                     "member at its full-D eta*; only the hold-window placement differs. s_D from R004; benchmark "
                     "scale approximated by k_bench = sqrt((1/25+1/25)/(1/98+1/50))."), part="window", R=a.R,
               k_bench=k_bench, names=NAMES, truths=res, D_c_quantiles=np.quantile(
                   [r["c"] for r in cal["members"] if np.isfinite(r["c"])], [0.1, 0.5, 0.9]).tolist(),
               start_overlap_note="window starts are random draws among candidates (different seeds), so start "
                                  "overlap is not informative beyond showing the placements differ", wall_s=time.perf_counter() - t0)
    CM.save("A006", "window.json", out, provisional=False)
    for tn, r in res.items():
        log(f"window {tn}: max|D_refit 5-10|/s_bench {r['max_abs_D_refit_over_s_bench_5_10']:.2f} ({r['argmax_coord']}) "
            f"control err {r['control_coord_max_abs_err']:.2e} overlap {r['start_overlap_median']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["d_adm", "c_adm", "gap_pv", "window"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--R", type=int, default=4)
    a = ap.parse_args()
    dict(d_adm=part_d_adm, c_adm=part_c_adm, gap_pv=part_gap_pv, window=part_window)[a.part](a)


if __name__ == "__main__":
    main()
