"""R102 / R103 - Block 3 benchmark on real D-GB (100 participants), and the R = 20 alignment check.

Reuses the R004 / R101 real-D calibration context (R004/ctx.pkl: member-indexed D-GB hold caches, frozen knots and
x_ref, calibration eta per member).  Each (truth, replicate) runs serially in its own process; replicates are
checkpointed to <run>/reps/<truth>_<rep>.json so the run is resumable.

Every scientific setting comes from the R105 lock (verified first); only --run and --workers are free:
R102:  --run R102   truths, replicates per truth, B_adm, B_band, R_bm = 4, oracle on   (lock["benchmark"])
R103:  --run R103   medium, 20 replicates, R_bm = 20, no oracle                         (lock["benchmark"]["r103"])
       reports decision agreement with the R = 4 replicate of the same (truth, rep) and the change in eligible
       participants and contributing realisations (Q3 pseudo-target and the Q4 members of A).
Checkpoints carry their configuration and are refused if it differs; the summary covers exactly the locked
(truth, rep) set and says whether it is complete.  Only D-split data are touched.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as CM
import r105_lock as LOCK
from kincost import benchmark as BM
from kincost import chain as C
from kincost import io_data as IO

_W = {}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def build_ctx(lock):
    k = LOCK.load_ctx(lock)
    idx, dev, gidx, gdev = LOCK.d_inputs(lock)
    pool = C.DonorPool.from_recordings([IO.load_gbvr(CM.DEF_ZIP, idx[p]) for p in dev])
    assert [int(p) for p in gdev[:IO.N_D_GB]] == k["gb_pids"], "D-GB split differs from the R004 calibration"
    recs = [IO.load_gb(gidx[p]) for p in gdev[:IO.N_D_GB]]
    return dict(recs=recs, pool=pool, e_set=CM.E_SET, win_ms=k["win_ms"], bands=k["bands"], knots=k["knots"],
                xref=k["xref"], caches=k["caches"], members=C.manifest(), eta_full=k["eta_full"])


def _job(args):
    tn, rep = args
    path = _W["dir"] / f"{tn}_{rep}.json"
    if path.exists():
        with open(path) as f:
            r = json.load(f)
        if r.get("config") != _W["cfg"]:
            raise SystemExit(f"checkpoint {path} has config {r.get('config')}, expected {_W['cfg']}")
        return r
    c0 = time.process_time()
    out = BM.run_replicate(_W["ctx"], tn, rep, B_adm=_W["B_adm"], B_band=_W["B_band"], R_bm=_W["R_bm"],
                           n812_scale=_W["n812_scale"], do_oracle=_W["oracle"], log=log)
    out["timings"]["cpu_s"] = time.process_time() - c0
    out["config"] = _W["cfg"]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, default=CM._default)
    tmp.rename(path)
    return json.loads(json.dumps(out, default=CM._default))


def compare(reps, ref_dir, ref_cfg):
    """R103: per replicate, R = 20 vs the R = 4 run of the same (truth, rep); every reference must exist and carry
    the locked R102 configuration, otherwise the alignment is reported incomplete."""
    rows, missing, bad = [], [], []
    for r in reps:
        p = ref_dir / f"{r['truth']}_{r['rep']}.json"
        if not p.exists():
            missing.append([r["truth"], r["rep"]])
            continue
        with open(p) as f:
            q = json.load(f)
        if q.get("config") != ref_cfg or (q["truth"], q["rep"]) != (r["truth"], r["rep"]):
            bad.append([r["truth"], r["rep"]])
            continue
        row = dict(truth=r["truth"], rep=r["rep"], dec_R20=r["decision"], tag_R20=r.get("tag"),
                   dec_R4=q["decision"], tag_R4=q.get("tag"), agree=r["decision"] == q["decision"],
                   agree_tag=(r["decision"], r.get("tag")) == (q["decision"], q.get("tag")),
                   same_A=sorted(r["A"]) == sorted(q["A"]))
        if "support" in r and "support" in q:
            ms = [m for m in r["support"] if m in q["support"]]
            ok_r = bool(r.get("band_estimable") and r["support_vr"]["ok"] and all(v["ok"] for v in r["support"].values()))
            ok_q = bool(q.get("band_estimable") and q["support_vr"]["ok"] and all(v["ok"] for v in q["support"].values()))
            row.update(
                vr_elig_R20=r["support_vr"]["n_elig"], vr_elig_R4=q["support_vr"]["n_elig"],
                gb_elig_R20_mean=float(np.mean([r["support"][m]["n_elig"] for m in ms])) if ms else None,
                gb_elig_R4_mean=float(np.mean([q["support"][m]["n_elig"] for m in ms])) if ms else None,
                gb_contrib_frac_R20=float(np.mean([r["support"][m]["contrib_frac"] for m in ms])) if ms else None,
                gb_contrib_frac_R4=float(np.mean([q["support"][m]["contrib_frac"] for m in ms])) if ms else None,
                m_star_R20=(max(abs(d) + r["c_band"] * s for d, s in zip(r["dhat"], r["se"]))
                            if ok_r else None),
                m_star_R4=(max(abs(d) + q["c_band"] * s for d, s in zip(q["dhat"], q["se"]))
                           if ok_q else None))
        rows.append(row)
    n = len(rows)
    k = sum(x["agree"] for x in rows)
    return dict(n=n, n_expected=len(reps), complete=not missing and not bad and n == len(reps) > 0,
                missing_reference=missing, incompatible_reference=bad, decision_agreement=[k / n if n else None, BM.wilson(k, n)],
                tag_agreement=(sum(x["agree_tag"] for x in rows) / n if n else None),
                same_A=(sum(x["same_A"] for x in rows) / n if n else None), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="R102", choices=["R102", "R103"])
    ap.add_argument("--workers", type=int, default=120)
    a = ap.parse_args()
    lock = LOCK.verify()
    bc = lock["benchmark"]
    if a.run == "R102":
        truths, reps, R_bm, oracle, compare_to = bc["truths"], bc["reps_per_truth"], bc["R_bm"], bc["oracle"], None
    else:
        r3 = bc["r103"]
        truths, reps, R_bm, oracle, compare_to = r3["truths"], r3["reps"], r3["R_bm"], r3["oracle"], r3["compare"]
    ctx = build_ctx(lock)
    P = len(ctx["recs"])
    n812_scale = (P // 4) / 220.0
    assert n812_scale == bc["n812_scale"]
    cfg = dict(R_bm=R_bm, B_adm=bc["B_adm"], B_band=bc["B_band"], K=bc["K_oracle"], oracle=oracle,
               n812_scale=n812_scale, code_hash=lock["code_hash"])
    d = CM.RES / a.run / "reps"
    d.mkdir(parents=True, exist_ok=True)
    _W.update(ctx=ctx, dir=d, B_adm=bc["B_adm"], B_band=bc["B_band"], R_bm=R_bm, oracle=oracle,
              n812_scale=n812_scale, cfg=cfg)
    # interleave truths so partial runs stay balanced across truths
    jobs = [(tn, r) for r in range(reps) for tn in truths]
    log(f"{a.run}: {len(jobs)} replicates, R_bm={R_bm}, oracle={oracle}, workers={a.workers}")
    t0 = time.perf_counter()
    reps_out = []
    with ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=mp.get_context("fork")) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=1)):
            reps_out.append(r)
            if (i + 1) % 20 == 0:
                log(f"  {i + 1}/{len(jobs)} done")
    wall = time.perf_counter() - t0
    # summarise exactly the locked (truth, rep) set
    allr = [r for r in reps_out if (r["truth"], r["rep"]) in set(jobs)]
    res = dict(n_replicates=len(allr), n_expected=len(jobs), complete=len(allr) == len(jobs), truths=truths,
               reps_per_truth=reps, config=cfg, wall_s=wall,
               cpu_h=float(sum(r["timings"]["cpu_s"] for r in allr) / 3600),
               rep_cpu_s_mean=float(np.mean([r["timings"]["cpu_s"] for r in allr])),
               summary=BM.summarise(allr))
    if compare_to:
        ref_cfg = dict(cfg, R_bm=bc["R_bm"], oracle=bc["oracle"])         # the locked R102 configuration
        res["alignment"] = compare(allr, CM.RES / compare_to / "reps", ref_cfg)
        res["complete"] = res["complete"] and res["alignment"]["complete"]
    p = CM.save(a.run, "summary.json", res, provisional=False)
    log(f"saved {p}")
    for tn, s in res["summary"].items():
        log(f"  {tn}: N={s['N']} retention={s.get('retention')} conclusive={s['conclusive'][0]:.2f} "
            f"falseEQ={s['false_equivalence_bounds']} falseMM={s['false_mismatch_bounds']}")
    if compare_to:
        log(f"  alignment: {res['alignment']['decision_agreement']}")


if __name__ == "__main__":
    main()
