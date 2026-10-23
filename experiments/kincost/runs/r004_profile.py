"""R004 - M0 profiling: one complete D calibration (276 members, R_cal, B) + one benchmark replicate per truth.

GB side: real D-GB (100) if GazeBase is present under GB_ROOT, else synthetic GB-like recordings (smoke /
timing only; calibration outcomes on synthetic GB are not results).  GBVR side: real D-split (50).
Outputs: R004/calibration.json, R004/bench_rep.json, R004/profile.json (CPU-h projection vs 250 CPU-h gate).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as CM
from kincost import benchmark as BM
from kincost import calibrate as CAL
from kincost import chain as C
from kincost import endpoint as EP
from kincost import holds as H
from kincost import io_data as IO
from kincost import pipeline as PL
from kincost import synth as S

_W = {}
GATE_CPU_H = 250.0


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fork_pool(workers):
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"))


def _nat(args):
    member, = args
    rec = IO.load_gbvr(CM.DEF_ZIP, member)
    t0 = time.perf_counter()
    o512 = PL.gbvr_native(rec, CM.E_SET)
    o256 = PL.gbvr_native(rec, CM.E_SET, win_ms=H.WIN_MS_FALLBACK, bands=H.BANDS_FALLBACK, want_events=False)
    return dict(rec=rec, Q512=o512["Q"], gap=o512["gap"], n512=o512["n_win"], Q256=o256["Q"], n256=o256["n_win"],
                primary=o512["primary"], t_s=time.perf_counter() - t0)


def _gb_nwin(i):
    rec = _W["gb"][i]
    tt, _ = PL.trans_info(rec)
    st, _ = PL.native_gb_windows(rec, tt, CM.E_SET, win_ms=_W["win_ms"])
    return len(st)


def _cache(i):
    t0, c0 = time.perf_counter(), time.process_time()
    out = CAL.gb_part_caches(_W["gb"][i], _W["pool"], CM.E_SET, _W["R"], 0, _W["win_ms"], _W["bands"],
                             _W["members"])
    out["t_s"] = time.perf_counter() - t0
    out["cpu_s"] = time.process_time() - c0
    return out


def _rung1(i):
    """Reference path GB rung 1: donor sampling + R4 + re-detection, identity filter, no noise."""
    rec = _W["gb"][i]
    m = next(m for m in _W["members"] if m.name == "B1_id_post_a0")
    geo = C.make_geometry(rec, _W["pool"], int(rec["pid"]), 0, 0, CM.E_SET)
    xg, yg = C.Chain(m).degraded(geo, rec, 0.0, 0.0)
    tt, targets = PL.trans_info(rec)
    return BM.detect_primary(geo, xg, yg, tt, targets)


def _bench(args):
    tn, rep = args
    c0 = time.process_time()
    out = BM.run_replicate(_W["ctx"], tn, rep, B_adm=_W["B_adm"], B_band=_W["B_band"],
                           n812_scale=_W["n812_scale"], log=log)
    out["timings"]["cpu_s"] = time.process_time() - c0
    return out


class TreeRSS:
    """Whole-job resident memory: samples the summed RSS of this process and all descendants (Linux /proc)."""

    def __init__(self, every=2.0):
        import os
        import threading
        self.os, self.every, self.peak = os, every, 0.0
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def _tree_rss_mib(self):
        os = self.os
        kids = {}
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    with open(f"/proc/{d}/stat") as f:
                        ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                    kids.setdefault(ppid, []).append(int(d))
                except Exception:
                    pass
        tot, todo = 0, [os.getpid()]
        while todo:
            p = todo.pop()
            todo += kids.get(p, [])
            try:
                with open(f"/proc/{p}/statm") as f:
                    tot += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except Exception:
                pass
        return tot / 2 ** 20

    def _run(self):
        while not self.stop.wait(self.every):
            try:
                self.peak = max(self.peak, self._tree_rss_mib())
            except Exception:
                pass

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set()


def load_gb(n_gb, synthetic):
    if not synthetic:
        idx, dev, _ = CM.gb_split()
        if dev:
            return [IO.load_gb(idx[p]) for p in dev[:n_gb]], "real"
    return [S.gb_like(i, pid=20000 + i) for i in range(n_gb)], "synthetic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--n_gb", type=int, default=IO.N_D_GB)
    ap.add_argument("--B_adm", type=int, default=500)
    ap.add_argument("--B_band", type=int, default=1000)
    ap.add_argument("--truths", default=",".join(BM.TRUTHS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--synthetic_gb", action="store_true")
    ap.add_argument("--members", type=int, default=None, help="limit manifest (smoke only)")
    a = ap.parse_args()
    with TreeRSS() as mem:
        _main(a, mem)


def _main(a, mem):
    T = {}
    members = C.manifest()
    if a.members:
        keep = {n for n in BM.IN_FAMILY.values()} | {"B1_id_post_a0"}
        members = [m for m in members if m.name in keep] + [m for m in members if m.name not in keep][: a.members]
        members = sorted(members, key=lambda m: m.mid)
    # 1. GBVR D native
    idx, dev, _ = CM.gbvr_split()
    t0 = time.perf_counter()
    nats = CM.pmap(_nat, [(idx[p],) for p in dev], a.workers)
    T["gbvr_native_wall_s"] = time.perf_counter() - t0
    T["gbvr_native_cpu_s"] = sum(n["t_s"] for n in nats)
    pool = C.DonorPool.from_recordings([n["rec"] for n in nats])
    gb, gb_kind = load_gb(a.n_gb, a.synthetic_gb)
    log(f"GBVR D={len(nats)}  GB D={len(gb)} ({gb_kind})  members={len(members)}")
    _W.update(gb=gb, pool=pool, R=a.R, members=members)
    # 2. window rule
    _W.update(win_ms=H.WIN_MS, bands=H.BANDS)
    with fork_pool(a.workers) as ex:
        n_gb = list(ex.map(_gb_nwin, range(len(gb))))
    n_vr = [n["n512"] for n in nats]
    use512 = CAL.window_rule(n_gb, n_vr)
    provisional = bool(gb_kind != "real" or a.members is not None)
    if not use512:
        # fallback: re-check the >= 80 % adequacy gate on both datasets at 256 ms; M0 stops if it still fails
        _W.update(win_ms=H.WIN_MS_FALLBACK, bands=H.BANDS_FALLBACK)
        with fork_pool(a.workers) as ex:
            n_gb = list(ex.map(_gb_nwin, range(len(gb))))
        n_vr = [n["n256"] for n in nats]
        if not CAL.window_rule(n_gb, n_vr):
            stop = dict(status="M0_STOP_window_inadequate", gb_kind=gb_kind,
                        frac_gb_ge10_256=float(np.mean(np.array(n_gb) >= H.MIN_WIN)),
                        frac_vr_ge10_256=float(np.mean(np.array(n_vr) >= H.MIN_WIN)))
            CM.save("R004", "calibration.json", stop, provisional=provisional)
            log(f"M0 STOP: window adequacy fails at 256 ms {stop}")
            return
    win_ms, bands = _W["win_ms"], _W["bands"]
    log(f"window rule: {'512' if use512 else '256'} ms")
    # 3. GB caches, all members
    t0 = time.perf_counter()
    with fork_pool(a.workers) as ex:
        parts = list(ex.map(_cache, range(len(gb)), chunksize=1))
    T["cache_wall_s"] = time.perf_counter() - t0
    T["cache_task_wall_s"] = sum(p["t_s"] for p in parts)
    T["cache_cpu_s"] = sum(p["cpu_s"] for p in parts)
    log(f"caches: wall {T['cache_wall_s']:.0f}s cpu {T['cache_cpu_s']:.0f}s")
    nb = len(bands)
    caches = [(m.mid, CAL.stack_gb(parts, i, nb)) for i, m in enumerate(members)]
    Qk = "Q512" if use512 else "Q256"
    nat = dict(Q=np.array([n[Qk] for n in nats])[:, None, :], gap=np.array([n["gap"] for n in nats])[:, None, :],
               nb=nb)
    lH, lV, lb, ac1, ac2 = H.participant_coords(nat["Q"][:, 0], nb)
    pp = np.column_stack([lH, lV, lH, lV, lb, ac1, ac2, nat["gap"][:, 0, 0],
                          nat["gap"][:, 0, 1] / np.maximum(nat["gap"][:, 0, 2], 1)])
    nat["pp_sd"] = np.nanstd(pp, 0)
    # 4. knots / x_ref from the member-independent reference path
    with fork_pool(a.workers) as ex:
        r1 = list(ex.map(_rung1, range(len(gb)), chunksize=1))
    ref = [e for n in nats for e in n["primary"]] + [e for p in r1 for e in p]
    knots = EP.make_knots([e["amp"] for e in ref])
    xref = EP.make_xref(knots, [e["theta"] for e in ref])
    log(f"knots {np.round(np.exp(knots), 2).tolist()}  ref events {len(ref)}")
    # 5. admissibility B
    t0 = time.perf_counter()
    mid2m = {m.mid: m for m in members}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork"), initializer=CAL._init,
                             initargs=_adm_initargs(caches, nat, a.B)) as ex:
        adm = list(ex.map(CAL._adm_worker, caches, chunksize=1))
    T["adm_wall_s"] = time.perf_counter() - t0
    T["adm_task_wall_s"] = sum(r["t_s"] for r in adm)
    T["adm_cpu_s"] = sum(r["cpu_s"] for r in adm)
    A = [r["mid"] for r in adm if r["admissible"]]
    nonest = [r["mid"] for r in adm if not r.get("estimable", True)]
    log(f"admissibility non-estimable: {len(nonest)}")
    log(f"admissibility: |A|={len(A)}/{len(adm)} wall {T['adm_wall_s']:.0f}s cpu {T['adm_cpu_s']:.0f}s")
    by_branch = {b: sum(1 for m in A if mid2m[m].branch == b) for b in ("B1", "B2")}
    eta_full = {mid2m[r["mid"]].name: np.array(r["eta"]) for r in adm}
    calib = dict(gb_kind=gb_kind, n_gbvr=len(nats), n_gb=len(gb), R=a.R, B=a.B, window_ms=win_ms,
                 window_rule_512=use512, frac_gb_ge10=float(np.mean(np.array(n_gb) >= H.MIN_WIN)),
                 frac_vr_ge10=float(np.mean(np.array(n_vr) >= H.MIN_WIN)), knots=knots, xref=xref,
                 A=A, A_names=[mid2m[m].name for m in A], A_by_branch=by_branch,
                 flag_frac_A=(sum(any(r["flags"].values()) for r in adm if r["admissible"]) / len(A)) if A else None,
                 boundary_n=sum(r["flags"]["boundary"] for r in adm), zero_se_drops=sum(len(r["dropped"]) for r in adm),
                 det_reject_n=sum(r["det_reject"] for r in adm), nonestimable_n=len(nonest),
                 nonestimable_names=[mid2m[m].name for m in nonest], bootstrap_setup=_W.get("adm_meta"),
                 native_coords=H.coords_native(nat).tolist(),
                 members=[dict(mid=r["mid"], name=mid2m[r["mid"]].name, Lambda=r["Lambda"], c=r["c"],
                               admissible=r["admissible"], eta=r["eta"], flags=r["flags"], D_hat=r["D_hat"], s=r["s"],
                               dropped=r["dropped"], det_reject=r["det_reject"], estimable=r.get("estimable"),
                               reason=r.get("reason")) for r in adm])
    CM.save("R004", "calibration.json", calib, provisional=provisional)
    # 6. benchmark replicate(s): one process per (truth, rep), serial inside
    P = len(gb)
    ctx = dict(recs=gb, pool=pool, e_set=CM.E_SET, win_ms=win_ms, bands=bands, knots=knots, xref=xref,
               caches=caches, members=[mid2m[m.mid] for m in C.manifest() if m.mid in mid2m] if a.members is None
               else _dense(members), eta_full=eta_full)
    if a.members:
        ctx["caches"] = [(i, c) for i, (_, c) in enumerate(caches)]   # dense re-index for the smoke manifest
    if not provisional:
        # persist the member-indexed D-GB caches and frozen endpoint constants for the M1 drivers (R102-R104)
        import pickle
        (CM.RES / "R004").mkdir(parents=True, exist_ok=True)
        with open(CM.RES / "R004" / "ctx.pkl", "wb") as f:
            pickle.dump(dict(caches=caches, knots=knots, xref=xref, eta_full=eta_full, win_ms=win_ms,
                             bands=bands, gb_pids=[int(r["pid"]) for r in gb], A=A), f)
    _W.update(ctx=ctx, B_adm=a.B_adm, B_band=a.B_band, n812_scale=(P // 4) / 220.0)
    jobs = [(tn, r) for tn in a.truths.split(",") for r in range(a.reps)]
    t0 = time.perf_counter()
    with fork_pool(min(a.workers, len(jobs))) as ex:
        reps = list(ex.map(_bench, jobs, chunksize=1))
    T["bench_wall_s"] = time.perf_counter() - t0
    CM.save("R004", "bench_rep.json", dict(replicates=reps, summary=BM.summarise(reps)), provisional=provisional)
    # 7. projection
    rep_cpu = float(np.mean([r["timings"]["cpu_s"] for r in reps]))
    full_members = 276
    scale_m = full_members / len(members)
    calib_cpu_h = (T["cache_cpu_s"] + T["adm_cpu_s"]) * scale_m / 3600
    mean_A = float(np.mean([len(r["A"]) for r in reps]))
    proj = dict(timings=T, rep_cpu_s_mean=rep_cpu, rep_timings=[r["timings"] for r in reps], mean_A_bench=mean_A,
                calib_cpu_h=calib_cpu_h, bench_cpu_h_600=600 * rep_cpu / 3600,
                total_projected_cpu_h=calib_cpu_h + 600 * rep_cpu / 3600 + 4 + 3,
                gate_cpu_h=GATE_CPU_H, members_profiled=len(members), member_scale=scale_m,
                note="admissibility / cache costs scale with members; replicate cost scales with |A_r| (oracle)")
    # the production gate is only evaluated on a non-provisional (real D-GB, full manifest) profile
    proj["within_gate"] = None if provisional else bool(proj["total_projected_cpu_h"] <= GATE_CPU_H)
    proj["cpu_basis"] = "process_time per task (cache, admissibility, replicate); wall times reported separately"
    proj["peak_tree_rss_mib"] = mem.peak
    for k, red in [("reps_60_per_truth", 360)]:
        proj[k + "_cpu_h"] = calib_cpu_h + red * rep_cpu / 3600 + 7
    CM.save("R004", "profile.json", proj, provisional=provisional)
    log(f"projection: {proj['total_projected_cpu_h']:.1f} CPU-h (gate {GATE_CPU_H}); replicate {rep_cpu:.0f}s")
    for r in reps:
        log(f"  {r['truth']}: |A|={len(r['A'])} retained={r['retained']} decision={r['decision']} {r.get('tag', '')}"
            f" oracle={r.get('oracle', {}).get('category')}")


def _dense(members):
    """Smoke mode: renumber the subset so that ctx['members'][mid] works."""
    return [C.Member(i, m.branch, m.ftype, m.fpar, m.noise_loc, m.alpha) for i, m in enumerate(members)]


def _adm_initargs(caches, nat, B):
    idx_gb, idx_vr, sd_raw, nb_, meta = CAL.boot_setup(caches, nat, B, 2026)
    _W["adm_meta"] = meta
    return (nat, idx_gb, idx_vr, sd_raw, nb_)


if __name__ == "__main__":
    main()
