"""R105 - preregistration lock (before any C-split saccade quantity).

`python r105_lock.py --write --reps_per_truth N` reads the D calibration (R004/calibration.json, R004/ctx.pkl)
and writes results/R105/lock.json and results/R105/PREREG.md: code hash over LOCKED_FILES, data manifest
(split participant lists, file names and sizes, GBVR zip size), constants, seeds, knots, x_ref, A, eta for every
member, the benchmark configuration and the confirmatory configuration.  The git commit / tag / bundle of the
locked tree is made by the shell wrapper (lock_git.sh).  `verify()` is called by r102 / r104 / r201 and refuses to
run unless lock.json exists and the current code hash equals the locked one; `load_ctx()` reads the frozen copy of
the calibration context (R105/ctx.pkl) and checks it against the lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

import common as CM
from kincost import benchmark as BM
from kincost import chain as C
from kincost import detector as DET
from kincost import endpoint as EP
from kincost import holds as H
from kincost import io_data as IO

EXP = Path(__file__).resolve().parents[2]          # experiments/
LOCKED_FILES = sorted([*(f"kincost/{p.name}" for p in (EXP / "kincost").glob("*.py")),
                       "kincost/runs/common.py", "kincost/runs/r102_benchmark.py", "kincost/runs/r104_ladder_real.py",
                       "kincost/runs/r105_lock.py", "kincost/runs/r201_confirm.py", "kincost/runs/r006_ladder.py"])
CONFIRM = dict(R_confirm=20, B_confirm=2000, stream_confirm=5000, seed_band_confirm=11000, R_cal=10, B_cal=2000,
               seed_r203=12000, n_low_lambda_r203=10)
BENCH = dict(B_adm=500, B_band=1000, R_bm=BM.R_BM, oracle=True, K_oracle=BM.K_ORACLE)
R103 = dict(truths=["medium"], reps=20, R_bm=20, oracle=False, compare="R102")
LADDER = dict(R=4, R_delta=4, B=2000, stream_delta=6000, stream_ladder=60, amp_bin=[8.0, 12.0],
              selection="member of A whose finite D-only delta is closest to the median finite delta (ties -> lower "
                        "mid); non-finite deltas excluded and listed; none finite -> rung 4 unavailable; A empty -> "
                        "fixed placeholder B1_bw4_30_post_a1 (flagged)")
LOCK_PATH = CM.RES / "R105" / "lock.json"
CTX_PATH = CM.RES / "R105" / "ctx.pkl"            # frozen copy of the R004 calibration context


def code_hashes():
    per = {f: hashlib.sha256((EXP / f).read_bytes()).hexdigest() for f in LOCKED_FILES}
    tot = hashlib.sha256("".join(f"{f}:{h}\n" for f, h in per.items()).encode()).hexdigest()
    return tot, per


def verify():
    if not LOCK_PATH.exists():
        raise SystemExit(f"R105 lock missing ({LOCK_PATH}); the confirmatory C-split runs are refused")
    with open(LOCK_PATH) as f:
        lock = json.load(f)
    tot, per = code_hashes()
    if tot != lock["code_hash"]:
        bad = [f for f in per if per[f] != lock["code_files"].get(f)]
        raise SystemExit(f"code differs from the R105 lock: {bad}")
    return lock


def load_ctx(lock):
    """The frozen calibration context, checked against the lock (A, eta of every member, knots, x_ref, window)."""
    with open(CTX_PATH, "rb") as f:
        ctx = pickle.load(f)
    members = C.manifest()
    assert sorted(int(m) for m in ctx["A"]) == sorted(lock["A"]), "ctx A differs from the lock"
    assert np.allclose(ctx["knots"], lock["knots"]) and np.allclose(ctx["xref"], lock["xref"])
    assert float(ctx["win_ms"]) == lock["window_ms"]
    assert [int(p) for p in ctx["gb_pids"]] == lock["split"]["gb_D"]
    for mid, e in lock["eta"].items():
        assert np.allclose(ctx["eta_full"][members[int(mid)].name], e, equal_nan=True), f"eta differs, mid {mid}"
    return ctx


def d_inputs(lock):
    """D-split inputs (GBVR / GB index and D lists), checked against the locked split and data manifest."""
    idx, dev, _ = CM.gbvr_split()
    gidx, gdev, _ = CM.gb_split()
    sp = lock["split"]
    assert [int(p) for p in dev] == sp["gbvr_D"] and [int(p) for p in gdev] == sp["gb_D"], "D split differs from lock"
    assert os.path.getsize(CM.DEF_ZIP) == lock["gbvr_zip_size"], "GBVR zip differs from the lock"
    assert all(idx[p] == lock["gbvr_members"][str(int(p))] for p in dev), "GBVR D members differ from the lock"
    for p in gdev:
        f = lock["gb_files"][str(int(p))]
        assert os.path.basename(gidx[p]) == f["name"] and os.path.getsize(gidx[p]) == f["size"], f"GB file {p} != lock"
    return idx, dev, gidx, gdev


def _manifest():
    idx, dev, conf = CM.gbvr_split()
    gidx, gdev, gconf = CM.gb_split()
    gb_files = {int(p): dict(name=os.path.basename(gidx[p]), size=os.path.getsize(gidx[p])) for p in gidx}
    return dict(gbvr_zip=os.path.basename(CM.DEF_ZIP), gbvr_zip_size=os.path.getsize(CM.DEF_ZIP),
                gbvr_members={int(p): idx[p] for p in idx}, gb_files=gb_files,
                split=dict(seed=IO.SPLIT_SEED, salts=dict(gbvr=1, gb=2), gbvr_D=[int(p) for p in dev],
                           gbvr_C=[int(p) for p in conf], gb_D=[int(p) for p in gdev], gb_C=[int(p) for p in gconf]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--reps_per_truth", type=int, required=True)
    ap.add_argument("--truths", default=",".join(BM.TRUTHS))
    a = ap.parse_args()
    assert a.write
    with open(CM.RES / "R004" / "calibration.json") as f:
        cal = json.load(f)
    assert not cal["_provisional"] and cal["gb_kind"] == "real", "R105 needs the real-D calibration"
    with open(CM.RES / "R004" / "ctx.pkl", "rb") as f:
        ctx = pickle.load(f)
    assert sorted(ctx["A"]) == sorted(cal["A"])
    members = C.manifest()
    assert cal["R"] == CONFIRM["R_cal"] and cal["B"] == CONFIRM["B_cal"], "calibration R / B differ from the plan"
    assert cal["n_gbvr"] == 50 and cal["n_gb"] == 100 and float(cal["window_ms"]) == float(ctx["win_ms"])
    assert np.allclose(cal["knots"], ctx["knots"]) and np.allclose(cal["xref"], ctx["xref"]), "cal / ctx basis differ"
    assert len(cal["members"]) == len(members) == 276 and [r["mid"] for r in cal["members"]] == list(range(276))
    for r in cal["members"]:
        assert r["name"] == members[r["mid"]].name
        assert np.allclose(ctx["eta_full"][r["name"]], r["eta"], equal_nan=True), f"ctx / calibration eta {r['name']}"
    _, gdev, _ = CM.gb_split()
    assert len(gdev) == IO.N_D_GB == 100 and [int(p) for p in ctx["gb_pids"]] == [int(p) for p in gdev]
    assert len(CM.gbvr_split()[1]) == IO.N_D_GBVR == 50
    tot, per = code_hashes()
    man = _manifest()
    lock = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S %Z"), code_hash=tot, code_files=per, **man,
        A=[int(m) for m in cal["A"]], A_names=cal["A_names"],
        eta={int(r["mid"]): r["eta"] for r in cal["members"]},
        calibration_members=[dict(mid=r["mid"], name=r["name"], Lambda=r["Lambda"], c=r["c"],
                                  admissible=r["admissible"]) for r in cal["members"]],
        knots=list(map(float, ctx["knots"])), xref=list(map(float, ctx["xref"])), window_ms=float(ctx["win_ms"]),
        bands=[list(b) for b in ctx["bands"]], E_set=float(CM.E_SET),
        constants=dict(**CONFIRM, margin=[EP.LO, EP.HI], margin_descriptive=[float(np.log(0.92)),
                       float(np.log(1.08))], band_level=EP.BAND_LEVEL, P_elig_min=EP.P_ELIG_MIN,
                       n812_min=EP.N_812_MIN, cond_max=EP.COND_MAX, min_win=H.MIN_WIN, k_win=H.K_WIN,
                       primary_lat=[80.0, 500.0], primary_amp=[5.0, 15.0], det_min_dur_ms=DET.MIN_DUR_MS,
                       grid_dt=C.GRID_DT, seed_base=C.SEED_BASE, calib_seed=2026, calib_R=cal["R"], calib_B=cal["B"]),
        benchmark=dict(truths=a.truths.split(","), reps_per_truth=a.reps_per_truth, **BENCH, alpha_mc=BM.ALPHA_MC,
                       n812_scale=(IO.N_D_GB // 4) / 220.0,
                       streams=dict(pseudo=BM.ST_PSEUDO, oracle=BM.ST_ORACLE, eval=BM.ST_EVAL),
                       r103=R103),
        ladder=LADDER,
        confirmatory=dict(
            R201="C-GBVR native (left eye, R = 1) vs C-GB (session 1) under every member of A at its D eta, R = 20 "
                 "geometries (stream 5000), shared participant bootstrap B = 2000, max-T 90 % bands, decision rule "
                 "EQUIVALENCE / MISMATCH / INCONCLUSIVE(tag) on [log .95, log 1.05], support thresholds unscaled; "
                 "m*, +-8 % descriptive, native GB delta; A empty -> CALIBRATION FAILURE, no C endpoint test",
            R202="same, 8-12 deg events only, design [log A, 4 harmonics], x_ref = [log 10, harmonic means]",
            R203="Lambda / c recomputed on C-split holds for A u the 10 lowest D Lambda/c members, B = 2000, "
                 "R_cal = 10 (diagnostic only)",
            R302="diagnostics: GBVR right eye; GB session 2 (same members, eta, seeds; the session-2 analysis uses "
                 "the locked C-GB participants that have a session-2 recording, missing ones listed)"),
        deviations=["benchmark (R102/R103) and ladder (R104) run after the lock, with their code and configuration "
                    "frozen in this lock (R102 / R103 / R104 verify the lock and read their settings from it); no benchmark "
                    "output feeds R201-R203",
                    "benchmark replicates per truth set by the 250 CPU-h gate (plan: 100); the CI half-width <= 0.1 "
                    "criterion is then not guaranteed and the achieved precision is reported",
                    "R004 did not persist the calibration GBVR participant IDs; the GBVR D cohort is the deterministic "
                    "split (seed, salt 1) of the same zip, whose size and member names are recorded here and checked "
                    "by every post-lock run"])
    assert set(lock["benchmark"]["truths"]) <= set(BM.TRUTHS) and a.reps_per_truth >= R103["reps"]
    assert set(R103["truths"]) <= set(lock["benchmark"]["truths"]), "R103 reference truth not in the R102 lock"
    p = CM.save("R105", "lock.json", lock, provisional=False)
    with open(CTX_PATH, "wb") as f:
        pickle.dump(ctx, f)
    md = ["# PREREG (R105 lock)", "", f"Created: {lock['created']}", f"Code hash (sha256 over {len(per)} files): "
          f"`{tot}`", "", "## Locked files", *[f"- `{f}` {h}" for f, h in per.items()], "",
          "## Calibration outcome (D split)", f"- |A| = {len(lock['A'])}: {', '.join(lock['A_names']) or '(empty)'}",
          f"- window {lock['window_ms']} ms; E_set {lock['E_set']} ms; knots (deg) "
          f"{[round(float(np.exp(k)), 3) for k in lock['knots']]}", f"- x_ref {lock['xref']}", "",
          "## Split", f"- GBVR D {len(man['split']['gbvr_D'])} / C {len(man['split']['gbvr_C'])}; "
          f"GB D {len(man['split']['gb_D'])} / C {len(man['split']['gb_C'])}; seed {IO.SPLIT_SEED}", "",
          "## Constants", "```json", json.dumps(lock["constants"], indent=1), "```", "",
          "## Benchmark configuration", "```json", json.dumps(lock["benchmark"], indent=1), "```", "",
          "## Ladder configuration (R104)", "```json", json.dumps(lock["ladder"], indent=1), "```", "",
          "## Confirmatory analyses", *[f"- **{k}**: {v}" for k, v in lock["confirmatory"].items()], "",
          "## Deviations", *[f"- {d}" for d in lock["deviations"]], "",
          "Per-member eta, Lambda and c, and the data manifest are in `lock.json`."]
    (CM.RES / "R105" / "PREREG.md").write_text("\n".join(md) + "\n")
    print(p, tot)


if __name__ == "__main__":
    main()
