"""Derive every number used by the figures/tables from experiments/results/** into data/derived.json.

Reads only result files (plus the chain manifest for filter settling times). Run from anywhere.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict
import numpy as np

from paper_plot_style import RES, ROOT, DATA

sys.path.insert(0, os.path.join(ROOT, "experiments"))
from kincost import chain  # noqa: E402


def load(p):
    with open(os.path.join(RES, p)) as f:
        return json.load(f)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


out = {}
cal = load("R004/calibration.json")
out["D_members"] = [dict(name=m["name"], ratio=m["Lambda"] / m["c"], Lambda=m["Lambda"], c=m["c"],
                         admissible=m["admissible"]) for m in cal["members"]]
r = [m["ratio"] for m in out["D_members"]]
out["D_summary"] = dict(n=len(r), n_admissible=sum(m["admissible"] for m in out["D_members"]),
                        min=min(r), median=sorted(r)[len(r) // 2] if len(r) % 2 else
                        0.5 * (sorted(r)[len(r) // 2 - 1] + sorted(r)[len(r) // 2]), max=max(r),
                        n_gb=cal["n_gb"], n_gbvr=cal["n_gbvr"], B=cal["B"])
r203 = load("R203/c_hold_lambda.json")
out["C_R203"] = [dict(name=m["name"], ratio=m["Lambda_over_c"], Lambda=m["Lambda"], c=m["c"],
                      admissible=m["admissible_C"], D_ratio=m["D_Lambda_over_c"]) for m in r203["members"]]
cadm = load("A006/c_adm.json")
dadm = load("A006/d_adm.json")
out["C_A006"] = [dict(name=m["name"], full=m["full"]["ratio"], clean10=m["clean10"]["ratio"],
                      clean10_binding=m["clean10"]["binding"], consistent=m["consistent"]["ratio"],
                      filtered=m["filtered"], in_R203=m["in_R203"]) for m in cadm["members"]]
out["D_A006"] = [dict(name=m["name"], full=m["full"]["ratio"], clean10=m["clean10"]["ratio"],
                      clean10_binding=m["clean10"]["binding"], consistent=m["consistent"]["ratio"],
                      filtered=m["filtered"], z=m["z"]) for m in dadm["members"]]
out["z_names"] = dadm["summary"]["z_names"]
out["A006_summary"] = dict(D=dadm["summary"], C=cadm["summary"])

# benchmark replicates
reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "R102/reps/*.json")))]
by = defaultdict(list)
for x in reps:
    by[x["truth"]].append(x)
summ = load("R102/summary.json")["summary"]
manifest = {m.name: m for m in chain.manifest()}
truth_member = {"light": "B1_bw2_60_post_a1", "medium": "B1_bw4_30_post_a1",
                "heavy": "B1_bw4_15_post_a1", "b2": "B2_bw4_40_post_a1"}
bench = {}
ff = []
for t in ["light", "medium", "heavy", "b2", "vel_adaptive", "kalman_rts"]:
    rs = by[t]
    n = len(rs)
    ne = [x for x in rs if len(x["A"]) > 0]
    ret = sum(1 for x in ne if x.get("retained") is True)
    dec = defaultdict(int)
    for x in rs:
        dec[(x["decision"] + ("/" + x["tag"] if x.get("tag") else ""))] += 1
    for x in rs:
        if len(x["A"]) == 0:
            ff.append(dict(truth=t, rep=x["rep"], min_ratio=min(x["Lambda_over_c"].values())))
    in_family = t in truth_member
    settling = None
    if in_family:
        m = manifest[truth_member[t]]
        settling = chain.step_settling_ms(m.ftype, m.fpar, 250.0 if m.branch == "B1" else 1000.0)
    s = summ[t]
    bench[t] = dict(
        N=n, n_nonempty=len(ne), n_empty=n - len(ne),
        retained=ret if in_family else None,
        retention_given_nonempty=(ret / len(ne)) if (in_family and ne) else None,
        retention_given_nonempty_wilson=wilson(ret, len(ne)) if in_family else None,
        retention_unconditional=(ret / n) if in_family else None,
        decisions=dict(dec),
        n_imprecise=dec.get("INCONCLUSIVE/imprecise", 0),
        n_mismatch=dec.get("MISMATCH", 0), n_equivalence=dec.get("EQUIVALENCE", 0),
        n_false_eq_cond=sum(bool(x.get("false_eq")) for x in rs),
        n_false_mm_cond=sum(bool(x.get("false_mm")) for x in rs),
        n_resolved=s["oracle_status"]["resolved"],
        false_mm_cond_wilson_upper=s["false_mismatch_cond_resolved"][1][1],
        false_mm_wilson_upper_all26=s["false_mismatch_bounds_wilson"][1][1],
        ambiguity=[x["ambiguity"] for x in rs if x.get("ambiguity") is not None],
        ambiguity_q_p10_50_90=s["ambiguity_q"], A_size_q_p10_50_90=s["A_size_q"],
        # quartiles (P25, P50, P75) from the per-replicate values; R102 stores P10/P50/P90
        ambiguity_q=np.quantile([x["ambiguity"] for x in rs if x.get("ambiguity") is not None], [.25, .5, .75]).tolist(),
        A_size_q=np.quantile([len(x["A"]) for x in rs], [.25, .5, .75]).tolist(),
        A_sizes=[len(x["A"]) for x in rs],
        simcov=s["simultaneous_coverage_plugin"]["rate"],
        simcov_wilson=s["simultaneous_coverage_plugin"]["wilson"],
        simcov_n=s["simultaneous_coverage_plugin"]["n_assessed"],
        member=truth_member.get(t), settling_ms=settling, in_family=in_family)
out["bench"] = bench
ff.sort(key=lambda d: d["min_ratio"])
out["bench_false_failures"] = dict(n=len(ff), min=ff[0]["min_ratio"], max=ff[-1]["min_ratio"], items=ff)
OOF = {"vel_adaptive", "kalman_rts"}
ffi = [x for x in ff if x["truth"] not in OOF]; ffo = [x for x in ff if x["truth"] in OOF]
out["bench_false_failures"].update(n_in=len(ffi), min_in=ffi[0]["min_ratio"], max_in=ffi[-1]["min_ratio"],
                                   n_out=len(ffo), min_out=ffo[0]["min_ratio"], max_out=ffo[-1]["min_ratio"])

# missingness ladder
gp = load("A006/gap_pv.json")
gd = load("R004/gap_pv_diagnostics.json")
out["missingness"] = dict(
    gb_raw=gp["native_gb_raw_invalid"]["median"],
    gb_timing_R4=gp["native_gb_through_R4"]["timing_R4_only"]["median_frac"],
    gb_timing_R4_renewal=gp["native_gb_through_R4"]["timing_R4_plus_renewal"]["median_frac"],
    calibration_point=gd["degraded_gb_D_median_frac"],
    gapfree_renewal=gd["synthetic_gapfree_source"]["timing_R4_plus_renewal"]["median_frac"],
    gbvr=gd["native_gbvr_R4"]["median_frac"], gbvr_frac_zero=gd["native_gbvr_R4"]["frac_zero"],
    R=gp["R"])
out["pv10"] = gp["pv10"]
out["native_ratio"] = dict(spline=gp["native_ratio_spline"], linear=gp["native_ratio_linear_8_12"])
lad = load("R104/ladder.json")
out["ladder"] = dict(A=lad["ladder_A"], B=lad["ladder_B"], placeholder_member=lad["rung4_member"])
ex = load("R005/existence.json")
out["R005"] = dict(core=ex["core_members"], members={k: dict(pv10=v["pv_det_ratio_10"],
                   dev=v["max_rel_coord_dev_1_10"], settling=v["step_settling_ms"])
                   for k, v in ex["members"].items()},
                   spread=ex["max_rel_coord_spread_1_10"], pv_spread=ex["pv10_detected_max_over_min_minus1"])
win = load("A006/window.json")
out["window"] = dict(truths={t: dict(member=v["member"], offset=v["max_abs_D_refit_over_s_bench_5_10"],
                     coord=v["argmax_coord"], n_native=v["n_win_native_median"],
                     n_degraded=v["n_win_degraded_median"], control=v["control_coord_max_abs_err"])
                     for t, v in win["truths"].items()}, D_c_quantiles=win["D_c_quantiles"],
                     k_bench=win["k_bench"])
# Same offsets on the real-data scales (post hoc; review item 3): D units use the member's own R004 s_D
# (window.json D_refit_over_sD); C units divide D_refit by the median C s over the 15 A006 C candidates.
_sC = np.median([m["s"][4:10] for m in cadm["members"]], axis=0)
for t, v in win["truths"].items():
    dD = [float(x) for x in v["D_refit_over_sD"][4:10]]
    dC = [float(x) for x in np.array(v["D_refit"][4:10]) / _sC]
    out["window"]["truths"][t].update(
        D_units=dD, C_units=dC, s_bench_units=[float(x) for x in v["D_refit_over_s_bench_5_10"]],
        offset_D=max(abs(x) for x in dD), coord_D=win["names"][4 + int(np.argmax(np.abs(dD)))],
        offset_C=max(abs(x) for x in dC), coord_C=win["names"][4 + int(np.argmax(np.abs(dC)))])
out["window"]["names_5_10"] = win["names"][4:10]
out["window"]["c10_D_median"] = float(np.median([m["clean10"]["c"] for m in dadm["members"]]))
out["window"]["c10_C_median"] = float(np.median([m["clean10"]["c"] for m in cadm["members"]]))
os.makedirs(DATA, exist_ok=True)
with open(os.path.join(DATA, "derived.json"), "w") as f:
    json.dump(out, f, indent=1)
print("D", out["D_summary"])
print("false failures", out["bench_false_failures"]["n"], out["bench_false_failures"]["min"], out["bench_false_failures"]["max"])
for t, b in bench.items():
    print(t, b["n_nonempty"], b["retained"], b["retention_given_nonempty"], b["n_imprecise"], b["n_mismatch"],
          b["n_resolved"], round(b["false_mm_cond_wilson_upper"], 3), b["settling_ms"], round(b["simcov"], 3),
          [round(q, 3) for q in b["ambiguity_q"]])
print(out["missingness"])
