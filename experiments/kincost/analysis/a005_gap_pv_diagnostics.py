"""Post-hoc descriptive diagnostics requested in auto-review round 1 (AGY items 2 and 5).  Locked code is only
imported, never modified; nothing here feeds a decision.

(1) Missingness decomposition on the R4 grid.  Gap-free synthetic 1000 Hz sources (duration = GB D median) are
    passed through chain.make_geometry with the real D-split GBVR donor pool, with and without the renewal mask:
    timing/R4 alone vs. timing/R4 + participant-pooled renewal.  Compared with native GBVR (regularised), and
    with native GB raw invalid fractions (R001 GB audit).  The native-GB component itself needs GB data (server).
(2) Native GBVR D-split peak velocity at 10 deg: frozen-spline standardised endpoint (R004 knots / xref) and the
    8-12 deg linear sensitivity, with a participant bootstrap.

Usage: python a005_gap_pv_diagnostics.py [--workers 8] [--B 2000]"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "runs"))
import common as CM  # noqa: E402
from kincost import chain as C  # noqa: E402
from kincost import endpoint as EP  # noqa: E402
from kincost import holds as H  # noqa: E402
from kincost import io_data as IO  # noqa: E402
from kincost import pipeline as PL  # noqa: E402

GB_DUR_S = 101.085          # R001 GB audit, dur_s_median
N_SYN = 100                 # = n_D GB
_W = {}


def _init(knots):
    idx, dev, _ = CM.gbvr_split()
    _W.update(idx=idx, knots=knots)


def _vr(pid):
    rec = IO.load_gbvr(CM.DEF_ZIP, _W["idx"][pid])
    o = PL.gbvr_native(rec, CM.E_SET)
    g = o["grid"]["valid"]
    f, sl, n = H.gap_stats(g)
    return dict(pid=pid, gap=(f, sl, n), raw_invalid=float((~rec["valid"]).mean()),
                rec=dict(t=rec["t"], valid=rec["valid"]),
                suff=EP.suff(o["primary"], _W["knots"]), lin=EP.suff_lin(o["primary"]))


def cohort(gaps):
    a = np.asarray(gaps, float)
    return dict(median_frac=float(np.median(a[:, 0])), mean_frac=float(a[:, 0].mean()),
                q75_frac=float(np.quantile(a[:, 0], 0.75)), frac_zero=float((a[:, 0] == 0).mean()),
                mean_loglen=float(a[:, 1].sum() / max(a[:, 2].sum(), 1e-12)))


def boot_E(red, xref, B, seed):
    E = EP.estimate(red, xref)[0]
    P = len(red["elig"])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(P, size=P), minlength=P).astype(float)
        bs.append(EP.estimate(red, xref, w)[0])
    bs = np.asarray(bs)
    bs = bs[np.isfinite(bs)]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return dict(logE=E, pv_deg_s=math.exp(E), ci95_deg_s=[math.exp(lo), math.exp(hi)],
                n_elig=int(red["elig"].sum()), P=P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    cal = json.loads((CM.RES / "R004" / "calibration.json").read_text())
    knots, xref = np.asarray(cal["knots"]), np.asarray(cal["xref"])
    _init(knots)
    dev = CM.gbvr_split()[1]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(knots,)) as ex:
        vr = list(ex.map(_vr, dev, chunksize=1))
    pool = C.DonorPool.from_recordings([v["rec"] for v in vr])

    # (1) gap decomposition with gap-free synthetic sources
    t = np.arange(0.0, GB_DUR_S * 1000.0, 1.0)
    src = dict(t=t, valid=np.ones(len(t), bool))
    syn = {}
    for lab, miss in (("timing_R4_only", False), ("timing_R4_plus_renewal", True)):
        gs = [H.gap_stats(C.make_geometry(src, pool, 900000 + i, 0, 0, CM.E_SET, use_missing=miss).valid_g)
              for i in range(N_SYN)]
        syn[lab] = cohort(gs)
    rl, gl = pool.run_lengths, pool.gap_lengths
    per_p_gap = np.array([v["gap"][0] for v in vr])
    gap_counts = np.array([v["gap"][2] for v in vr])
    pooled_frac = gl.sum() / max(gl.sum() + rl.sum(), 1)
    gb_audit = json.loads((CM.RES / "R001" / "gb_summary.json").read_text()) \
        if (CM.RES / "R001" / "gb_summary.json").exists() else {}
    out = dict(
        note=("Post-hoc descriptive (auto-review round 1). Gap-free synthetic sources isolate the timing/R4 and "
              "pooled-renewal components; the native-GB component requires GB data and is reported only as the "
              "raw-sample invalid fraction from the R001 GB audit."),
        native_gbvr_R4=cohort([v["gap"] for v in vr]),
        native_gbvr_raw_invalid_median=float(np.median([v["raw_invalid"] for v in vr])),
        native_gb_raw_invalid=dict(median=gb_audit.get("invalid_median"), q90=gb_audit.get("invalid_q90")),
        donor_pool=dict(n_runs=int(len(rl)), n_gaps=int(len(gl)), run_mean=float(rl.mean()),
                        run_median=float(np.median(rl)), gap_mean=float(gl.mean()),
                        gap_median=float(np.median(gl)), pooled_raw_gap_frac=float(pooled_frac),
                        max_participant_gap_count_share=float(gap_counts.max() / max(gap_counts.sum(), 1)),
                        participant_R4_frac_max=float(per_p_gap.max())),
        synthetic_gapfree_source=syn,
        degraded_gb_D_median_frac=float(cal["native_coords"][10] + cal["members"][0]["D_hat"][10]),
    )
    # (2) native GBVR D-split PV at 10 deg
    red = EP.reduce_participants(EP.stack([[v["suff"]] for v in vr]))
    red_lin = EP.reduce_participants(EP.stack([[v["lin"]] for v in vr]))
    out["native_gbvr_D_pv10"] = dict(spline_standardised=boot_E(red, xref, a.B, 7101),
                                     linear_8_12=boot_E(red_lin, EP.xref_lin(xref), a.B, 7102),
                                     note="GB native counterpart requires GB data (server); C split not computed "
                                          "under the preregistered A = empty stopping rule.")
    out["wall_s"] = time.time() - t0
    p = CM.RES / "R004" / "gap_pv_diagnostics.json"
    p.write_text(json.dumps(out, indent=1, default=CM._default))
    print(json.dumps(out, indent=1, default=CM._default))


if __name__ == "__main__":
    main()
