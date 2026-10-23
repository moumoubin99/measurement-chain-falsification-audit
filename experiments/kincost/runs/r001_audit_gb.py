"""R001 (GB half) - GazeBase frame / timing / validity audit (lock-safe).

All R1 S1 RAN recordings: dt distribution, validity fraction and codes, transition counts, dwell durations,
target ranges.  D split only: primary-saccade counts, latency, hold-window counts (512 / 256 ms) on the native
settling-excluded usable mask, gaze-target frame correlation and bias.  No C-split saccade quantity is computed.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common as CM
from kincost import detector as DET
from kincost import holds as H
from kincost import io_data as IO
from kincost import pipeline as PL


def basic(path):
    df = pd.read_csv(path, usecols=["n", "val"])
    rec = IO.load_gb(path)
    dt = np.diff(rec["t"])
    idx, tt = IO.transitions(rec)
    dwell = np.diff(tt)
    codes, cnt = np.unique(df["val"].to_numpy(), return_counts=True)
    return dict(pid=rec["pid"], n=len(rec["t"]), dur_s=float((rec["t"][-1] - rec["t"][0]) / 1000),
                dt_median=float(np.median(dt)), dt_frac_1=float(np.mean(np.abs(dt - 1.0) < 0.5)),
                dt_max=float(dt.max()), dt_nonpos=int((dt <= 0).sum()),
                val_codes={int(c): int(k) for c, k in zip(codes, cnt)}, invalid=float(1 - rec["valid"].mean()),
                n_trans=int(len(idx)), dwell_med=float(np.median(dwell)) if len(dwell) else None,
                dwell_min=float(dwell.min()) if len(dwell) else None,
                dwell_max=float(dwell.max()) if len(dwell) else None,
                xT_range=[float(np.nanmin(rec["xT"])), float(np.nanmax(rec["xT"]))],
                yT_range=[float(np.nanmin(rec["yT"])), float(np.nanmax(rec["yT"]))])


def dev_detail(path):
    rec = IO.load_gb(path)
    tt, targets = PL.trans_info(rec)
    st512, _ = PL.native_gb_windows(rec, tt, CM.E_SET)
    st256, _ = PL.native_gb_windows(rec, tt, CM.E_SET, win_ms=H.WIN_MS_FALLBACK)
    usable = PL.native_gb_usable(rec, CM.E_SET)
    x = np.where(usable, rec["x"], 0.0)
    y = np.where(usable, rec["y"], 0.0)
    ev, thr, info = DET.detect(x, y, usable, 1000.0)
    prim = DET.primary_saccades(ev, rec["t"], x, y, tt, targets)
    v = rec["valid"]
    cx = np.corrcoef(rec["x"][v], rec["xT"][v])[0, 1]
    cy = np.corrcoef(rec["y"][v], rec["yT"][v])[0, 1]
    land = [float(np.hypot(x[e["off"]] - targets[e["trial"]][0], y[e["off"]] - targets[e["trial"]][1]))
            for e in prim]
    return dict(pid=rec["pid"], n_events=len(ev), n_primary=len(prim), thr=thr,
                latency_med=float(np.median([e["latency"] for e in prim])) if prim else None,
                landing_med=float(np.median(land)) if land else None,
                amp_q=np.quantile([e["amp"] for e in prim], [0.1, 0.5, 0.9]).tolist() if prim else None,
                n_amp_8_12=int(sum(8 <= e["amp"] <= 12 for e in prim)),
                n_win512=len(st512), n_win256=len(st256), usable=float(usable.mean()),
                corr_x=float(cx), corr_y=float(cy),
                bias_x=float(np.median((rec["x"] - rec["xT"])[usable])),
                bias_y=float(np.median((rec["y"] - rec["yT"])[usable])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    idx, dev, conf = CM.gb_split()
    assert idx, f"no GazeBase R1 S1 RAN recordings under {CM.DEF_GB}"
    pids = list(idx)[: a.limit] if a.limit else list(idx)
    with CM.Timer() as T:
        b = CM.pmap(basic, [idx[p] for p in pids], a.workers)
        devs = [p for p in dev if p in pids]
        d = CM.pmap(dev_detail, [idx[p] for p in devs], a.workers)
    n512 = np.array([x["n_win512"] for x in d])
    n256 = np.array([x["n_win256"] for x in d])
    codes = {}
    for x in b:
        for c, k in x["val_codes"].items():
            codes[c] = codes.get(c, 0) + k
    summ = dict(n_recordings=len(b), n_dev=len(d), n_conf=len(conf), dev=dev, E_set=CM.E_SET,
                dt_median=float(np.median([x["dt_median"] for x in b])),
                dt_frac1_median=float(np.median([x["dt_frac_1"] for x in b])),
                dt_nonpos_total=int(sum(x["dt_nonpos"] for x in b)),
                dt_max_max=float(max(x["dt_max"] for x in b)),
                val_codes_total=codes, invalid_median=float(np.median([x["invalid"] for x in b])),
                invalid_q90=float(np.quantile([x["invalid"] for x in b], 0.9)),
                dur_s_median=float(np.median([x["dur_s"] for x in b])),
                n_trans_median=float(np.median([x["n_trans"] for x in b])),
                dwell_med_median=float(np.median([x["dwell_med"] for x in b if x["dwell_med"] is not None])),
                dev_frac_ge10_win512=float(np.mean(n512 >= H.MIN_WIN)) if len(d) else None,
                dev_frac_ge10_win256=float(np.mean(n256 >= H.MIN_WIN)) if len(d) else None,
                dev_win512_median=float(np.median(n512)) if len(d) else None,
                dev_latency_median=float(np.median([x["latency_med"] for x in d if x["latency_med"]])) if d else None,
                dev_landing_median=float(np.median([x["landing_med"] for x in d if x["landing_med"]])) if d else None,
                dev_primary_median=float(np.median([x["n_primary"] for x in d])) if d else None,
                dev_n812_total=int(sum(x["n_amp_8_12"] for x in d)),
                dev_corr_x_min=float(min(x["corr_x"] for x in d)) if d else None,
                dev_corr_y_min=float(min(x["corr_y"] for x in d)) if d else None,
                runtime_s=T.dt)
    CM.save("R001", "gb_basic.json", b, provisional=False)
    CM.save("R001", "gb_dev_detail.json", d, provisional=False)
    p = CM.save("R001", "gb_summary.json", summ, provisional=False)
    print(p)
    print({k: v for k, v in summ.items() if k != "dev"})


if __name__ == "__main__":
    main()
