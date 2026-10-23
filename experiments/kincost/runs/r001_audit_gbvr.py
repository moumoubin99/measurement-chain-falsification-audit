"""R001 - GBVR frame / timing / dt audit (lock-safe).

All R1 S1 RAN recordings: dt distribution, NaN fractions, transition counts, dwell durations.
D split only: saccade latency, landing error, primary-saccade counts, hold-window counts (512 / 256 ms),
gap coordinates, gaze-target frame correlation.  No C-split saccade quantity is computed.
"""
from __future__ import annotations

import argparse

import numpy as np

import common as CM
from kincost import holds as H
from kincost import io_data as IO
from kincost import pipeline as PL


def basic(member):
    rec = IO.load_gbvr(CM.DEF_ZIP, member)
    rr = IO.load_gbvr(CM.DEF_ZIP, member, eye="right")
    dt = np.diff(rec["t"])
    idx, tt = IO.transitions(rec)
    dwell = np.diff(tt)
    u, cnt = np.unique(np.round(dt, 3), return_counts=True)
    top = np.argsort(cnt)[::-1][:6]
    return dict(pid=rec["pid"], n=len(rec["t"]), dur_s=float((rec["t"][-1] - rec["t"][0]) / 1000),
                dt_median=float(np.median(dt)), dt_frac_4=float(np.mean(np.abs(dt - 4.0) < 0.5)),
                dt_max=float(dt.max()), dt_nonpos=int((dt <= 0).sum()), dt_top=[[float(u[i]), int(cnt[i])] for i in top],
                nan_left=float(1 - rec["valid"].mean()), nan_right=float(1 - rr["valid"].mean()),
                n_trans=int(len(idx)), dwell_med=float(np.median(dwell)) if len(dwell) else None,
                dwell_min=float(dwell.min()) if len(dwell) else None, dwell_max=float(dwell.max()) if len(dwell) else None,
                dwell_mod4=float(np.mean(np.abs(np.mod(dwell + 2, 4.0) - 2) < 0.5)) if len(dwell) else None,
                xT_range=[float(np.nanmin(rec["xT"])), float(np.nanmax(rec["xT"]))],
                yT_range=[float(np.nanmin(rec["yT"])), float(np.nanmax(rec["yT"]))])


def dev_detail(member):
    rec = IO.load_gbvr(CM.DEF_ZIP, member)
    nat = PL.gbvr_native(rec, CM.E_SET)
    nat256 = PL.gbvr_native(rec, CM.E_SET, win_ms=H.WIN_MS_FALLBACK, bands=H.BANDS_FALLBACK, want_events=False)
    prim = nat["primary"]
    v = rec["valid"]
    cx = np.corrcoef(rec["x"][v], rec["xT"][v])[0, 1]
    cy = np.corrcoef(rec["y"][v], rec["yT"][v])[0, 1]
    land = []
    tt, targets = PL.trans_info(rec)
    g = nat["grid"]
    for e in prim:
        tx, ty = targets[e["trial"]]
        land.append(float(np.hypot(g["x"][e["off"]] - tx, g["y"][e["off"]] - ty)))
    # median fixation offset (systematic frame bias) in the hold windows region: gaze - target over usable samples
    ug = nat["usable"]
    xT_g = np.interp(g["t"], rec["t"], rec["xT"])
    yT_g = np.interp(g["t"], rec["t"], rec["yT"])
    return dict(pid=rec["pid"], n_events=len(nat["events"]), n_primary=len(prim), det_info=nat["det_info"],
                thr=nat["thr"], latency_med=float(np.median([e["latency"] for e in prim])) if prim else None,
                landing_med=float(np.median(land)) if land else None,
                amp_q=np.quantile([e["amp"] for e in prim], [0.1, 0.5, 0.9]).tolist() if prim else None,
                n_win512=nat["n_win"], n_win256=nat256["n_win"], gap=list(nat["gap"]),
                corr_x=float(cx), corr_y=float(cy),
                bias_x=float(np.median((g["x"] - xT_g)[ug])), bias_y=float(np.median((g["y"] - yT_g)[ug])),
                Q=nat["Q"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    idx, dev, conf = CM.gbvr_split()
    pids = list(idx)[: a.limit] if a.limit else list(idx)
    with CM.Timer() as T:
        b = CM.pmap(basic, [idx[p] for p in pids], a.workers)
        devs = [p for p in dev if p in pids]
        d = CM.pmap(dev_detail, [idx[p] for p in devs], a.workers)
    n512 = np.array([x["n_win512"] for x in d])
    n256 = np.array([x["n_win256"] for x in d])
    summ = dict(n_recordings=len(b), n_dev=len(d), n_conf=len(conf), dev=dev, E_set=CM.E_SET, E_worst=CM.E_WORST,
                dt_median=float(np.median([x["dt_median"] for x in b])),
                dt_frac4_median=float(np.median([x["dt_frac_4"] for x in b])),
                dt_nonpos_total=int(sum(x["dt_nonpos"] for x in b)),
                nan_left_median=float(np.median([x["nan_left"] for x in b])),
                n_trans_median=float(np.median([x["n_trans"] for x in b])),
                dwell_mod4_median=float(np.median([x["dwell_mod4"] for x in b if x["dwell_mod4"] is not None])),
                dev_frac_ge10_win512=float(np.mean(n512 >= H.MIN_WIN)) if len(d) else None,
                dev_frac_ge10_win256=float(np.mean(n256 >= H.MIN_WIN)) if len(d) else None,
                dev_latency_median=float(np.median([x["latency_med"] for x in d if x["latency_med"]])) if d else None,
                dev_landing_median=float(np.median([x["landing_med"] for x in d if x["landing_med"]])) if d else None,
                dev_primary_median=float(np.median([x["n_primary"] for x in d])) if d else None,
                dev_corr_x_min=float(min(x["corr_x"] for x in d)) if d else None,
                dev_corr_y_min=float(min(x["corr_y"] for x in d)) if d else None,
                runtime_s=T.dt)
    CM.save("R001", "gbvr_basic.json", b, provisional=False)
    CM.save("R001", "gbvr_dev_detail.json", d, provisional=False)
    p = CM.save("R001", "summary.json", summ, provisional=False)
    print(p)
    print({k: v for k, v in summ.items() if k != "dev"})


if __name__ == "__main__":
    main()
