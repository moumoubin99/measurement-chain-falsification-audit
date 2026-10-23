"""Fig. 3 (post hoc diagnostics): (a) missingness ladder, (b) clean-window recalibration, (c) z-profiles."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paper_plot_style import DATA, DOUBLE, OI, BAND_COLOR, FILTER_COLORS, panel_label, save, parse_member

d = json.load(open(os.path.join(DATA, "derived.json")))
mi = d["missingness"]
ff = d["bench_false_failures"]
rng = np.random.default_rng(1)

fig = plt.figure(figsize=(DOUBLE, 5.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1, 1.05], hspace=0.62, wspace=0.42)
ax = fig.add_subplot(gs[0, 0])
cx = fig.add_subplot(gs[0, 1])
bx = fig.add_subplot(gs[1, :])

# (a) missingness ladder (median over participants, %)
rows = [("GazeBase native (raw)", mi["gb_raw"], OI["grey"]),
        ("+ donor timing, 4 ms grid", mi["gb_timing_R4"], OI["grey"]),
        ("+ pooled renewal (re-simulated)", mi["gb_timing_R4_renewal"], OI["blue"]),
        ("calibration point estimate", mi["calibration_point"], OI["sky"]),
        ("gap-free source + renewal", mi["gapfree_renewal"], OI["orange"]),
        ("GazeBaseVR target", mi["gbvr"], OI["verm"])]
y = np.arange(len(rows))[::-1]
for yi, (lab, v, col) in zip(y, rows):
    ax.barh(yi, 100 * v, color=col, height=0.62)
    ax.text(100 * v + 0.04, yi, f"{100 * v:.2f}%", va="center", fontsize=6.5)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
ax.set_xlabel("median gap fraction per participant (%)")
ax.set_xlim(0, 2.6)
ax.axvline(100 * mi["gbvr"], color=OI["verm"], lw=0.6, ls=":")
ax.set_title("Missingness (development split)", loc="left", fontsize=8)
panel_label(ax, "a", x=-0.75)

# (c) clean-window recalibration
for xi, (lab, ms) in enumerate([("D", d["D_A006"]), ("C", d["C_A006"])]):
    for m in ms:
        filt = m["filtered"]
        xx = xi + rng.uniform(-0.18, 0.18)
        cx.scatter(xx, m["clean10"], s=9, marker="o", facecolor=OI["blue"] if filt else "none",
                   edgecolor=OI["blue"] if filt else OI["black"], lw=0.6, alpha=0.8, zorder=3)
cx.axhspan(ff["min_in"], ff["max_in"], color=BAND_COLOR, zorder=0, lw=0)
cx.axhline(1.0, color="k", ls="--", lw=0.8)
cx.set_yscale("log")
cx.set_yticks([1, 2, 5, 10])
cx.set_yticklabels(["1", "2", "5", "10"])
cx.minorticks_off()
cx.set_xticks([0, 1])
cx.set_xticklabels([f"development\n(all {len(d['D_A006'])} members)", f"confirmatory\n({len(d['C_A006'])} development-selected)"],
                   fontsize=6.5)
cx.set_xlim(-0.5, 1.5)
cx.set_ylabel(r"$\Lambda_{10} / c_{10}$")
sD, sC = d["A006_summary"]["D"]["clean10"], d["A006_summary"]["C"]["clean10"]
cx.text(0.25, sD["ratio_min"] * 1.02, f"min {sD['ratio_min']:.2f}\n(filtered {sD['ratio_min_filtered']:.2f})", fontsize=6, va="center")
cx.text(1.22, sC["ratio_min"], f"{sC['ratio_min']:.2f}", fontsize=6, va="center")
cx.text(1.22, sC["ratio_min_filtered"] * 1.03, f"≥ {sC['ratio_min_filtered']:.2f}", fontsize=6, va="center")
cx.legend(handles=[Line2D([], [], ls="", marker="o", mfc=OI["blue"], mec=OI["blue"], ms=4, label="filtered chain"),
                   Line2D([], [], ls="", marker="o", mfc="none", mec="k", ms=4, label="identity (no filter)")],
          loc="upper right", fontsize=6.5, handletextpad=0.2)
cx.set_title("Clean-window test, coordinates 1–10", loc="left", fontsize=8)
panel_label(cx, "b", x=-0.3)

# (b) z-profile of the D-best member per noise exponent
names = d["z_names"]
labels = ["med H", "med V", "IQR H", "IQR V", "4–10 Hz", "10–30 Hz", "30–60 Hz", "60–125 Hz",
          "AC lag 1", "AC lag 2", "gap frac.", "log gap len."]
x = np.arange(len(names))
for lo, hi, lab in [(-0.5, 3.5, "1–4: scale (profiled by η)"), (3.5, 9.5, "5–10: clean-window spectral shape"),
                    (9.5, 11.5, "11–12: missingness")]:
    bx.axvspan(lo, hi, color="#F2F2F2" if lo != 3.5 else "#E8EEF6", lw=0, zorder=0)
    bx.text((lo + hi) / 2, 13.3, lab, ha="center", fontsize=6.5)
cq = d["window"]["D_c_quantiles"]
for s in (1, -1):
    bx.axhline(s * cq[1], color="k", ls="--", lw=0.6)
bx.axhline(0, color="k", lw=0.4)
bx.text(-0.4, cq[1] + 0.4, f"± median $c$ = {cq[1]:.2f}", ha="left", fontsize=6.5)
# post hoc window-placement sensitivity (A006/window.json): range over the four in-family truths, in D units
_wt = d["window"]["truths"].values()
for j in range(6):
    vals = [v["D_units"][j] for v in _wt] + [0.0]
    bx.fill_between([4 + j - 0.32, 4 + j + 0.32], min(vals), max(vals), color="#9A9A9A", alpha=0.55, lw=0, zorder=1,
                    label="window-placement offset range\n(in-family truths, D units)" if j == 0 else None)
cols = [OI["black"], OI["orange"], OI["blue"], OI["purple"]]
mks = ["o", "s", "^", "D"]
for (al, col, mk) in zip([0.0, 0.5, 1.0, 1.5], cols, mks):
    ms = [m for m in d["D_A006"] if parse_member(m["name"])["alpha"] == al]
    b = min(ms, key=lambda m: m["full"])
    bx.plot(x, b["z"], marker=mk, color=col, ms=3.5, lw=0.9,
            label=fr"$\alpha$ = {al:g}: {b['name']} ($\Lambda/c$ = {b['full']:.2f})")
bx.set_xticks(x)
bx.set_xticklabels(labels, fontsize=6.5, rotation=25, ha="right", rotation_mode="anchor")
bx.set_xlim(-0.5, 11.5)
bx.set_ylim(-7, 15.5)
bx.set_ylabel(r"standardised deviation $z_k$" + "\n(simulated − GazeBaseVR)")
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=6.5, ncol=2, handlelength=1.6, columnspacing=1.5)
bx.set_title("Hold-coordinate profile of the best member per noise exponent (development split)", loc="left", fontsize=8)
panel_label(bx, "c", x=-0.075)
save(fig, "fig3_diagnostics")
