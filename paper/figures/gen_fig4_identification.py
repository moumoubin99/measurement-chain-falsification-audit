"""Fig. 4: fixation holds do not identify the filter.
(a) R005 analytic construction; (b) benchmark PV ambiguity; (c) retention vs settling; (d) post hoc window offsets."""
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from paper_plot_style import DATA, DOUBLE, OI, panel_label, save

d = json.load(open(os.path.join(DATA, "derived.json")))
rng = np.random.default_rng(2)
TRUTH_LABEL = {"light": "light", "medium": "medium", "heavy": "heavy", "b2": "B2",
               "vel_adaptive": "vel.-\nadaptive", "kalman_rts": "Kalman\nRTS"}
TRUTH_COL = {"light": OI["sky"], "b2": OI["green"], "medium": OI["blue"], "heavy": OI["verm"],
             "vel_adaptive": OI["grey"], "kalman_rts": OI["grey"]}

fig = plt.figure(figsize=(DOUBLE, 4.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.55, wspace=0.45)
ax = fig.add_subplot(gs[0, 0])
bx = fig.add_subplot(gs[1, 0])
gs_r = gs[:, 1].subgridspec(2, 1, hspace=0.32)
cx = fig.add_subplot(gs_r[0])
dx = fig.add_subplot(gs_r[1], sharex=cx)

# (a) R005
r5 = d["R005"]
order = ["B1_id_post_a1", "B1_bw2_60_post_a1", "B1_bw4_30_post_a1", "B1_ma5_post_a1", "B1_bw4_15_post_a1"]
lab = {"B1_id_post_a1": "identity", "B1_bw2_60_post_a1": "BW2, 60 Hz",
       "B1_bw4_30_post_a1": "BW4, 30 Hz", "B1_ma5_post_a1": "MA, 5 samples",
       "B1_bw4_15_post_a1": "BW4, 15 Hz"}
ys = np.arange(len(order))[::-1]
core = r5["core"]
for y, k in zip(ys, order):
    m = r5["members"][k]
    is_core = k in core
    ax.scatter(m["pv10"], y, s=28, marker="o", facecolor=OI["blue"] if is_core else "none",
               edgecolor=OI["blue"] if is_core else OI["grey"], lw=0.9, zorder=3)
    dev = m["dev"]
    if dev == 0:
        devs = "0"
    elif dev < 0.01:
        mant, ex = f"{dev:.1e}".split("e")
        devs = rf"{mant}$\times10^{{{int(ex)}}}$"
    else:
        devs = f"{dev:.2f}"
    ax.text(m["pv10"] + 0.012, y, "dev. " + devs, va="center", ha="left", fontsize=6.3)
ax.set_yticks(ys)
ax.set_yticklabels([lab[k] + ("" if k in core else " (not matched)") for k in order], fontsize=6.3)
ax.set_xlim(0.64, 1.04)
ax.set_xlabel(r"detected PV@10$^\circ$ / latent (analytic) PV")
pvs = [r5["members"][k]["pv10"] for k in core]
ax.annotate("", xy=(min(pvs), 4.55), xytext=(max(pvs), 4.55),
            arrowprops=dict(arrowstyle="<->", lw=0.6, color="k"))
ax.text(np.mean([min(pvs), max(pvs)]), 4.62, f"{100 * r5['pv_spread']:.1f}% spread (max/min − 1)",
        ha="center", va="bottom", fontsize=6.3)
ax.set_ylim(-0.6, 5.4)
ax.set_title("Analytic construction: near-identical holds", loc="left", fontsize=8)
panel_label(ax, "a", x=-0.36)

# (b) ambiguity
truths = ["light", "b2", "medium", "heavy", "vel_adaptive", "kalman_rts"]
for i, t in enumerate(truths):
    a = np.array(d["bench"][t]["ambiguity"])
    bx.scatter(i + rng.uniform(-0.17, 0.17, len(a)), a, s=6, color=TRUTH_COL[t], alpha=0.6, lw=0, zorder=2)
    q = d["bench"][t]["ambiguity_q"]
    bx.plot([i, i], [q[0], q[2]], color="k", lw=1.2, zorder=3)
    bx.scatter(i, q[1], marker="_", s=120, color="k", lw=1.6, zorder=4)
w = math.log(1.05 / 0.95)
bx.axhline(w, color="k", ls="--", lw=0.8)
bx.text(3.62, w - 0.008, "full equivalence width\n" + r"$\ln(1.05/0.95) \approx$ " + f"{w:.3f}", ha="left", va="top", fontsize=6.3, bbox=dict(fc="white", ec="none", pad=0.5))
bx.set_xticks(range(len(truths)))
bx.set_xticklabels([TRUTH_LABEL[t] for t in truths], fontsize=6.0)
bx.axvline(3.5, color="#BBBBBB", lw=0.6)
bx.text(1.5, 0.405, "in family", ha="center", fontsize=6.3, color="#555555")
bx.text(4.5, 0.405, "out of family", ha="center", fontsize=6.3, color="#555555")
bx.set_ylim(0, 0.43)
bx.set_ylabel("PV ambiguity across $A_r$\n(max − min log PV, log units)")
bx.set_title("Benchmark: residual PV ambiguity per replicate", loc="left", fontsize=8)
panel_label(bx, "b", x=-0.2)

# (c) retention vs settling ; (d) window offset vs settling
inf = ["light", "b2", "medium", "heavy"]
for t in inf:
    b = d["bench"][t]
    s = b["settling_ms"]
    r = b["retention_given_nonempty"]
    lo, hi = b["retention_given_nonempty_wilson"]
    cx.errorbar(s, r, yerr=[[max(0.0, r - lo)], [max(0.0, hi - r)]], fmt="o", color=TRUTH_COL[t], ms=4, capsize=2, lw=0.8)
    off = {"light": (4, 0.035, "right"), "b2": (-4, 0.035, "left"), "medium": (4, 0.0, "left"), "heavy": (4, 0.0, "left")}[t]
    cx.text(s + off[0], r + off[1], f"{TRUTH_LABEL[t]}: {b['retained']}/{b['n_nonempty']}", fontsize=6.3, va="bottom" if off[1] else "center", ha=off[2])
    wv = d["window"]["truths"][t]
    dx.scatter(s, wv["offset_D"], color=TRUTH_COL[t], s=20, zorder=3)
    dx.scatter(s, wv["offset_C"], facecolor="white", edgecolor=TRUTH_COL[t], s=20, lw=1.0, zorder=3)
    dx.plot([s, s], [wv["offset_D"], wv["offset_C"]], color=TRUTH_COL[t], lw=0.6, zorder=2)
    dx.text(s + 3, wv["offset_C"], f"{TRUTH_LABEL[t]}", fontsize=6.3, va="center", ha="left")
cx.set_ylim(0.3, 1.12)
cx.set_ylabel("retention of generating chain\ngiven $A_r \\neq \\varnothing$")
cx.tick_params(labelbottom=False)
cx.set_title("Benchmark: retention vs. filter settling time", loc="left", fontsize=8)
panel_label(cx, "c", x=-0.24)
cq = d["window"]["c10_D_median"]
dx.axhline(cq, color="k", ls="--", lw=0.8)
dx.text(58, cq + 0.08, f"median $c_{{10}}$ = {cq:.2f}", fontsize=6.3, va="bottom", ha="left")
dx.scatter([], [], color="#555555", s=20, label="D units ($s_D$)")
dx.scatter([], [], facecolor="white", edgecolor="#555555", s=20, lw=1.0, label="C units ($s_C$)")
dx.legend(loc="upper left", fontsize=6.3, frameon=False, handletextpad=0.3, borderaxespad=0.2)
dx.set_ylim(0, 4.9)
dx.set_xlim(0, 135)
dx.set_xlabel("2% step-settling time of generating filter (ms)")
dx.set_ylabel("window-placement offset\n(max $|\\Delta_k|/s_k$, coords 5–10)")
dx.text(24, 0.25, "post hoc; confounded with\nwindow count (22 vs 40)",
        fontsize=6.3, va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", lw=0.5))
dx.set_title("Window placement vs. filter settling time (post hoc)", loc="left", fontsize=8)
panel_label(dx, "d", x=-0.24)
save(fig, "fig4_identification")
