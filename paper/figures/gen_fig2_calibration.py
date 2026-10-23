"""Fig. 2: Stage-1 calibration failure. (a) D split, 276 members; (b) confirmatory-split holds."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from paper_plot_style import (DATA, DOUBLE, FILTER_COLORS, FILTER_LABELS, BAND_COLOR, OI,
                              panel_label, save, parse_member)

d = json.load(open(os.path.join(DATA, "derived.json")))
ff = d["bench_false_failures"]
ALPHAS = [0.0, 0.5, 1.0, 1.5]
rng = np.random.default_rng(0)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(DOUBLE, 3.2), gridspec_kw=dict(width_ratios=[3.0, 1.15]),
                             sharey=True)
for a in (ax, bx):
    a.axhspan(ff["min_in"], ff["max_in"], color=BAND_COLOR, zorder=0, lw=0)
    a.axhline(1.0, color=OI["black"], ls="--", lw=0.8, zorder=1)
    a.set_yscale("log")

# (a) D split
for i, al in enumerate(ALPHAS):
    ms = [m for m in d["D_members"] if parse_member(m["name"])["alpha"] == al]
    for m in ms:
        p = parse_member(m["name"])
        x = i + (-0.2 if p["loc"] == "post" else 0.2) + rng.uniform(-0.14, 0.14)
        mk = "o" if p["branch"] == "B1" else "^"
        ax.scatter(x, m["ratio"], s=9, marker=mk, facecolor="none" if p["loc"] == "pre" else FILTER_COLORS[p["ftype"]],
                   edgecolor=FILTER_COLORS[p["ftype"]], lw=0.5, alpha=0.85, zorder=3)
    mn = min(m["ratio"] for m in ms)
    ax.text(i, 27, f"min {mn:.2f}\n($n$ = {len(ms)})", ha="center", va="bottom", fontsize=6.5)
    ax.text(i - 0.2, 2.2, "post", ha="center", va="top", fontsize=6, color="#555555")
    ax.text(i + 0.2, 2.2, "pre", ha="center", va="top", fontsize=6, color="#555555")
ax.set_xticks(range(4))
ax.set_xticklabels([r"$\alpha$ = 0", r"$\alpha$ = 0.5", r"$\alpha$ = 1", r"$\alpha$ = 1.5"])
ax.set_xlim(-0.6, 3.6)
ax.set_ylim(0.8, 42)
for a in (ax, bx):
    a.set_yticks([1, 2, 5, 10, 20])
    a.set_yticklabels(["1", "2", "5", "10", "20"])
    a.minorticks_off()
ax.set_ylabel(r"$\Lambda / c$ (log scale)")
ax.set_xlabel("noise exponent (post- / pre-filter noise placement)", labelpad=1)
ax.set_title(f"Development split: 0/{d['D_summary']['n']} admissible", loc="left", fontsize=8)
ax.text(-0.55, (ff["min_in"] * ff["max_in"]) ** 0.5, f"benchmark false failures, in-family truths ({ff['n_in']} replicates): {ff['min_in']:.2f}–{ff['max_in']:.2f} (observed range)",
        ha="left", va="center", fontsize=6.5, color="#444444")
ax.text(-0.55, 0.97, r"$\Lambda / c$ = 1 (admission threshold)", ha="left", va="top", fontsize=6.5)
handles = [Line2D([], [], ls="", marker="o", color=FILTER_COLORS[k], ms=4, label=FILTER_LABELS[k]) for k in ("id", "bw", "ma")]
handles += [Line2D([], [], ls="", marker="o", mfc=OI["grey"], mec=OI["grey"], ms=4, label="B1 (250 Hz filter)"),
            Line2D([], [], ls="", marker="^", mfc=OI["grey"], mec=OI["grey"], ms=4, label="B2 (1000 Hz filter)"),
            Line2D([], [], ls="", marker="o", mfc="none", mec=OI["grey"], ms=4, label="noise before filter")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=6.5,
          handletextpad=0.2, columnspacing=0.9)
panel_label(ax, "a", x=-0.09)

# (b) confirmatory holds
r203 = [m["ratio"] for m in d["C_R203"]]
a006 = [m["full"] for m in d["C_A006"]]
bx.scatter(np.zeros(len(r203)) + rng.uniform(-0.15, 0.15, len(r203)), r203, s=12, marker="s",
           color=OI["blue"], zorder=3)
bx.scatter(np.ones(len(a006)) + rng.uniform(-0.15, 0.15, len(a006)), a006, s=12, marker="D",
           facecolor="none", edgecolor=OI["verm"], lw=0.7, zorder=3)
bx.set_xticks([0, 1])
bx.set_xticklabels([f"lock-spec.\n$n$ = {len(r203)}", f"post hoc\n$n$ = {len(a006)}"], fontsize=6.5)
bx.set_xlim(-0.6, 1.6)
bx.text(0, max(r203) * 1.15, f"{min(r203):.2f}–{max(r203):.2f}", ha="center", fontsize=6.5)
bx.text(1, max(a006) * 1.15, f"{min(a006):.2f}–{max(a006):.2f}", ha="center", fontsize=6.5)
bx.set_title("Confirmatory-split holds", loc="left", fontsize=8)
panel_label(bx, "b", x=-0.2)
fig.tight_layout(w_pad=0.6)
save(fig, "fig2_calibration")
print("D min", d["D_summary"]["min"], "R203", min(r203), max(r203), "A006", min(a006), max(a006))
