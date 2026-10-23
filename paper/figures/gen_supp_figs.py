"""Supplementary Figs. S1 (hold z-profiles over the family) and S2 (R005 analytic construction, per member)."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

def _sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"

from paper_plot_style import DOUBLE, OI, RES, panel_label, save

fp = json.load(open(os.path.join(RES, "R004", "failure_profile.json")))
r5 = json.load(open(os.path.join(RES, "R005", "existence.json")))
dadm = json.load(open(os.path.join(RES, "A006", "d_adm.json")))
NAMES = ["med_lH", "med_lV", "iqr_lH", "iqr_lV", "bp_4_10", "bp_10_30", "bp_30_60", "bp_60_125", "ac1", "ac2",
         "gap_frac_median", "gap_loglen_mean"]
LAB = ["med H", "med V", "IQR H", "IQR V", "4–10 Hz", "10–30 Hz", "30–60 Hz", "60–125 Hz", "lag-1 AC", "lag-2 AC",
       "gap frac.", "gap log-len."]
x = np.arange(12)


def blocks(ax, ytxt):
    for lo, hi, t in [(-0.5, 3.5, "scale"), (3.5, 9.5, "spectral shape"), (9.5, 11.5, "missingness")]:
        ax.axvspan(lo, hi, color="#F2F2F2" if t == "spectral shape" else "white", zorder=0, lw=0)
        ax.text((lo + hi) / 2, ytxt, t, ha="center", va="bottom", fontsize=6.3, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(LAB, rotation=45, ha="right", fontsize=6.3)
    ax.set_xlim(-0.5, 11.5)


# ---------------- S1 ----------------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), gridspec_kw=dict(wspace=0.3))
cols = [OI["blue"], OI["verm"], OI["green"], OI["purple"], OI["orange"], OI["sky"]]
for m, c in zip(fp["best_members"], cols):
    z = [m["z"][n] for n in NAMES]
    ax.plot(x, z, "-o", color=c, ms=2.5, lw=0.8, label=f"{m['name']} ({m['Lambda'] / m['c']:.2f})")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("standardised hold discrepancy $z$")
blocks(ax, 9.6)
ax.set_ylim(-6, 10.8)
ax.legend(fontsize=5.4, loc="upper left", bbox_to_anchor=(0.0, 0.9), ncol=1, handlelength=1.2,
          title=r"six lowest $\Lambda/c$ on D", title_fontsize=5.8)
ax.set_title("Best members: profile across the 12 coordinates", loc="left", fontsize=8)
panel_label(ax, "a", x=-0.13)

# median of |z| over members (A006 reproduces R004 exactly); not |median of signed z|
zmed = dict(zip(dadm["summary"]["z_names"], dadm["summary"]["z_median_abs"]))
med = [zmed[n] for n in NAMES]
mn = [fp["z_abs_min_over_members_MARGINAL"][n] for n in NAMES]
bx.bar(x - 0.2, med, 0.38, color=OI["blue"], label="median $|z|$ over 276 members")
bx.bar(x + 0.2, mn, 0.38, color=OI["verm"], label="minimum $|z|$ over members (per coordinate)")
bx.set_ylabel("$|z|$")
blocks(bx, 12.4)
bx.set_ylim(0, 13.6)
bx.legend(fontsize=5.6, loc="upper left", bbox_to_anchor=(0.0, 0.9))
bx.set_title("Family-wide: which coordinates no member can match", loc="left", fontsize=8)
panel_label(bx, "b", x=-0.1)
save(fig, "figS1_zprofiles")

# ---------------- S2 ----------------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.32))
mem = list(r5["members"])
MCOL = {mem[0]: OI["black"], mem[1]: OI["blue"], mem[2]: OI["green"], mem[3]: OI["verm"], mem[4]: OI["purple"]}
MLAB = {n: n.replace("_post_a1", "").replace("B1_", "").replace("id", "identity").replace("bw", "BW").replace("ma", "MA")
        for n in mem}
for i, n in enumerate(mem):
    e = r5["members"][n]
    core = n in r5["core_members"]
    ax.plot(np.arange(10) + (i - 2) * 0.06, e["coords_1_10"], "o" if core else "x", ms=3, color=MCOL[n],
            label=f"{MLAB[n]} ({e['step_settling_ms']:.0f} ms" + (")" if core else "; boundary)"))
ax.set_xticks(np.arange(10))
ax.set_xticklabels(LAB[:10], rotation=45, ha="right", fontsize=6.3)
ax.set_ylabel("hold coordinate value")
ax.legend(fontsize=5.4, loc="center", bbox_to_anchor=(0.42, 0.52), handlelength=1, title="filter (step settling)",
          title_fontsize=5.6)
ax.set_title(f"Hold coordinates 1–10 (core spread {_sci(r5['max_rel_coord_spread_1_10'])})", loc="left", fontsize=8)
panel_label(ax, "a", x=-0.14)

amps = [5, 10, 15]
for i, n in enumerate(mem):
    e = r5["members"][n]
    core = n in r5["core_members"]
    bx.plot(amps, [e[f"pv_det_ratio_{a}"] for a in amps], "-o" if core else "--x", ms=3, lw=0.8, color=MCOL[n])
bx.set_xticks(amps)
bx.set_xlabel("saccade amplitude (deg)")
bx.set_ylabel("detected PV / latent PV")
bx.set_title(f"PV differs ({100 * r5['pv10_detected_max_over_min_minus1']:.1f}% at 10° across core)", loc="left", fontsize=8)
panel_label(bx, "b", x=-0.2)
save(fig, "figS2_r005")
print("ok")
