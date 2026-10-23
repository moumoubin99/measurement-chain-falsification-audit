"""Fig. 5: descriptive anchors under the Stage-1 stop. (a) native PV@10 deg (D split); (b) within-GazeBase ladder."""
import json
import os
import matplotlib.pyplot as plt
from paper_plot_style import DATA, DOUBLE, OI, panel_label, save

d = json.load(open(os.path.join(DATA, "derived.json")))
pv = d["pv10"]
nr = d["native_ratio"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.8))

# (a)
spec = [("spline", "native_gb_spline", "native_gbvr_spline"), ("linear 8–12°", "native_gb_linear_8_12", "native_gbvr_linear_8_12")]
for j, (lab, kg, kv) in enumerate(spec):
    for k, key, col, dxo, name in [(0, kg, OI["blue"], -0.12, "GazeBase"), (1, kv, OI["verm"], 0.12, "GazeBaseVR")]:
        e = pv[key]
        ax.errorbar(j + dxo, e["pv_deg_s"], yerr=[[e["pv_deg_s"] - e["ci95_deg_s"][0]], [e["ci95_deg_s"][1] - e["pv_deg_s"]]],
                    fmt="o" if k == 0 else "s", color=col, ms=4, capsize=2, lw=0.8,
                    label=name if j == 0 else None)
        ax.text(j + dxo + (0.05 if k else -0.05), e["pv_deg_s"], f"{e['pv_deg_s']:.1f}\n({e['n_elig']}/{e['P']})",
                ha="left" if k else "right", va="center", fontsize=6)
    r = nr["spline" if j == 0 else "linear"]
    ax.text(j, 392, f"ratio {r['ratio_vr_over_gb']:.3f}\n90% band\n[{r['band'][0]:.3f}, {r['band'][1]:.3f}]",
            ha="center", va="bottom", fontsize=6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["spline", "linear 8–12°"])
ax.set_xlim(-0.75, 1.6)
ax.set_ylim(305, 425)
ax.set_ylabel(r"native PV@10$^\circ$ (deg s$^{-1}$), 95% CI")
ax.legend(loc="lower center", ncol=2, fontsize=6.5, handletextpad=0.2)
ax.set_title("Native peak velocity (D split)", loc="left", fontsize=8)
panel_label(ax, "a", x=-0.3)

# (b) ladder A
lad = d["ladder"]["A"]
names = ["native GazeBase", "+ 250 Hz sampling,\n4 ms grid", "+ re-detection", "+ noise", "+ filter\n(BW4, 30 Hz)"]
ys = list(range(len(lad)))[::-1]
for y, r, nm in zip(ys, lad, names):
    col = OI["grey"] if r["key"] == "r0" else OI["blue"]
    bx.errorbar(r["ratio"], y, xerr=[[r["ratio"] - r["ratio_lo"]], [r["ratio_hi"] - r["ratio"]]], fmt="o", color=col, ms=4, capsize=2, lw=0.8)
    if r["key"] != "r0":
        bx.text(r["ratio_hi"] + 0.003, y, f"{r['ratio']:.3f}", va="center", fontsize=6.3)
bx.axvline(nr["spline"]["ratio_vr_over_gb"], color=OI["verm"], ls=":", lw=0.9)
bx.text(nr["spline"]["ratio_vr_over_gb"] + 0.003, 4.35, "native GazeBaseVR /\nGazeBase ratio (0.902)", fontsize=6, color=OI["verm"], va="top")
bx.set_yticks(ys)
bx.set_yticklabels(names, fontsize=6.3)
bx.set_xlim(0.89, 1.015)
bx.set_xlabel(r"PV@10$^\circ$ ratio to native GazeBase" + "\n(95% CI)")
bx.text(0.892, -0.75, "illustrates measurement sensitivity;\nnot a decomposition of the native gap ($A = \\varnothing$)",
        fontsize=6.0, style="italic", va="center")
bx.set_ylim(-1.1, 4.5)
bx.set_title("Within-GazeBase degradation ladder", loc="left", fontsize=8)
panel_label(bx, "b", x=-0.42)
save(fig, "fig5_native")
