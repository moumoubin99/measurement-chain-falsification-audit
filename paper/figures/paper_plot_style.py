"""Shared style for all data figures (Scientific Reports; serif, colourblind-safe)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "experiments", "results")
DATA = os.path.join(HERE, "data")

FONT_SIZE = 8
matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
})

# Okabe-Ito palette (colourblind-safe)
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
      "grey": "#999999"}
FILTER_COLORS = {"id": OI["black"], "bw": OI["blue"], "ma": OI["orange"]}
FILTER_LABELS = {"id": "identity", "bw": "Butterworth", "ma": "moving average"}
BAND_COLOR = "#E5E5E5"

SINGLE = 3.46   # in (88 mm)
DOUBLE = 7.08   # in (180 mm)


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=FONT_SIZE + 2, fontweight="bold",
            va="bottom", ha="left")


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    fig.savefig(os.path.join(HERE, name + ".png"))
    plt.close(fig)


def parse_member(name):
    """'B1_bw4_15_post_a0.5' -> dict(branch, ftype, fpar, loc, alpha)."""
    parts = name.split("_")
    branch = parts[0]
    alpha = float(parts[-1][1:])
    loc = parts[-2]
    f = "_".join(parts[1:-2])
    if f == "id":
        ftype = "id"
    elif f.startswith("bw"):
        ftype = "bw"
    else:
        ftype = "ma"
    return dict(branch=branch, ftype=ftype, f=f, loc=loc, alpha=alpha)
