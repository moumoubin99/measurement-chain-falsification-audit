"""Post-hoc profile of the R004 calibration failure (A = empty): per-coordinate standardized hold
discrepancies z = D_hat / s for every member, the binding coordinate, and per-branch Lambda/c.
Descriptive only (reads R004/calibration.json; no data access)."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results")
NAMES = ["med_lH", "med_lV", "iqr_lH", "iqr_lV", "bp_4_10", "bp_10_30", "bp_30_60", "bp_60_125",
         "ac1", "ac2", "gap_frac_median", "gap_loglen_mean"]   # holds.coords_from_Q #11-12
CLEAN = list(range(10))   # fixation-window coordinates only (no missingness coordinates)
cal = json.loads((RES / "R004" / "calibration.json").read_text())
ms = [m for m in cal["members"] if m.get("estimable", True)]
Z = np.zeros((len(ms), 12))
for i, m in enumerate(ms):
    D, s = np.asarray(m["D_hat"], float), np.asarray(m["s"], float)
    keep = np.array([k not in m["dropped"] for k in range(12)]) & (s > 0)
    Z[i, keep] = D[keep] / s[keep]
cs = np.array([m["c"] for m in ms])
ratio = np.array([m["Lambda"] / m["c"] for m in ms])
ratio_clean = np.abs(Z[:, CLEAN]).max(1) / cs   # joint (same-member) max over coords 0-9, not a marginal min
bind = np.abs(Z).argmax(1)


def branch(name):
    p = name.split("_")
    filt = "identity" if p[1] == "id" else re.sub(r"\d+$", "", p[1])
    return dict(family=p[0], filter=filt, alpha=p[-1])


groups = defaultdict(list)
for i, m in enumerate(ms):
    b = branch(m["name"])
    for key in ("family", "filter", "alpha"):
        groups[f"{key}={b[key]}"].append(i)


def drop_test(drop):
    keep = [k for k in range(12) if k not in drop]
    return int((np.abs(Z[:, keep]).max(1) <= cs).sum())


def clean_row(i):
    return dict(name=ms[i]["name"], ratio_clean=round(float(ratio_clean[i]), 3),
                binding=NAMES[int(np.abs(Z[i, CLEAN]).argmax())],
                z={NAMES[k]: round(float(Z[i, k]), 2) for k in CLEAN})


alphas = sorted({branch(m["name"])["alpha"] for m in ms}, key=lambda a: float(a[1:]))
clean_profile = dict(
    n_admissible=int((ratio_clean <= 1).sum()),
    ratio_min=float(ratio_clean.min()),
    ratio_min_filtered=float(min(ratio_clean[i] for i, m in enumerate(ms) if branch(m["name"])["filter"] != "identity")),
    best_members=[clean_row(i) for i in np.argsort(ratio_clean)[:6]],
    best_by_alpha={a: clean_row(min((i for i, m in enumerate(ms) if branch(m["name"])["alpha"] == a),
                                    key=lambda i: ratio_clean[i])) for a in alphas},
    binding_counts={NAMES[k]: int(c) for k, c in zip(*np.unique(np.abs(Z[:, CLEAN]).argmax(1), return_counts=True))},
)


out = dict(
    n_members=len(ms), A_size=int(sum(m["admissible"] for m in cal["members"])),
    lambda_over_c=dict(min=float(ratio.min()), median=float(np.median(ratio)), max=float(ratio.max())),
    best_members=[dict(name=ms[i]["name"], Lambda=ms[i]["Lambda"], c=ms[i]["c"], eta=ms[i]["eta"],
                       z={NAMES[k]: round(float(Z[i, k]), 2) for k in range(12)})
                  for i in np.argsort(ratio)[:6]],
    binding_coordinate_counts={NAMES[k]: int((bind == k).sum()) for k in sorted(set(bind))},
    z_median_over_members={NAMES[k]: float(np.median(Z[:, k])) for k in range(12)},
    z_abs_min_over_members_MARGINAL={NAMES[k]: float(np.abs(Z[:, k]).min()) for k in range(12)},
    per_branch={g: dict(n=len(ix), ratio_min=float(ratio[ix].min()), ratio_median=float(np.median(ratio[ix])))
                for g, ix in sorted(groups.items())},
    clean_window_profile=clean_profile,
    exploratory_drop_tests={"+".join(NAMES[k] for k in d): drop_test(d)
                            for d in ([10], [11], [10, 11], [4], [8], [4, 10], [8, 10], [4, 8, 10])},
    note=("Post-hoc and descriptive. The pre-registered hold set is unchanged; drop tests are not a re-calibration. "
          "z_abs_min_over_members_MARGINAL takes the minimum over DIFFERENT members per coordinate and says nothing "
          "about joint admissibility; clean_window_profile is the joint same-member profile over coordinates 0-9."),
)
(RES / "R004" / "failure_profile.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("A_size", "lambda_over_c", "binding_coordinate_counts",
                                      "clean_window_profile", "exploratory_drop_tests")}, indent=1))
print(json.dumps(out["per_branch"], indent=1))
