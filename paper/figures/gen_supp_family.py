"""Supplementary Table S5: enumeration of the declared 276-member family with 2% step-settling times,
generated from experiments/kincost/chain.py (manifest and settling computation; no numbers typed by hand)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments"))
from kincost import chain as C  # noqa: E402

ms = C.manifest()
e_set, worst = C.settling_exclusion_ms()


def flabel(ft, fp):
    if ft == "id":
        return "Identity"
    if ft == "butter":
        return f"Butterworth order {fp[0]}, {fp[1]} Hz"
    return f"Moving average, {fp[0]} samples"


rows = {}
for m in ms:
    fs = 250.0 if m.branch == "B1" else 1000.0
    key = (m.branch, m.ftype, m.fpar)
    r = rows.setdefault(key, dict(n=0, loc=set(), st=C.step_settling_ms(m.ftype, m.fpar, fs)))
    r["n"] += 1
    r["loc"].add(m.noise_loc)
order = {"id": 0, "butter": 1, "ma": 2}
body = ""
for br in ("B1", "B2"):
    ks = sorted([k for k in rows if k[0] == br], key=lambda k: (order[k[1]], k[2]))
    n_br = sum(rows[k]["n"] for k in ks)
    body += r"\multicolumn{4}{@{}l}{\textit{Branch " + br + (" (filtered at 250\\,Hz)" if br == "B1" else " (filtered at 1000\\,Hz)") + f", {n_br} members" + r"}} \\" + "\n"
    for k in ks:
        r = rows[k]
        loc = " and ".join(sorted(r["loc"], key=lambda s: s != "pre"))
        body += f"\\quad {flabel(k[1], k[2])} & {loc} & {r['n']} & {r['st']:.0f} \\\\\n"
s = r"""\begin{table}[ht]
\centering
\caption{\textbf{Enumeration of the declared measurement-chain family.} Every filter is combined with the listed noise placement(s) and with the four noise exponents $\alpha \in \{0, 0.5, 1, 1.5\}$, giving """ + str(len(ms)) + r""" members. Settling time is the 2\% step-response settling time of the causal filter at its operating rate; the slowest is """ + f"{worst:.0f}" + r"""\,ms, which sets the settling exclusion after valid-run starts to """ + f"{e_set:.0f}" + r"""\,ms (the larger of 100\,ms and the slowest settling time, rounded up to the 4\,ms grid). Generated from \texttt{kincost/chain.py}.}
\label{tab:s5}
\footnotesize
\begin{tabular}{@{}llrr@{}}
\toprule
Filter & Noise placement & Members & Settling (ms) \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(HERE, "TABLE_S5_family.tex"), "w").write(s)
print("ok", len(ms), e_set, worst)
