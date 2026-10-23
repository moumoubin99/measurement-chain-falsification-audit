"""Supplementary Tables S1-S4, written as LaTeX from result files (no numbers typed by hand)."""
import json
import numpy as np
import os

from paper_plot_style import RES

OUT = os.path.dirname(os.path.abspath(__file__))


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def j(*p):
    return json.load(open(os.path.join(RES, *p)))


def w(name, s):
    open(os.path.join(OUT, name), "w").write(s)


def tex(s):
    return s.replace("_", r"\_")


fp = j("R004", "failure_profile.json")
dadm = j("A006", "d_adm.json")["summary"]
cadm = j("A006", "c_adm.json")["summary"]
_dm = j("A006", "d_adm.json")["members"]
_cids = {m["mid"] for m in j("A006", "c_adm.json")["members"]}
_rest = [m for m in _dm if m["mid"] not in _cids]


def rest_rows():
    import numpy as np
    out = ""
    for a in sorted({float(m["name"].split("_")[-1][1:]) for m in _rest}):
        g = [m for m in _rest if float(m["name"].split("_")[-1][1:]) == a]
        r = np.array([m["clean10"]["ratio"] for m in g])
        rf = [m["clean10"]["ratio"] for m in g if m["filtered"]]
        q = np.quantile(r, [.1, .5, .9])
        nadm = sum(m["clean10"]["admissible"] for m in g)
        out += (f"\\quad $\\alpha = {a:g}$ & {nadm}/{len(g)} & {r.min():.3f} & {min(rf):.3f} & "
                f"{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}] & --- & --- \\\\\n")
    return out
win = j("A006", "window.json")
bench = j("R102", "summary.json")["summary"]

# ---------------- S1: per-branch Lambda/c on D ----------------
pb = fp["per_branch"]
rows = [("Noise exponent", [("$\\alpha = 0$", "alpha=a0"), ("$\\alpha = 0.5$", "alpha=a0.5"),
                         ("$\\alpha = 1$", "alpha=a1"), ("$\\alpha = 1.5$", "alpha=a1.5")]),
        ("Filtering rate", [("B1 (filtered at 250 Hz)", "family=B1"), ("B2 (filtered at 1000 Hz)", "family=B2")]),
        ("Filter type", [("Identity", "filter=identity"), ("Butterworth", "filter=bw"), ("Moving average", "filter=ma")])]
body = ""
for head, items in rows:
    body += r"\multicolumn{4}{@{}l}{\textit{" + head + r"}} \\" + "\n"
    for lab, k in items:
        e = pb[k]
        body += f"\\quad {lab} & {e['n']} & {e['ratio_min']:.3f} & {e['ratio_median']:.3f} \\\\\n"
bc = fp["binding_coordinate_counts"]
bind = ", ".join(f"\\texttt{{{tex(k)}}} {v}" for k, v in sorted(bc.items(), key=lambda kv: -kv[1]))
lc = fp["lambda_over_c"]
s1 = r"""\begin{table}[ht]
\centering
\caption{\textbf{Hold admissibility of the declared family on the development split, by branch.} $\Lambda/c$ is the max-T statistic over the 12 hold coordinates (less those dropped by the zero-variability rule, Supplementary Note~S3) divided by its bootstrap critical value ($B = 2000$); a member is admissible iff $\Lambda/c \le 1$. No branch contains an admissible member. All 276 members: min """ + f"{lc['min']:.3f}, median {lc['median']:.3f}, max {lc['max']:.2f}" + r""". Coordinate attaining the maximum (counts over members): """ + bind + r""". Source: \texttt{R004/failure\_profile.json}.}
\label{tab:s1}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
Branch & Members & min $\Lambda/c$ & median $\Lambda/c$ \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{table}
"""
w("TABLE_S1_branches.tex", s1)

# ---------------- S2: full / clean-window / consistent-population ----------------
LAB = {"full": "Full 12-coordinate test (as prespecified)", "clean10": "Clean window, coordinates 1--10 ($c_{10}$)",
       "consistent": "Consistent bootstrap/point-estimate population"}


def adm_row(summ, key, n):
    e = summ[key]
    q = e["ratio_q"]
    bc = ", ".join(f"\\texttt{{{tex(k)}}} {v}" for k, v in sorted(e["binding_counts"].items(), key=lambda kv: -kv[1]))
    return (f"\\quad {LAB[key]} & {e['n_admissible']}/{n} & {e['ratio_min']:.3f} & {e['ratio_min_filtered']:.3f} & "
            f"{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}] & {e['c_q'][1]:.2f} & {bc} \\\\\n")


s2 = r"""\begin{table}[ht]
\centering
\caption{\textbf{Post hoc admissibility variants on the development (D) and confirmatory (C) splits.} ``Full'' repeats the prespecified test; ``clean window'' drops the two missingness coordinates and recalibrates its own critical value $c_{10}$; ``consistent population'' computes bootstrap and point estimate on the same eligible participants. C is assessed for the 15 development-selected candidates (the union of the 10 lock-specified R203 members and the 10 members with the lowest clean-window ratio on D). ``min (filtered)'' is the minimum over members with a non-identity filter. Quantiles are median [10th, 90th percentiles] over members; $c$ is the median critical value. Binding: coordinate attaining the maximum, counts over members. The last block gives, for the D members not assessed on C, the clean-window ratio by noise exponent. Sources: \texttt{A006/d\_adm.json}, \texttt{A006/c\_adm.json}. These are post hoc diagnostics, not decisions.}
\label{tab:s2}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.6cm}rrrrr>{\raggedright\arraybackslash}p{3.4cm}@{}}
\toprule
Variant & Admissible & min $\Lambda/c$ & min (filtered) & $\Lambda/c$ median [P10, P90] & $c$ & Binding coordinate \\
\midrule
\multicolumn{7}{@{}l}{\textit{Development split D, all """ + str(dadm["n"]) + r""" members}} \\
""" + "".join(adm_row(dadm, k, dadm["n"]) for k in ["full", "clean10", "consistent"]) + r"""\addlinespace
\multicolumn{7}{@{}l}{\textit{Confirmatory split C, """ + str(cadm["n"]) + r""" development-selected candidates}} \\
""" + "".join(adm_row(cadm, k, cadm["n"]) for k in ["full", "clean10", "consistent"]) + r"""\addlinespace
\multicolumn{7}{@{}l}{\textit{D split, clean window, the """ + str(len(_rest)) + r""" members not among the C candidates, by noise exponent}} \\
""" + rest_rows() + r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
w("TABLE_S2_variants.tex", s2)

# ---------------- S3: per-truth coverage and oracle details ----------------
T = ["light", "medium", "heavy", "b2", "vel_adaptive", "kalman_rts"]
HEAD = ["Light", "Medium", "Heavy", "B2", "Vel.-adaptive", "Kalman RTS"]
MEM = {t: win["truths"][t]["member"] for t in win["truths"]}


def ci2(v):
    return f"[{v[0]:.2f}, {v[1]:.2f}]"


def r(label, fn):
    return label + " & " + " & ".join(fn(bench[t], t) for t in T) + r" \\" + "\n"


s3 = r"""\begin{table}[ht]
\centering
\caption{\textbf{Benchmark details per generating truth} (26 replicates each; complements Table 3 of the main text). Oracle: the member-wise PV contrast $\delta^*$ estimated on all 100 participants from $K = 6$ independent batches, with Bonferroni $t$ intervals; it is resolved when every configuration of member contrasts inside the box of oracle intervals gives the same decision category (Supplementary Note~S4). Per-member band coverage pools all (replicate, member) pairs with $A_r \neq \varnothing$; the bracket is a replicate-bootstrap 95\% interval. ``Definite'' bounds count a member as covered only if the whole oracle interval lies inside the band (lower) or as uncovered only if the two are disjoint (upper). Simultaneous coverage asks whether the max-T band covers every retained member at once (nominal 0.90). Generating chains: """ + "; ".join(f"{h} \\texttt{{{tex(MEM[t])}}}" for t, h in zip(T, HEAD) if t in MEM) + r"""; the other two truths are out of family. Source: \texttt{R102/summary.json}.}
\label{tab:s3}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.6cm}""" + "c" * 6 + r"""@{}}
\toprule
 & """ + " & ".join(HEAD) + r""" \\
\midrule
""" + r("Oracle resolved, $k/26$", lambda b, t: str(b["oracle_status"]["resolved"])) \
    + r("Oracle unresolved / not run", lambda b, t: f"{b['oracle_status']['unresolved']} / {b['oracle_status']['not_run']}") \
    + r("Not applicable ($A_r = \\varnothing$), $k/26$", lambda b, t: str(b["N"] - b["oracle_status"]["resolved"] - b["oracle_status"]["unresolved"] - b["oracle_status"]["not_run"])) \
    + r("Support failure, $k/26$", lambda b, t: f"{round(b['support_failure'][0] * b['N'])}") \
    + r("Per-member band coverage (plug-in)", lambda b, t: f"{b['band_coverage_pooled']['plugin'][0]:.3f}") \
    + r(r"\quad 95\% bootstrap interval", lambda b, t: ci2(b["band_coverage_pooled"]["plugin"][1])) \
    + r(r"\quad definite lower / upper", lambda b, t: f"{b['band_coverage_pooled']['definite_lower'][0]:.2f} / {b['band_coverage_pooled']['definite_upper'][0]:.2f}") \
    + r("Simultaneous coverage (plug-in)", lambda b, t: f"{b['simultaneous_coverage_plugin']['rate']:.2f} ({b['simultaneous_coverage_plugin']['n_assessed']})") \
    + r(r"\quad definite bounds", lambda b, t: ci2(b["simultaneous_coverage_definite_bounds"]["bounds"])) \
    + r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
w("TABLE_S3_bench_details.tex", s3)

# ---------------- S4: window-placement offsets ----------------
NAMES5 = win["names"][4:10]
NL = [r"4--10 Hz", r"10--30 Hz", r"30--60 Hz", r"60--125 Hz", "lag-1 AC", "lag-2 AC"]
WT = ["light", "b2", "medium", "heavy"]
WL = {"light": "Light", "b2": "B2", "medium": "Medium", "heavy": "Heavy"}
_sC = np.median([m["s"][4:10] for m in j("A006", "c_adm.json")["members"]], axis=0)
_c10D = float(np.median([m["clean10"]["c"] for m in j("A006", "d_adm.json")["members"]]))
_c10C = float(np.median([m["clean10"]["c"] for m in j("A006", "c_adm.json")["members"]]))
SCALES = [("benchmark scale ($s_{\\mathrm{bench}}$)", lambda e: e["D_refit_over_s_bench_5_10"]),
          ("development scale ($s_D$, member's own R004 standard errors)", lambda e: e["D_refit_over_sD"][4:10]),
          ("confirmatory scale ($s_C$, median over the 15 C candidates)", lambda e: list(np.array(e["D_refit"][4:10]) / _sC))]
body = ""
for si, (lab, f) in enumerate(SCALES):
    body += (r"\midrule" + "\n" if si else "") + r"\multicolumn{9}{@{}l}{\textit{" + lab + r"}} \\" + "\n"
    for t in WT:
        e = win["truths"][t]
        v = [float(x) for x in f(e)]
        body += (f"{WL[t]} & " + " & ".join(f"{x:+.2f}" for x in v) +
                 f" & {max(abs(x) for x in v):.2f} & " + (sci(e['control_max_rel_err']) if si == 0 else "") + " \\\\\n")
e0 = win["truths"]["light"]
cq = win["D_c_quantiles"]
s4 = r"""\begin{table}[ht]
\centering
\caption{\textbf{Post hoc window-placement diagnostic.} For each in-family truth, the generating chain is applied to native GazeBase with the benchmark's own geometry and noise, holds are re-extracted with native-style (sparser) versus degraded-style window placement, and $\eta$ is refitted. Entries are the refitted hold discrepancy on spectral-shape coordinates 5--10 (native-style minus degraded-style placement, the same direction as the real-data discrepancy, simulated minus GazeBaseVR), in three units: the benchmark standard error $s_{\mathrm{bench}}$, the development-split standard error $s_D$ and the confirmatory-split standard error $s_C$. The underlying offset is a population quantity, so its standardised size grows with the sample size of the split. With identical placement the discrepancy vanishes (control column). The largest magnitude rises with filter settling time. It stays below the median clean-window critical value $c_{10}$ (""" + f"{_c10D:.2f}" + r""" on D, """ + f"{_c10C:.2f}" + r""" on C) in D units, but exceeds it in C units, mainly through the lag-1 autocorrelation; in C units the 4--10\,Hz offset is at most """ + f"{max(abs(float(x)) for t in WT for x in [np.array(win['truths'][t]['D_refit'][4]) / _sC[0]]):.2f}" + r""". The entries show a settling-ordered discrepancy under jointly changed placement and window count and do not apportion Stage-1 exclusion. Window counts per recording (median): native-style """ + f"{e0['n_win_native_median']:.0f}" + r""" (windows placed on native GazeBase that remain usable under the degraded validity mask; """ + f"{e0['n_win_native_placed_median']:.0f}" + r""" placed before the mask), degraded-style """ + f"{e0['n_win_degraded_median']:.0f}" + r""" (windows placed on the degraded signal itself). Eligible recordings: native """ + f"{e0['eligible_native']}" + r""", degraded """ + f"{e0['eligible_degraded']}" + r""". Control: identical placement reproduces the cached hold coordinates (maximum relative error shown). Source: \texttt{A006/window.json}.}
\label{tab:s4}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}l""" + "r" * 8 + r"""@{}}
\toprule
Truth & """ + " & ".join(NL) + r""" & max $|\cdot|$ & control err. \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
w("TABLE_S4_window.tex", s4)
print("ok", NAMES5)
