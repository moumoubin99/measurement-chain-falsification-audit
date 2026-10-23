"""Tables 1-3 (main text) and Supplementary Tables S1-S4, written as LaTeX from data/derived.json and result files."""
import json
import os
import sys
from collections import Counter

from paper_plot_style import DATA, RES, ROOT

d = json.load(open(os.path.join(DATA, "derived.json")))
OUT = os.path.dirname(os.path.abspath(__file__))


def w(name, s):
    open(os.path.join(OUT, name), "w").write(s)


def f2(x):
    return f"{x:.2f}"


# ---------------- Table 1: datasets and protocol ----------------
gb = json.load(open(os.path.join(RES, "R001", "gb_summary.json")))
vr = json.load(open(os.path.join(RES, "R001", "summary.json")))
r203 = json.load(open(os.path.join(RES, "R203", "c_hold_lambda.json")))
cal = json.load(open(os.path.join(RES, "R004", "calibration.json")))
pv = d["pv10"]
t1 = r"""\begin{table}[ht]
\centering
\caption{\textbf{Datasets, protocols and participant splits.} Protocol parameters are from the dataset descriptors\cite{gazebase,gazebasevr}; recording counts and splits from \texttt{R001/summary.json} and \texttt{R001/gb\_summary.json}; hold-eligible counts from \texttt{R004/calibration.json} (D) and \texttt{R203/c\_hold\_lambda.json} (C); saccade-eligible counts from \texttt{A006/gap\_pv.json}. Both tasks are random-saccade (RAN) tasks. $^{a}$Over all recordings, at the native sampling rate and before any re-timing; development-split gap fractions on the common 4\,ms grid are given in the Results (missingness diagnostic).}
\label{tab:datasets}
\small
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{6.2cm}>{\raggedright\arraybackslash}p{4.5cm}>{\raggedright\arraybackslash}p{4.3cm}@{}}
\toprule
 & GazeBase (desktop) & GazeBaseVR (headset) \\
\midrule
Eye tracker & EyeLink 1000, chin/forehead rest & SMI tracker in HTC Vive \\
Nominal sampling & 1000 Hz & 250 Hz (timestamp jitter) \\
Eye analysed (primary) & left (monocular recording) & left (binocular recording) \\
Recordings (round 1, session 1, RAN) & """ + f"{gb['n_recordings']}" + r""" & """ + f"{vr['n_recordings']}" + r""" \\
Target range (horizontal / vertical) & $\pm15^\circ$ / $\pm9^\circ$ & $\pm15^\circ$ / $\pm10^\circ$ \\
Target dwell time & 1 s & 1--1.5 s (uniform) \\
Minimum target jump & $2^\circ$ & $3^\circ$ \\
Background & black & light grey \\
Development (D) / confirmatory (C) participants & """ + f"{gb['n_dev']} / {gb['n_conf']}" + r""" & """ + f"{vr['n_dev']} / {vr['n_conf']}" + r""" \\
Hold-eligible ($\geq 10$ windows), D / C & """ + f"{cal['bootstrap_setup']['n_gb_eligible']} / {r203['bootstrap_setup']['n_gb_eligible']}" + r""" & """ + f"{cal['bootstrap_setup']['n_vr_eligible']} / {r203['bootstrap_setup']['n_vr_eligible']}" + r""" \\
Saccade-eligible for native PV@10$^\circ$ (D, spline) & """ + f"{pv['native_gb_spline']['n_elig']}/{pv['native_gb_spline']['P']}" + r""" & """ + f"{pv['native_gbvr_spline']['n_elig']}/{pv['native_gbvr_spline']['P']}" + r""" \\
Median raw invalid-sample fraction (all recordings, native rate)$^{a}$ & """ + f"{100*gb['invalid_median']:.2f}\\%" + r""" & """ + f"{100*vr['nan_left_median']:.2f}\\%" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
w("TABLE_1_datasets.tex", t1)

# ---------------- Table 2: admissibility blocks ----------------
A = d["A006_summary"]
Dfull, Dc10, Dcons = A["D"]["full"], A["D"]["clean10"], A["D"]["consistent"]
Cfull, Cc10, Ccons = A["C"]["full"], A["C"]["clean10"], A["C"]["consistent"]
r203r = [m["ratio"] for m in d["C_R203"]]
cA = d["C_A006"]
c10_filt = [m["clean10"] for m in cA if m["filtered"]]
c10_unf = [m["clean10"] for m in cA if not m["filtered"]]
cons = [m["consistent"] for m in cA]
ff = d["bench_false_failures"]


def bc(counts):
    tot = sum(counts.values())
    top = sorted(counts.items(), key=lambda kv: -kv[1])
    assert tot == 276
    return "; ".join(f"\\texttt{{{k.replace('_', chr(92) + '_')}}} {v}" for k, v in top)


t2 = r"""\begin{table}[ht]
\centering
\caption{\textbf{Fixation-hold admissibility of the declared 276-member family.} A member is admissible if $\Lambda \le c$, i.e.\ $\Lambda/c \le 1$. Block A is the prespecified, code-locked calibration on D and the lock-specified diagnostic check (R203) on C; block B contains post hoc diagnostic recalibrations (A006) that are not decisions; block C is the finite-sample reference from the benchmark with known truths. R203 and A006 on C evaluate nested candidate subsets on the same confirmatory split, not independent replications. Rows bound by \texttt{gap\_frac\_median} share one value of $\Lambda$, because the missingness coordinates do not depend on the member$^{c}$; their $\Lambda/c$ differ only through $c$. Binding coordinate = coordinate attaining $\Lambda$ (counts over members). Sources: \texttt{R004/calibration.json}, \texttt{R203/c\_hold\_lambda.json}, \texttt{A006/d\_adm.json}, \texttt{A006/c\_adm.json}, \texttt{R102/reps/*.json}.}
\label{tab:admissibility}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.0cm}lcc>{\raggedright\arraybackslash}p{4.9cm}@{}}
\toprule
Analysis & Members & Admissible & $\Lambda/c$: min [range] & Binding coordinate \\
\midrule
\multicolumn{5}{@{}l}{\textit{A. Prespecified 12-coordinate calibration}} \\
D split, full family & 276 & 0 & """ + f"{f2(d['D_summary']['min'])} [{f2(d['D_summary']['min'])}, {d['D_summary']['max']:.1f}]" + r""" & """ + bc(Dfull["binding_counts"]) + r""" \\
C split, R203 lock-specified hold check (D-best) & 10 & 0 & """ + f"{f2(min(r203r))} [{f2(min(r203r))}, {f2(max(r203r))}]" + r""" & \texttt{gap\_frac\_median} (all)$^{a}$ \\
\addlinespace
\multicolumn{5}{@{}l}{\textit{B. Post hoc diagnostic recalibrations (A006)}} \\
C split, 12 coordinates (D-selected) & 15 & 0 & """ + f"{f2(Cfull['ratio_min'])} [{f2(Cfull['ratio_min'])}, {f2(max(m['full'] for m in cA))}]" + r""" & \texttt{gap\_frac\_median} (all) \\
D split, clean window (1--10), own $c_{10}$ & 276 & 0 & """ + f"{f2(Dc10['ratio_min'])} (filtered {f2(Dc10['ratio_min_filtered'])})" + r""" & """ + bc(Dc10["binding_counts"]) + r""" \\
C split, clean window (1--10), own $c_{10}$ & 15 & 0 & """ + f"{f2(min(c10_unf))} (filtered {f2(min(c10_filt))}--{f2(max(c10_filt))})" + r""" & \texttt{bp\_4\_10} (all) \\
D split, consistent population & 276 & 0 & """ + f"{f2(Dcons['ratio_min'])}" + r""" & """ + bc(Dcons["binding_counts"]) + r""" \\
C split, consistent population & 15 & 0 & """ + f"{f2(min(cons))} [{f2(min(cons))}, {f2(max(cons))}]" + r""" & \texttt{gap\_frac\_median} (all) \\
\addlinespace
\multicolumn{5}{@{}l}{\textit{C. Benchmark replicates with no admissible member ($A_r = \varnothing$)}} \\
In-family truths (false failures) & 11 of 104 & 0 & """ + f"{f2(ff['min_in'])}--{f2(ff['max_in'])}$^{{b}}$" + r""" & --- \\
Out-of-family truths & 6 of 52 & 0 & """ + f"{f2(ff['min_out'])}--{f2(ff['max_out'])}$^{{b}}$" + r""" & --- \\
\bottomrule
\end{tabular}
\end{adjustbox}

\smallskip
\raggedright\footnotesize
$^{a}$R203 $\Lambda/c$ values are reproduced exactly by the A006 recomputation, which records the binding coordinate. $^{b}$Per replicate, the minimum $\Lambda/c$ over the 276 members; 1.51 is the observed maximum over the 11 in-family replicates, not a bound. Benchmark pseudo-headset data were generated with the family's own missingness model, so the missingness coordinates are matched in expectation by construction; block C is therefore no reference for failures that are set by \texttt{gap\_frac\_median}, as in blocks A and B. $^{c}$On D, $z = 8.43$ for \texttt{gap\_frac\_median} and $z = 2.51$ for \texttt{gap\_loglen\_mean} for every member; 99 members are tied at $\Lambda = 8.43$, with $c$ of 3.02--3.10. On C, $\Lambda = 15.57$ for all 15 candidates (14.80 under the consistent population), with $c$ of 3.47--3.53. Binding coordinates: \texttt{ac1}, \texttt{ac2}, lag-1 and lag-2 autocorrelation of horizontal sample-to-sample velocity (coordinates 9--10); \texttt{bp\_4\_10}, \texttt{bp\_60\_125}, log band power of horizontal gaze position at 4--10 and 60--125\,Hz (coordinates 5 and 8); \texttt{gap\_frac\_median}, median gap fraction on the 4\,ms grid (coordinate 11).
\end{table}
"""
w("TABLE_2_admissibility.tex", t2)

# ---------------- Table 3: benchmark operating characteristics ----------------
B = d["bench"]
T = ["light", "medium", "heavy", "b2", "vel_adaptive", "kalman_rts"]
HEAD = ["Light", "Medium", "Heavy", "B2", "Vel.-adaptive", "Kalman RTS"]


def row(label, fn):
    return label + " & " + " & ".join(fn(B[t]) for t in T) + r" \\" + "\n"


def wil(lo_hi):
    return f"[{lo_hi[0]:.2f}, {lo_hi[1]:.2f}]"


def ret_un(b):
    return f"{b['retained']}/{b['N']}" if b["in_family"] else "n/a"


def ret_c(b):
    return f"{b['retained']}/{b['n_nonempty']}" if b["in_family"] else "n/a"


def ret_cw(b):
    return wil(b["retention_given_nonempty_wilson"]) if b["in_family"] else ""


def wpm(b):
    return str(b["decisions"].get("MISMATCH", 0)) + f"/{b['N']}"


t3 = r"""\begin{table}[ht]
\centering
\caption{\textbf{Operating characteristics of the two-stage procedure on the benchmark with known truths} (26 replicates per truth, each a random partition of the 100 development participants into four disjoint arms of 25; real GazeBase recordings degraded by a known chain; in-family truths are post-filter-noise $\alpha = 1$ members). Every row states its denominator. \emph{Whole-procedure} quantities count all 26 replicates; \emph{conditional} quantities condition on $A_r \ne \varnothing$ and judge each Stage-2 decision against the oracle computed on the same $A_r$. Intervals are 95\% Wilson intervals; $|A_r|$ and ambiguity are reported as median and quartiles. Sources: \texttt{R102/summary.json} and \texttt{R102/reps/*.json} (counts).}
\label{tab:benchmark}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.6cm}""" + "c" * 6 + r"""@{}}
\toprule
 & \multicolumn{4}{c}{In family} & \multicolumn{2}{c}{Out of family} \\
\cmidrule(lr){2-5}\cmidrule(l){6-7}
 & """ + " & ".join(HEAD) + r""" \\
\midrule
Generating-chain settling time (ms) & """ + " & ".join(f"{B[t]['settling_ms']:.0f}" if B[t]["in_family"] else "---" for t in T) + r""" \\
$A_r = \varnothing$ (calibration failure), $k/26$ & """ + " & ".join(f"{B[t]['n_empty']}" for t in T) + r""" \\
Truth retained in $A_r$, whole procedure, $k/26$ & """ + " & ".join(ret_un(B[t]) for t in T) + r""" \\
Truth retained given $A_r \ne \varnothing$ & """ + " & ".join(ret_c(B[t]) for t in T) + r""" \\
\quad 95\% Wilson interval & """ + " & ".join(ret_cw(B[t]) for t in T) + r""" \\
$|A_r|$ (all 26 replicates), median & """ + " & ".join(f"{B[t]['A_size_q'][1]:.0f}" for t in T) + r""" \\
\quad quartiles & """ + " & ".join(f"[{B[t]['A_size_q'][0]:.0f}, {B[t]['A_size_q'][2]:.0f}]" for t in T) + r""" \\
PV ambiguity (log units), median$^{a}$ & """ + " & ".join(f"{B[t]['ambiguity_q'][1]:.2f}" for t in T) + r""" \\
\quad quartiles & """ + " & ".join(f"[{B[t]['ambiguity_q'][0]:.2f}, {B[t]['ambiguity_q'][2]:.2f}]" for t in T) + r""" \\
\addlinespace
\multicolumn{7}{@{}l}{\textit{Stage-2 decisions, $k/\#\{A_r \ne \varnothing\}$}} \\
\quad INCONCLUSIVE (imprecise) & """ + " & ".join(f"{B[t]['n_imprecise']}/{B[t]['n_nonempty']}" for t in T) + r""" \\
\quad INCONCLUSIVE (assumption-sensitive) & """ + " & ".join(f"{B[t]['decisions'].get('INCONCLUSIVE/assumption-sensitive', 0)}/{B[t]['n_nonempty']}" for t in T) + r""" \\
\quad MISMATCH & """ + " & ".join(f"{B[t]['decisions'].get('MISMATCH', 0)}/{B[t]['n_nonempty']}" for t in T) + r""" \\
\quad EQUIVALENCE & """ + " & ".join(f"{B[t]['decisions'].get('EQUIVALENCE', 0)}/{B[t]['n_nonempty']}" for t in T) + r""" \\
Conditional false EQUIVALENCE or MISMATCH & """ + " & ".join(f"{B[t]['n_false_eq_cond'] + B[t]['n_false_mm_cond']}/{B[t]['n_resolved']}" for t in T) + r""" \\
\quad 95\% Wilson upper limit$^{b}$ & """ + " & ".join(f"{B[t]['false_mm_cond_wilson_upper']:.3f}" for t in T) + r""" \\
Whole-procedure false MISMATCH (true ratio 1), $k/26$ & """ + " & ".join(wpm(B[t]) for t in T) + r""" \\
\addlinespace
Simultaneous max-T coverage over $A_r$, replicates with $A_r \ne \varnothing$ (nominal 0.90) & """ + " & ".join(f"{B[t]['simcov']:.2f}" for t in T) + r""" \\
\quad 95\% Wilson interval & """ + " & ".join(wil(B[t]["simcov_wilson"]) for t in T) + r""" \\
\bottomrule
\end{tabular}
\end{adjustbox}

\smallskip
\raggedright\footnotesize
$^{a}$Per replicate, the max $-$ min range of member PV contrasts $\hat\delta$ over $A_r$; replicates with $A_r = \varnothing$ are excluded. For scale, the full equivalence width is $\ln(1.05/0.95) \approx 0.100$. $^{b}$At the non-empty-$A_r$ denominators shown; counting all 26 replicates gives 0.129 per truth. Pooled over truths, 0/""" + str(sum(B[t]["n_resolved"] for t in T)) + r""" replicates with non-empty $A_r$ (0/156 counting empty $A_r$ as no decision). Retention is not defined for out-of-family truths (n/a).
\end{table}
"""
w("TABLE_3_benchmark.tex", t3)
print("ok", sum(B[t]["n_resolved"] for t in T))
