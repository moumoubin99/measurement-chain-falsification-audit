"""Frozen spline basis, per-(participant, realisation) sufficient quantities, standardised log-PV
endpoint, max-T simultaneous bands and the preregistered decision rule."""
from __future__ import annotations

import math

import numpy as np

LO, HI = math.log(0.95), math.log(1.05)
P_ELIG_MIN = 0.60
N_812_MIN = 500.0
COND_MAX = 1e8
BAND_LEVEL = 0.90
NCOL = 7


def ns_basis(X, knots):
    """ESL natural cubic spline basis with K=4 knots (boundary log5, log15): [X, d1-d3, d2-d3]."""
    k = np.asarray(knots, float)
    K = len(k)

    def d(j):
        return (np.maximum(X - k[j], 0) ** 3 - np.maximum(X - k[K - 1], 0) ** 3) / (k[K - 1] - k[j])
    dK = d(K - 2)
    return np.column_stack([X] + [d(j) - dK for j in range(K - 2)])


def design(amp, theta, knots):
    X = np.log(np.asarray(amp, float))
    th = np.asarray(theta, float)
    return np.column_stack([ns_basis(X, knots), np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)])


def make_knots(ref_amps):
    la = np.log(np.asarray(ref_amps, float))
    q = np.quantile(la, [1 / 3, 2 / 3])
    return [math.log(5.0), float(q[0]), float(q[1]), math.log(15.0)]


def make_xref(knots, ref_theta):
    th = np.asarray(ref_theta, float)
    h = [np.cos(th).mean(), np.sin(th).mean(), np.cos(2 * th).mean(), np.sin(2 * th).mean()]
    return np.concatenate([ns_basis(np.array([math.log(10.0)]), knots)[0], h])


def suff(events, knots):
    """Sufficient quantities for one (participant, realisation) from a list of event dicts."""
    n = len(events)
    out = dict(n=n, xbar=np.zeros(NCOL), ybar=0.0, Sxx=np.zeros((NCOL, NCOL)), Sxy=np.zeros(NCOL), n812=0)
    if n == 0:
        return out
    amp = np.array([e["amp"] for e in events])
    X = design(amp, [e["theta"] for e in events], knots)
    y = np.log([e["pv"] for e in events])
    xb, yb = X.mean(0), y.mean()
    Xc, yc = X - xb, y - yb
    out.update(xbar=xb, ybar=float(yb), Sxx=Xc.T @ Xc, Sxy=Xc.T @ yc, n812=int(((amp >= 8) & (amp <= 12)).sum()))
    return out


def design_lin(amp, theta):
    """8-12 deg sensitivity design: linear in log A plus the same 4 harmonics."""
    th = np.asarray(theta, float)
    return np.column_stack([np.log(np.asarray(amp, float)), np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)])


def suff_lin(events, lo=8.0, hi=12.0):
    """Sufficient quantities of the 8-12 deg linear-in-log-A sensitivity estimator (events restricted to [lo, hi])."""
    ev = [e for e in events if lo <= e["amp"] <= hi]
    n = len(ev)
    out = dict(n=n, xbar=np.zeros(5), ybar=0.0, Sxx=np.zeros((5, 5)), Sxy=np.zeros(5), n812=n)
    if n == 0:
        return out
    X = design_lin([e["amp"] for e in ev], [e["theta"] for e in ev])
    y = np.log([e["pv"] for e in ev])
    xb, yb = X.mean(0), y.mean()
    Xc, yc = X - xb, y - yb
    out.update(xbar=xb, ybar=float(yb), Sxx=Xc.T @ Xc, Sxy=Xc.T @ yc)
    return out


def xref_lin(xref):
    return np.concatenate([[math.log(10.0)], np.asarray(xref)[-4:]])


def stack(per_pr):
    """per_pr: list over participants of list over realisations of suff dicts -> arrays."""
    P, R = len(per_pr), len(per_pr[0])
    nc = len(per_pr[0][0]["xbar"])          # NCOL for the frozen basis; 5 for the 8-12 deg linear sensitivity
    n = np.zeros((P, R))
    xb = np.zeros((P, R, nc))
    yb = np.zeros((P, R))
    Sxx = np.zeros((P, R, nc, nc))
    Sxy = np.zeros((P, R, nc))
    n812 = np.zeros((P, R))
    for p, row in enumerate(per_pr):
        for r, s in enumerate(row):
            n[p, r], xb[p, r], yb[p, r] = s["n"], s["xbar"], s["ybar"]
            Sxx[p, r], Sxy[p, r], n812[p, r] = s["Sxx"], s["Sxy"], s["n812"]
    return dict(n=n, xbar=xb, ybar=yb, Sxx=Sxx, Sxy=Sxy, n812=n812)


def reduce_participants(st):
    """Eligibility and per-participant reduced quantities (sums over contributing realisations)."""
    n = st["n"]
    P, R = n.shape
    contrib = n >= 2
    nbar = np.where(contrib, n, 0).mean(1)   # non-contributing realisations count as 0
    elig = (contrib.sum(1) >= math.ceil(R / 2)) & (nbar >= 5)
    w = contrib & elig[:, None]
    Rp = np.maximum(w.sum(1), 1)
    Sxx_p = (st["Sxx"] * w[..., None, None]).sum(1)
    Sxy_p = (st["Sxy"] * w[..., None]).sum(1)
    ybar_p = (st["ybar"] * w).sum(1) / Rp
    xbar_p = (st["xbar"] * w[..., None]).sum(1) / Rp[:, None]
    n812_p = np.where(contrib, st["n812"], 0).mean(1)
    return dict(elig=elig, Sxx=Sxx_p, Sxy=Sxy_p, ybar=ybar_p, xbar=xbar_p, n812=n812_p, nbar=nbar,
                n_contrib=contrib.sum(1), R=R)


def estimate(red, xref, weights=None):
    """E-hat with participant multiplicity weights (bootstrap).  Returns (E, beta, cond)."""
    e = red["elig"].astype(float)
    w = e if weights is None else e * weights
    if w.sum() == 0:
        return np.nan, None, np.inf
    S = np.tensordot(w, red["Sxx"], axes=(0, 0))
    v = np.tensordot(w, red["Sxy"], axes=(0, 0))
    cond = float(np.linalg.cond(S))
    if not np.isfinite(cond) or cond > 1e14:
        return np.nan, None, cond
    beta = np.linalg.solve(S, v)
    m = red["ybar"] - red["xbar"] @ beta
    E = float((w * m).sum() / w.sum() + beta @ xref)
    return E, beta, cond


def support(red, cond):
    P = len(red["elig"])
    frac = red["elig"].mean() if P else 0.0
    n812 = float(red["n812"][red["elig"]].sum())
    ok = (frac >= P_ELIG_MIN) and (n812 >= N_812_MIN) and (cond < COND_MAX)
    return dict(ok=bool(ok), frac_elig=float(frac), n812=n812, cond=float(cond))


def weights_from_idx(idx, P):
    return np.bincount(idx, minlength=P).astype(float)


BOOT_NONFINITE_MAX = 0          # number of non-finite bootstrap draws tolerated
SE_MIN = 1e-12


def bands(red_vr, reds_gb, xref, B, seed, level=BAND_LEVEL):
    """Max-T simultaneous bands over members.  reds_gb: dict member -> reduced GB quantities (same participants)."""
    Pvr = len(red_vr["elig"])
    Pgb = len(next(iter(reds_gb.values()))["elig"])
    rng = np.random.default_rng([8_2026_1018, 999, seed])
    iv = rng.integers(0, Pvr, size=(B, Pvr))
    ig = rng.integers(0, Pgb, size=(B, Pgb))
    E_vr, _, _ = estimate(red_vr, xref)
    mems = list(reds_gb)
    dhat = np.array([E_vr - estimate(reds_gb[m], xref)[0] for m in mems])
    Db = np.empty((B, len(mems)))
    for b in range(B):
        ev = estimate(red_vr, xref, weights_from_idx(iv[b], Pvr))[0]
        wg = weights_from_idx(ig[b], Pgb)
        Db[b] = [ev - estimate(reds_gb[m], xref, wg)[0] for m in mems]
    # explicit estimability rules (no NaN dropping): every point estimate finite; any bootstrap draw with a
    # non-finite delta makes the bands non-estimable (conditioning the max-T distribution on successful draws is
    # not the prespecified procedure); a member with zero bootstrap SE makes the bands non-estimable.
    reason = ""
    row_ok = np.isfinite(Db).all(1)
    n_bad = int((~row_ok).sum())
    if not np.isfinite(dhat).all():
        reason = "nonfinite_point_estimate"
    elif n_bad > BOOT_NONFINITE_MAX:
        reason = "nonfinite_bootstrap"
    Dg = Db[row_ok]
    se = Dg.std(0, ddof=1) if len(Dg) > 1 else np.full(len(mems), np.nan)
    if not reason and not (np.isfinite(se).all() and (se > SE_MIN).all()):
        reason = "zero_or_nonfinite_se"
    if reason:
        nan = np.full(len(mems), np.nan)
        return dict(members=mems, dhat=dhat, se=se, c=np.nan, lo=nan, hi=nan.copy(), E_vr=E_vr, estimable=False,
                    reason=reason, n_boot_nonfinite=n_bad)
    T = np.max(np.abs(Dg - dhat) / se, axis=1)
    c = float(np.quantile(T, level))
    return dict(members=mems, dhat=dhat, se=se, c=c, lo=dhat - c * se, hi=dhat + c * se, E_vr=E_vr, estimable=True,
                reason="", n_boot_nonfinite=n_bad)


def decide(lo, hi, support_ok=True, LOm=LO, HIm=HI, estimable=True):
    lo, hi = np.asarray(lo), np.asarray(hi)
    if not support_ok:
        return "INCONCLUSIVE", "support"
    if not estimable or not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return "INCONCLUSIVE", "non-estimable"
    if np.all((lo >= LOm) & (hi <= HIm)):
        return "EQUIVALENCE", ""
    if np.all(hi < LOm) or np.all(lo > HIm):
        return "MISMATCH", ""
    inside = (lo >= LOm) & (hi <= HIm)
    below, above = hi < LOm, lo > HIm
    if (inside.any() and (below.any() or above.any())) or (below.any() and above.any()):
        return "INCONCLUSIVE", "assumption-sensitive"
    return "INCONCLUSIVE", "imprecise"


def m_star(dhat, se, c):
    return float(np.max(np.abs(dhat) + c * np.asarray(se)))
