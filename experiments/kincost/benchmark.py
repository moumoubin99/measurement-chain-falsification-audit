"""Participant-disjoint known-transformation benchmark (Block 4).

Per replicate: quarters Q1..Q4 of the 100 D-GB participants; pseudo-GBVR = truth applied to Q1/Q3 (one
realisation, separate seed stream); full procedure = noise refit + profiled admissibility (Q1 vs Q2 holds,
B_adm) -> A_r; endpoint + max-T bands (Q3 vs Q4, R_bm realisations, B_band) -> decision; replicate-aligned
oracle on all 100 participants (K batches x R_bm realisations; truth endpoint = batch mean of R = 1 endpoints; Bonferroni t intervals) -> RESOLVED/UNRESOLVED.

Truths: 4 in-family manifest members at their full-D eta*, and 2 out-of-family chains (velocity-adaptive
Butterworth blend; constant-velocity Kalman RTS smoother) with post noise at the medium truth's eta*.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import stats

from . import calibrate as CAL
from . import chain as C
from . import detector as DET
from . import endpoint as EP
from . import holds as H
from . import pipeline as PL
from .fastfit import njit

IN_FAMILY = {"light": "B1_bw2_60_post_a1", "medium": "B1_bw4_30_post_a1", "heavy": "B1_bw4_15_post_a1",
             "b2": "B2_bw4_40_post_a1"}
OUT_OF_FAMILY = ("vel_adaptive", "kalman_rts")
TRUTHS = tuple(IN_FAMILY) + OUT_OF_FAMILY
VA_HEAVY = ("butter", (4, 15))
VA_LIGHT = ("butter", (2, 60))
VA_V0, VA_WIDTH = 30.0, 5.0          # deg/s: logistic blend centre / width on light-filtered speed
KF_FC = 25.0                          # Hz: CV smoother corner, q/r = (2 pi fc)^4 * dt (discrete r)
ALPHA_MC = 0.05
K_ORACLE = 6
R_BM = 4
ST_PSEUDO, ST_ORACLE, ST_EVAL = 1000, 2000, 3000


# --------------------------------------------------------------------------------------- out-of-family truths
@njit(cache=True)
def _rts_cv(t_s, z, q, r):
    """Constant-velocity Kalman filter + RTS smoother on one run (irregular times t_s in seconds)."""
    n = len(z)
    xf0 = np.empty(n)
    xf1 = np.empty(n)
    P00 = np.empty(n)
    P01 = np.empty(n)
    P11 = np.empty(n)
    xp0 = np.empty(n)
    xp1 = np.empty(n)
    Q00 = np.empty(n)
    Q01 = np.empty(n)
    Q11 = np.empty(n)
    a0, a1 = z[0], 0.0
    p00, p01, p11 = r, 0.0, 1e6
    for k in range(n):
        if k > 0:
            dt = t_s[k] - t_s[k - 1]
            a0 = a0 + dt * a1
            q00, q01, q11 = q * dt ** 3 / 3, q * dt ** 2 / 2, q * dt
            n00 = p00 + 2 * dt * p01 + dt * dt * p11 + q00
            n01 = p01 + dt * p11 + q01
            n11 = p11 + q11
            p00, p01, p11 = n00, n01, n11
        else:
            q00 = q01 = q11 = 0.0
        xp0[k], xp1[k], Q00[k], Q01[k], Q11[k] = a0, a1, p00, p01, p11
        s = p00 + r
        k0, k1 = p00 / s, p01 / s
        innov = z[k] - a0
        a0, a1 = a0 + k0 * innov, a1 + k1 * innov
        p00, p01, p11 = (1 - k0) * p00, (1 - k0) * p01, p11 - k1 * p01
        xf0[k], xf1[k], P00[k], P01[k], P11[k] = a0, a1, p00, p01, p11
    out = np.empty(n)
    s0, s1 = xf0[n - 1], xf1[n - 1]
    out[n - 1] = s0
    for k in range(n - 2, -1, -1):
        dt = t_s[k + 1] - t_s[k]
        # predicted cov at k+1: Q.. arrays; G = P_k F^T Ppred^-1
        f00, f01 = P00[k] + dt * P01[k], P01[k]
        f10, f11 = P01[k] + dt * P11[k], P11[k]
        d00, d01, d11 = Q00[k + 1], Q01[k + 1], Q11[k + 1]
        det = d00 * d11 - d01 * d01
        i00, i01, i11 = d11 / det, -d01 / det, d00 / det
        g00 = f00 * i00 + f01 * i01
        g01 = f00 * i01 + f01 * i11
        g10 = f10 * i00 + f11 * i01
        g11 = f10 * i01 + f11 * i11
        e0, e1 = s0 - xp0[k + 1], s1 - xp1[k + 1]
        s0 = xf0[k] + g00 * e0 + g01 * e1
        s1 = xf1[k] + g10 * e0 + g11 * e1
        out[k] = s0
    return out


class FamilyTruth:
    """In-family truth: a manifest member at fixed eta*."""

    def __init__(self, member, eta):
        self.name = member.name
        self.member = member
        self.chain = C.Chain(member)
        self.eta = np.asarray(eta, float)

    def degraded(self, geo, rec, sH, sV):
        return self.chain.degraded(geo, rec, sH, sV)


class OOFTruth:
    """Out-of-family truth: nonlinear (velocity-adaptive) or two-sided state-space smoother at the donor stage,
    then shaped post noise (alpha) added at the donor stage, as for B1 post members."""

    def __init__(self, kind, eta, alpha=1.0):
        self.name = kind
        self.kind = kind
        self.eta = np.asarray(eta, float)
        self.alpha = alpha
        self.ba_h = C.filter_ba(*VA_HEAVY, 250.0)
        self.ba_l = C.filter_ba(*VA_LIGHT, 250.0)
        self.q_over_r = (2 * math.pi * KF_FC) ** 4 * (C.GRID_DT / 1000.0)

    def _donor(self, geo, x1000):
        x = np.where(geo.valid1000, np.nan_to_num(x1000), 0.0)
        return np.where(geo.valid_d, C.apply_interp(x, geo.op_d), 0.0)

    def _smooth(self, geo, xd, yd):
        if self.kind == "vel_adaptive":
            xh, yh = C.filt_runs(xd, geo.runs_d, self.ba_h), C.filt_runs(yd, geo.runs_d, self.ba_h)
            xl, yl = C.filt_runs(xd, geo.runs_d, self.ba_l), C.filt_runs(yd, geo.runs_d, self.ba_l)
            sp = np.zeros_like(xd)
            for a, b in geo.runs_d:
                if b - a >= 3:
                    tt = geo.t_d[a:b]
                    sp[a:b] = 1000.0 * np.hypot(np.gradient(xl[a:b], tt), np.gradient(yl[a:b], tt))
            w = 1.0 / (1.0 + np.exp(-(sp - VA_V0) / VA_WIDTH))
            return w * xl + (1 - w) * xh, w * yl + (1 - w) * yh
        xs, ys = xd.copy(), yd.copy()
        for a, b in geo.runs_d:
            if b - a >= 2:
                ts = geo.t_d[a:b] / 1000.0
                xs[a:b] = _rts_cv(ts, xd[a:b], self.q_over_r, 1.0)
                ys[a:b] = _rts_cv(ts, yd[a:b], self.q_over_r, 1.0)
        return xs, ys

    def degraded(self, geo, rec, sH, sV):
        xd, yd = self._donor(geo, rec["x"]), self._donor(geo, rec["y"])
        xs, ys = self._smooth(geo, xd, yd)
        key = ("post", self.alpha, 0)
        for ax in (0, 1):
            k = ("post", self.alpha, ax)
            if k not in geo.noise_cache:
                geo.noise_cache[k] = C.shaped_noise(geo.white_post[ax], self.alpha, 250.0)
        xs = xs + np.where(geo.valid_d, sH * geo.noise_cache[key], 0.0)
        ys = ys + np.where(geo.valid_d, sV * geo.noise_cache[("post", self.alpha, 1)], 0.0)
        return C.apply_interp(xs, geo.op_g), C.apply_interp(ys, geo.op_g)


def make_truth(name, members_by_name, eta_by_name):
    if name in IN_FAMILY:
        m = members_by_name[IN_FAMILY[name]]
        return FamilyTruth(m, eta_by_name[m.name])
    return OOFTruth(name, eta_by_name[IN_FAMILY["medium"]])


# --------------------------------------------------------------------------------------- per-participant helpers
def sigma(eta, z):
    sH, sV = H.sigma_of(eta, np.array([z]))
    return float(sH[0]), float(sV[0])


def detect_primary(geo, xg, yg, tt, targets):
    x = np.where(geo.usable_g, xg, 0.0)
    y = np.where(geo.usable_g, yg, 0.0)
    ev, thr, info = DET.detect(x, y, geo.usable_g, H.GRID_FS)
    return DET.primary_saccades(ev, geo.grid_t, x, y, tt, targets)


def pseudo_target(rec, pool, e_set, truth, rep, ti, z, win_ms, bands, knots, want_events):
    """One pseudo-GBVR realisation of participant rec under the truth: holds (as native) and endpoint suff."""
    pid = int(rec["pid"])
    geo = C.make_geometry(rec, pool, pid, rep, ST_PSEUDO + ti, e_set)
    xg, yg = truth.degraded(geo, rec, *sigma(truth.eta, z))
    tt, targets = PL.trans_info(rec)
    out = PL.grid_native(geo.grid_t, xg, yg, geo.valid_g, geo.usable_g, tt, targets, pid, win_ms, bands,
                         want_events=want_events)
    res = dict(Q=out["Q"], gap=out["gap"], n_win=out["n_win"])
    if want_events:
        res["suff"] = EP.suff(out["primary"], knots)
    return res


def endpoint_suffs(rec, pool, e_set, chains, etas, z, knots, geo_rs, stream):
    """Exact noise-scaled re-detection for several chains on shared geometries.
    chains: list of objects with .degraded; etas: matching list.  Returns list over chains of list over r."""
    pid = int(rec["pid"])
    tt, targets = PL.trans_info(rec)
    out = [[None] * len(geo_rs) for _ in chains]
    for j, r in enumerate(geo_rs):
        geo = C.make_geometry(rec, pool, pid, r, stream, e_set)
        for i, (ch, eta) in enumerate(zip(chains, etas)):
            xg, yg = ch.degraded(geo, rec, *sigma(eta, z))
            out[i][j] = EP.suff(detect_primary(geo, xg, yg, tt, targets), knots)
    return out


# --------------------------------------------------------------------------------------- oracle
def oracle_category(l, u, LOm=EP.LO, HIm=EP.HI):
    """RESOLVED iff every point configuration in the box prod[l_m, u_m] gives the same decision category."""
    l, u = np.asarray(l), np.asarray(u)
    if np.all((l >= LOm) & (u <= HIm)):
        return "EQUIVALENCE", True
    if np.all(u < LOm) or np.all(l > HIm):
        return "MISMATCH", True
    eq_possible = np.all((u >= LOm) & (l <= HIm))
    mm_possible = np.all(l < LOm) or np.all(u > HIm)
    if not eq_possible and not mm_possible:
        return "INCONCLUSIVE", True
    return "UNRESOLVED", False


def oracle_intervals(dk, alpha=ALPHA_MC):
    """dk: (K, M) batch-level delta*.  Bonferroni t intervals over all M oracle members.  A member whose K batch
    estimates are not all finite gets NaN mean/sd/interval (never dropped silently); the caller then reports the
    oracle as UNRESOLVED."""
    K, M = dk.shape
    fin = np.isfinite(dk).all(0)
    mean = np.full(M, np.nan)
    sd = np.full(M, np.nan)
    mean[fin] = dk[:, fin].mean(0)
    sd[fin] = dk[:, fin].std(0, ddof=1)
    tq = stats.t.ppf(1 - alpha / (2 * M), K - 1)
    hw = tq * sd / math.sqrt(K)
    return mean, sd, mean - hw, mean + hw


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def quarters(P, rep, ti):
    rng = np.random.default_rng([C.SEED_BASE, 4040, ti, rep])
    perm = rng.permutation(P)
    q = P // 4
    return [np.sort(perm[i * q:(i + 1) * q]) for i in range(4)]


def _red_E(per_part, xref, Pidx=None):
    st = EP.stack(per_part if Pidx is None else [per_part[i] for i in Pidx])
    red = EP.reduce_participants(st)
    E, _, cond = EP.estimate(red, xref)
    return red, E, cond


# --------------------------------------------------------------------------------------- replicate
def run_replicate(ctx, truth_name, rep, B_adm=500, B_band=1000, R_bm=R_BM, K=K_ORACLE, n812_scale=None,
                  do_oracle=True, log=None):
    """ctx: dict with recs (100 D-GB), pool, e_set, win_ms, bands, knots, xref, caches [(mid, cache)] on the
    100 D-GB participants (R_cal), members (list), eta_full {name: eta}.  Serial; returns dict with timings."""
    tm = {}
    t0 = time.perf_counter()
    members = ctx["members"]
    mbn = {m.name: m for m in members}
    truth = make_truth(truth_name, mbn, ctx["eta_full"])
    ti = TRUTHS.index(truth_name)
    recs = ctx["recs"]
    P = len(recs)
    z = np.array([PL.z_of(r["pid"]) for r in recs])
    q1, q2, q3, q4 = quarters(P, rep, ti)
    win_ms, bands, knots, xref, e_set, pool = (ctx[k] for k in ("win_ms", "bands", "knots", "xref", "e_set",
                                                                "pool"))
    # 1. pseudo-GBVR on Q1 (holds) and Q3 (holds not needed; endpoint suff)
    pt1 = [pseudo_target(recs[i], pool, e_set, truth, rep, ti, z[i], win_ms, bands, knots, False) for i in q1]
    pt3 = [pseudo_target(recs[i], pool, e_set, truth, rep, ti, z[i], win_ms, bands, knots, True) for i in q3]
    nat = dict(Q=np.array([p["Q"] for p in pt1])[:, None, :], gap=np.array([p["gap"] for p in pt1])[:, None, :],
               nb=len(bands))
    lH, lV, lb, ac1, ac2 = H.participant_coords(nat["Q"][:, 0], len(bands))
    pp = np.column_stack([lH, lV, lH, lV, lb, ac1, ac2, nat["gap"][:, 0, 0],
                          nat["gap"][:, 0, 1] / np.maximum(nat["gap"][:, 0, 2], 1)])
    nat["pp_sd"] = np.nanstd(pp, 0)
    tm["pseudo_target_s"] = time.perf_counter() - t0
    # 2. admissibility on Q1 vs Q2
    t1 = time.perf_counter()
    sub = [(mid, {k: (v[q2] if isinstance(v, np.ndarray) and v.shape[:1] == (P,) else v) for k, v in c.items()})
           for mid, c in ctx["caches"]]
    adm, adm_meta = CAL.run_admissibility(sub, nat, B_adm, seed=7000 + 100 * ti + rep, workers=1)
    tm["admissibility_s"] = time.perf_counter() - t1
    A = [r["mid"] for r in adm if r["admissible"]]
    eta_r = {r["mid"]: np.array(r["eta"]) for r in adm}
    g_mid = mbn[IN_FAMILY[truth_name]].mid if truth_name in IN_FAMILY else None
    out = dict(truth=truth_name, rep=rep, quarters=[q.tolist() for q in (q1, q2, q3, q4)], A=A,
               retained=(g_mid in A) if g_mid is not None else None,
               Lambda_over_c={r["mid"]: (r["Lambda"] / r["c"] if r["c"] > 0 else None) for r in adm},
               eta_r={mid: e.tolist() for mid, e in eta_r.items()},
               flags_A=sum(any(r["flags"].values()) for r in adm if r["admissible"]))
    if log:
        log(f"[{truth_name} r{rep}] |A|={len(A)} retained={out['retained']} adm {tm['admissibility_s']:.0f}s")
    if not A:
        out.update(decision="CALIBRATION FAILURE", tag="", timings=tm)
        return out
    # 3. endpoint on Q3 (pseudo) vs Q4 (members in A at eta_r), R_bm realisations
    t2 = time.perf_counter()
    chains = [C.Chain(members[m]) for m in A]
    etas = [eta_r[m] for m in A]
    q4_suff = [endpoint_suffs(recs[i], pool, e_set, chains, etas, z[i], knots,
                              geo_rs=[rep * 1000 + j for j in range(R_bm)], stream=ST_EVAL + ti) for i in q4]
    tm["endpoint_s"] = time.perf_counter() - t2
    red_vr = EP.reduce_participants(EP.stack([[p["suff"]] for p in pt3]))
    E_vr, _, cond_vr = EP.estimate(red_vr, xref)
    reds = {}
    sup = {}
    scale = n812_scale if n812_scale is not None else 1.0
    for k, m in enumerate(A):
        red = EP.reduce_participants(EP.stack([q4_suff[p][k] for p in range(len(q4))]))
        reds[m] = red
        _, _, cond = EP.estimate(red, xref)
        sup[m] = _support(red, cond, scale)
    sup_vr = _support(red_vr, cond_vr, scale)
    support_ok = sup_vr["ok"] and all(s["ok"] for s in sup.values())
    t3 = time.perf_counter()
    bd = EP.bands(red_vr, reds, xref, B_band, seed=9000 + 100 * ti + rep)
    tm["bands_s"] = time.perf_counter() - t3
    dec, tag = EP.decide(bd["lo"], bd["hi"], support_ok, estimable=bd["estimable"])
    out.update(decision=dec, tag=tag, support_vr=sup_vr, support=sup, dhat=bd["dhat"].tolist(),
               se=bd["se"].tolist(), c_band=bd["c"], lo=bd["lo"].tolist(), hi=bd["hi"].tolist(),
               band_estimable=bd["estimable"], band_reason=bd["reason"], n_boot_nonfinite=bd["n_boot_nonfinite"],
               ambiguity=(float(np.max(bd["dhat"]) - np.min(bd["dhat"])) if np.isfinite(bd["dhat"]).all() else None))
    if log:
        log(f"[{truth_name} r{rep}] decision={dec} {tag} endpoint {tm['endpoint_s']:.0f}s")
    # 4. oracle on all P participants: K batches x R_bm, truth vs members (A u g) at eta_r, shared geometry
    if do_oracle:
        t4 = time.perf_counter()
        omids = list(A) + ([g_mid] if g_mid is not None and g_mid not in A else [])
        ochains = [truth] + [C.Chain(members[m]) for m in omids]
        oetas = [truth.eta] + [eta_r[m] for m in omids]
        per = [endpoint_suffs(recs[i], pool, e_set, ochains, oetas, z[i], knots,
                              geo_rs=[rep * 1000 + j for j in range(K * R_bm)], stream=ST_ORACLE + ti)
               for i in range(P)]
        dk = np.full((K, len(omids)), np.nan)
        for k in range(K):
            sl = slice(k * R_bm, (k + 1) * R_bm)
            # truth: procedure-aligned with the pseudo-target (one realisation per participant, R = 1 eligibility);
            # the batch value is the mean of the R_bm single-realisation endpoints
            Et = float(np.mean([EP.estimate(EP.reduce_participants(EP.stack([[per[p][0][j]] for p in range(P)])),
                                            xref)[0] for j in range(k * R_bm, (k + 1) * R_bm)]))
            for j in range(len(omids)):
                Em = EP.estimate(EP.reduce_participants(EP.stack([per[p][j + 1][sl] for p in range(P)])), xref)[0]
                dk[k, j] = Et - Em
        mean, sd, l, u = oracle_intervals(dk)
        inA = np.array([m in A for m in omids])
        # Bonferroni over all oracle members (A u g); the decision category is classified over A only
        if not np.all(np.isfinite(l[inA]) & np.isfinite(u[inA])):
            cat, resolved, why = "UNRESOLVED", False, "nonfinite_batch"
        else:
            cat, resolved = oracle_category(l[inA], u[inA])
            why = ""
        # coverage of the oracle delta* by the endpoint bands, per member of A:
        #   plug-in: oracle mean inside the band (None if either is unavailable);
        #   definite: oracle interval inside ("covered") / disjoint ("uncovered") / overlapping ("uncertain");
        #   unassessable entries are labelled "band_unavailable" or "oracle_unavailable", never "uncertain"
        cov, cov_def = [], []
        for i, m in enumerate(A):
            j = omids.index(m)
            blo, bhi = bd["lo"][i], bd["hi"][i]
            band_ok = bool(np.isfinite(blo) and np.isfinite(bhi))
            cov.append(bool(blo <= mean[j] <= bhi) if np.isfinite(mean[j]) and band_ok else None)
            if not band_ok:
                cov_def.append("band_unavailable")
            elif not (np.isfinite(l[j]) and np.isfinite(u[j])):
                cov_def.append("oracle_unavailable")
            elif blo <= l[j] and u[j] <= bhi:
                cov_def.append("covered")
            elif u[j] < blo or l[j] > bhi:
                cov_def.append("uncovered")
            else:
                cov_def.append("uncertain")
        out.update(oracle=dict(mids=omids, dk=dk.tolist(), mean=mean.tolist(), sd=sd.tolist(), lo=l.tolist(),
                               hi=u.tolist(), category=cat, resolved=resolved, reason=why, inA=inA.tolist()),
                   coverage_plugin=cov, coverage_definite=cov_def, coverage_mids=list(A),
                   coverage_all_plugin=(None if (not cov or any(c is None for c in cov)) else bool(all(cov))),
                   false_eq=bool(resolved and dec == "EQUIVALENCE" and cat != "EQUIVALENCE"),
                   false_mm=bool(resolved and dec == "MISMATCH" and cat != "MISMATCH"))
        tm["oracle_s"] = time.perf_counter() - t4
        if log:
            log(f"[{truth_name} r{rep}] oracle={cat} resolved={resolved} {tm['oracle_s']:.0f}s")
    tm["total_s"] = time.perf_counter() - t0
    out["timings"] = tm
    return out


def _support(red, cond, scale):
    P = len(red["elig"])
    frac = red["elig"].mean() if P else 0.0
    n812 = float(red["n812"][red["elig"]].sum())
    thr = EP.N_812_MIN * scale
    ok = (frac >= EP.P_ELIG_MIN) and (n812 >= thr) and (cond < EP.COND_MAX)
    nc = red.get("n_contrib")
    return dict(ok=bool(ok), frac_elig=float(frac), n812=n812, n812_min=thr, cond=float(cond),
                n_elig=int(red["elig"].sum()),
                contrib_frac=(float(nc.mean() / red["R"]) if nc is not None and P else None))


# --------------------------------------------------------------------------------------- summary
def summarise(reps):
    """Rates per truth with Wilson CIs."""
    out = {}
    for tn in TRUTHS:
        rr = [r for r in reps if r["truth"] == tn]
        N = len(rr)
        if not N:
            continue
        cnt = lambda f: int(sum(1 for r in rr if f(r)))   # noqa: E731
        d = dict(N=N)
        if tn in IN_FAMILY:
            k = cnt(lambda r: r["retained"])
            d["retention"] = [k / N, wilson(k, N)]
        for lab, f in [("calibration_failure", lambda r: r["decision"] == "CALIBRATION FAILURE"),
                       ("support_failure", lambda r: r.get("tag") == "support"),
                       ("conclusive", lambda r: r["decision"] in ("EQUIVALENCE", "MISMATCH")),
                       ("equivalence", lambda r: r["decision"] == "EQUIVALENCE"),
                       ("mismatch", lambda r: r["decision"] == "MISMATCH")]:
            k = cnt(f)
            d[lab] = [k / N, wilson(k, N)]
        # false-decision bounds over all N replicates: calibration failures and replicates without a resolved
        # oracle contribute 0 / U respectively; lower = F/N, upper = (F + U)/N with U = unresolved or missing oracle
        st = [("calibration_failure" if r["decision"] == "CALIBRATION FAILURE" else
               "resolved" if r.get("oracle", {}).get("resolved") else
               "unresolved" if "oracle" in r else "not_run") for r in rr]
        d["oracle_status"] = {k: st.count(k) for k in ("resolved", "unresolved", "not_run", "calibration_failure")}
        U = st.count("unresolved") + st.count("not_run")
        res = [r for r, s_ in zip(rr, st) if s_ == "resolved"]
        for lab, key in [("false_equivalence", "false_eq"), ("false_mismatch", "false_mm")]:
            F = sum(bool(r[key]) for r in res)
            d[lab + "_cond_resolved"] = [F / len(res) if res else None, wilson(F, len(res))]
            d[lab + "_bounds"] = [F / N, (F + U) / N]
            # pointwise Wilson intervals of the two endpoints (not a joint confidence region)
            d[lab + "_bounds_wilson"] = [wilson(F, N), wilson(F + U, N)]
        d["unresolved"] = [U / N, wilson(U, N)]
        # band coverage of the oracle delta*: plug-in (oracle mean) and definite bounds from the oracle intervals,
        # per member and pooled; pooled CIs resample whole replicates (members within a replicate are dependent)
        orc = [r for r in rr if "oracle" in r]
        ASSESS = ("covered", "uncovered", "uncertain")
        per_m = {}
        for r in orc:
            for m, cp, cd in zip(r["coverage_mids"], r["coverage_plugin"], r["coverage_definite"]):
                e = per_m.setdefault(str(m), dict(n=0, plugin=0, plugin_n=0, covered=0, uncovered=0, assessed=0,
                                                  band_unavailable=0, oracle_unavailable=0))
                e["n"] += 1
                if cp is not None:
                    e["plugin_n"] += 1
                    e["plugin"] += int(cp)
                e["covered"] += int(cd == "covered")
                e["uncovered"] += int(cd == "uncovered")
                e["assessed"] += int(cd in ASSESS)
                e["band_unavailable"] += int(cd == "band_unavailable")
                e["oracle_unavailable"] += int(cd == "oracle_unavailable")
        # per member: plug-in coverage with its own denominator and Wilson CI; definite bounds conditional on
        # assessable entries (band and oracle interval both available), reported with the band-failure frequency
        d["band_coverage_per_member"] = {
            m: dict(n=e["n"], plugin_n=e["plugin_n"],
                    plugin=(e["plugin"] / e["plugin_n"] if e["plugin_n"] else None),
                    plugin_wilson=wilson(e["plugin"], e["plugin_n"]),
                    assessed_n=e["assessed"],
                    definite_bounds=([e["covered"] / e["assessed"], 1 - e["uncovered"] / e["assessed"]]
                                     if e["assessed"] else None),
                    band_unavailable_frac=e["band_unavailable"] / e["n"],
                    oracle_unavailable_frac=e["oracle_unavailable"] / e["n"])
            for m, e in per_m.items()}

        def _pooled(rs):
            num = sum(sum(c is True for c in r["coverage_plugin"]) for r in rs)
            den = sum(sum(c is not None for c in r["coverage_plugin"]) for r in rs)
            n = sum(sum(c in ASSESS for c in r["coverage_definite"]) for r in rs)
            lo = sum(r["coverage_definite"].count("covered") for r in rs)
            hi = n - sum(r["coverage_definite"].count("uncovered") for r in rs)
            return (num / den if den else np.nan), (lo / n if n else np.nan), (hi / n if n else np.nan)
        if orc:
            pt = _pooled(orc)
            rng = np.random.default_rng([C.SEED_BASE, 4141, TRUTHS.index(tn)])
            bs = np.array([_pooled([orc[i] for i in rng.integers(0, len(orc), len(orc))]) for _ in range(2000)])

            def ci(j):
                v = bs[:, j][np.isfinite(bs[:, j])]
                return np.percentile(v, [2.5, 97.5]).tolist() if len(v) else None
            nb_unav = sum(sum(c == "band_unavailable" for c in r["coverage_definite"]) for r in orc)
            n_all = sum(len(r["coverage_definite"]) for r in orc)
            d["band_coverage_pooled"] = dict(plugin=[pt[0], ci(0)], definite_lower=[pt[1], ci(1)],
                                             definite_upper=[pt[2], ci(2)], n_reps=len(orc),
                                             band_unavailable_frac=(nb_unav / n_all if n_all else None))
            sp = [r.get("coverage_all_plugin") for r in orc]
            sa = [x for x in sp if x is not None]
            k = sum(bool(x) for x in sa)
            d["simultaneous_coverage_plugin"] = dict(rate=(k / len(sa) if sa else None), wilson=wilson(k, len(sa)),
                                                     n_assessed=len(sa), n_unavailable=len(sp) - len(sa))
            ra = [r for r in orc if r["coverage_definite"] and all(c in ASSESS for c in r["coverage_definite"])]
            k = sum(all(c == "covered" for c in r["coverage_definite"]) for r in ra)
            k2 = sum(not any(c == "uncovered" for c in r["coverage_definite"]) for r in ra)
            d["simultaneous_coverage_definite_bounds"] = dict(
                bounds=([k / len(ra), k2 / len(ra)] if ra else None), n_assessed=len(ra),
                n_unavailable=len(orc) - len(ra))
        amb_all = [r.get("ambiguity") for r in rr]
        amb = [x for x in amb_all if x is not None and np.isfinite(x)]
        d["ambiguity_q"] = np.quantile(amb, [0.1, 0.5, 0.9]).tolist() if amb else None
        d["ambiguity_n_excluded"] = len(amb_all) - len(amb)
        d["A_size_q"] = np.quantile([len(r["A"]) for r in rr], [0.1, 0.5, 0.9]).tolist()
        out[tn] = d
    return out
