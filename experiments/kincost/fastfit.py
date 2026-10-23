"""Fast exactly-determined per-axis noise fit (2 parameters -> median and IQR of log RMS-S2S).

Box-constrained Levenberg-Marquardt with a central-difference Jacobian, JIT-compiled with numba when it
is available (pure-Python fallback otherwise, same arithmetic).  The objective is the proposal's squared
standardised error; only the optimiser differs from L-BFGS-B (implementation choice recorded in PREREG).
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:          # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]


@njit(cache=True)
def _pct_sorted(v, n, q):
    # numpy 'linear' percentile on the first n entries of sorted v
    if n == 0:
        return np.nan
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    w = pos - lo
    return v[lo] * (1 - w) + v[hi] * w


@njit(cache=True)
def axis_stats(a, b, c, z, mu, s):
    P, R = a.shape
    med = 0.0
    iqr = 0.0
    buf = np.empty(P)
    for r in range(R):
        n = 0
        for p in range(P):
            sig = math.exp(mu + s * z[p])
            q = a[p, r] + 2.0 * sig * b[p, r] + sig * sig * c[p, r]
            if q > 0 and not math.isnan(q):
                buf[n] = 0.5 * math.log(q)
                n += 1
        if n == 0:
            return np.nan, np.nan
        v = np.sort(buf[:n])
        med += _pct_sorted(v, n, 0.5)
        iqr += _pct_sorted(v, n, 0.75) - _pct_sorted(v, n, 0.25)
    return med / R, iqr / R


@njit(cache=True)
def _resid(a, b, c, z, x0, x1, tm, ti, sm, si):
    m, i = axis_stats(a, b, c, z, x0, x1)
    return (m - tm) / sm, (i - ti) / si


@njit(cache=True)
def lm_box(a, b, c, z, tm, ti, sm, si, x0, x1, lo0, hi0, lo1, hi1, maxit=100):
    r0, r1 = _resid(a, b, c, z, x0, x1, tm, ti, sm, si)
    if not (math.isfinite(r0) and math.isfinite(r1)):
        return x0, x1, 1e12
    f = r0 * r0 + r1 * r1
    lam = 1e-3
    h = 1e-5
    for it in range(maxit):
        # central-difference Jacobian (one-sided at the box)
        J = np.empty((2, 2))
        for j in range(2):
            xp0, xp1, xm0, xm1 = x0, x1, x0, x1
            if j == 0:
                xp0 = min(x0 + h, hi0)
                xm0 = max(x0 - h, lo0)
                d = xp0 - xm0
            else:
                xp1 = min(x1 + h, hi1)
                xm1 = max(x1 - h, lo1)
                d = xp1 - xm1
            p0, p1 = _resid(a, b, c, z, xp0, xp1, tm, ti, sm, si)
            m0, m1 = _resid(a, b, c, z, xm0, xm1, tm, ti, sm, si)
            J[0, j] = (p0 - m0) / d
            J[1, j] = (p1 - m1) / d
        if not (math.isfinite(J[0, 0]) and math.isfinite(J[0, 1]) and math.isfinite(J[1, 0])
                and math.isfinite(J[1, 1])):
            break
        g0 = J[0, 0] * r0 + J[1, 0] * r1
        g1 = J[0, 1] * r0 + J[1, 1] * r1
        A00 = J[0, 0] ** 2 + J[1, 0] ** 2
        A11 = J[0, 1] ** 2 + J[1, 1] ** 2
        A01 = J[0, 0] * J[0, 1] + J[1, 0] * J[1, 1]
        improved = False
        for _ in range(12):
            B00 = A00 * (1 + lam) + 1e-12
            B11 = A11 * (1 + lam) + 1e-12
            det = B00 * B11 - A01 * A01
            if det == 0:
                lam *= 10
                continue
            d0 = -(B11 * g0 - A01 * g1) / det
            d1 = -(-A01 * g0 + B00 * g1) / det
            n0 = min(max(x0 + d0, lo0), hi0)
            n1 = min(max(x1 + d1, lo1), hi1)
            q0, q1 = _resid(a, b, c, z, n0, n1, tm, ti, sm, si)
            if math.isfinite(q0) and math.isfinite(q1):
                fn = q0 * q0 + q1 * q1
                if fn < f:
                    df = f - fn
                    x0, x1, r0, r1, f = n0, n1, q0, q1, fn
                    lam = max(lam / 3, 1e-9)
                    improved = True
                    break
            lam *= 4
        if not improved or f < 1e-20:
            break
        if df < 1e-14 * (1 + f):
            break
    return x0, x1, f
