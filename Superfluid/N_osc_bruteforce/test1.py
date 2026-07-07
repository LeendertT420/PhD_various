from equations import *
import numpy as np
import matplotlib.pyplot as plt

deltas = np.linspace(-5, 0, 100)
alphas = np.linspace(0, 2, 100)
N=10

#plt.plot(deltas, upper_boundary(N, deltas))
#plt.plot(deltas, lower_boundary(N, deltas))

def upper_boundary_inverse(N, alpha):
    # handle invalid inputs early
    if not np.isfinite(N) or not np.isfinite(alpha):
        return np.nan

    coeffs = [1, 2*N*alpha, 2, 0, 1]
    roots = np.roots(coeffs)

    # ---- CLEAN ROOTS ----
    # remove NaN / Inf entries
    roots = np.array([
        r for r in roots
        if np.isfinite(r.real) and np.isfinite(r.imag)
    ])

    if len(roots) == 0:
        return np.nan

    # keep only real roots (within tolerance)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real

    if len(real_roots) == 0:
        return np.nan

    # physical branch constraint: t > 0 (and typically >= sqrt(3))
    candidates = real_roots[real_roots > 0]

    if len(candidates) == 0:
        return np.nan

    # back-substitution check (most reliable selector)
    def alpha_check(t):
        try:
            delta = 0.5 * (t + 3/t)
            if not np.isfinite(delta):
                return np.inf
            s2 = delta*delta - 3
            if s2 < 0:
                return np.inf
            s = np.sqrt(s2)

            val = (2/(27*N)) * (2*delta + s)**2 * (s - delta)

            if not np.isfinite(val):
                return np.inf

            return abs(val - alpha)

        except:
            return np.inf

    best = min(candidates, key=alpha_check)

    # final δ reconstruction
    delta = 0.5 * (best + 3/best)

    return delta

I = []
for alpha in alphas:
    I.append(upper_boundary_inverse(N, alpha))


plt.plot(I, alphas)
plt.show()