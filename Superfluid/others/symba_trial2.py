import sympy as sp
import numpy as np



def derive_threshold_polynomials(mu_vals, gamma_vals, tau):
    """
    Derive the omega^2 polynomial equation:
    
        g_Re*h_Im - g_Im*h_Re = 0
    
    and solve for omega^2.
    """

    N = len(mu_vals)

    # Symbols
    lam = sp.symbols('lambda')
    omega = sp.symbols('omega', real=True)
    x = sp.symbols('x', real=True)   # x = omega^2

    # Construct J0
    J0 = construct_J0(mu_vals, gamma_vals, tau)

    dim = J0.shape[0]

    # ------------------------------------------------------------
    # Characteristic polynomial g(lambda)
    # ------------------------------------------------------------

    M = J0 - lam * sp.eye(dim)

    g = sp.expand(M.det())

    # ------------------------------------------------------------
    # Construct B(lambda)
    #
    # Remove:
    #   row = last row
    #   col = first column
    # ------------------------------------------------------------

    B = M.copy()

    B.row_del(dim - 1)
    B.col_del(0)

    h = sp.expand(B.det() / tau)

    # ------------------------------------------------------------
    # Substitute lambda = i*omega
    # ------------------------------------------------------------

    gi = sp.expand(g.subs(lam, sp.I * omega))
    hi = sp.expand(h.subs(lam, sp.I * omega))

    # ------------------------------------------------------------
    # Separate real and imaginary parts
    # ------------------------------------------------------------

    g_re = sp.expand(sp.re(gi))
    g_im = sp.expand(sp.im(gi) / omega)

    h_re = sp.expand(sp.re(hi))
    h_im = sp.expand(sp.im(hi) / omega)

    # ------------------------------------------------------------
    # Convert to polynomials in omega^2
    # ------------------------------------------------------------

    substitutions = {
        omega**2: x
    }

    g_re = sp.expand(g_re.subs(substitutions))
    g_im = sp.expand(g_im.subs(substitutions))

    h_re = sp.expand(h_re.subs(substitutions))
    h_im = sp.expand(h_im.subs(substitutions))

    # ------------------------------------------------------------
    # Elimination polynomial
    # ------------------------------------------------------------

    threshold_poly = sp.expand(g_re*h_im - g_im*h_re)

    threshold_poly = sp.collect(threshold_poly, x)

    # Convert to explicit polynomial in x
    poly_x = sp.Poly(threshold_poly, x)

    # Solve for omega^2
    omega2_solutions = sp.solve(poly_x, x)

    return {
        "g(lambda)": g,
        "h(lambda)": h,
        "g_Re(x)": g_re,
        "g_Im(x)": g_im,
        "h_Re(x)": h_re,
        "h_Im(x)": h_im,
        "threshold_polynomial": threshold_poly,
        "omega2_solutions": omega2_solutions,
    }


# ============================================================
# Example
# ============================================================

mu_vals = [1.0, 1.2, 0.9]
gamma_vals = [0.5, 0.6, 0.4]

tau = sp.Rational(3, 2)

result = derive_threshold_polynomials(
    mu_vals,
    gamma_vals,
    tau
)

print("\n=== g_Re(omega^2) ===")
sp.pprint(result["g_Re(x)"])

print("\n=== g_Im(omega^2) ===")
sp.pprint(result["g_Im(x)"])

print("\n=== h_Re(omega^2) ===")
sp.pprint(result["h_Re(x)"])

print("\n=== h_Im(omega^2) ===")
sp.pprint(result["h_Im(x)"])

print("\n=== Threshold polynomial ===")
sp.pprint(result["threshold_polynomial"])

print("\n=== omega^2 solutions ===")
sp.pprint(result["omega2_solutions"])