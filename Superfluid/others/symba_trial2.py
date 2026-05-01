import numpy as np
from sympy import symbols, Matrix, lambdify

# =========================================================
# 1. SYMBOLIC DEFINITIONS
# =========================================================

rho, gamma1, gamma2, tau, dL = symbols(
    'rho gamma1 gamma2 tau dL', real=True
)

# Jacobian (IMPORTANT: uses dL = ∂L at fixed point)
J = Matrix([
    [0, 1, 0, 0, 0],
    [-rho, -gamma1, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, -1/rho, -gamma2, 1],
    [dL/tau, 0, dL/tau, 0, -1/tau]
])

# =========================================================
# 2. NUMERICAL JACOBIAN BUILDER
# =========================================================

def J_numeric(rho_val, g1, g2, tau_val, dL_val):
    return np.array([
        [0, 1, 0, 0, 0],
        [-rho_val, -g1, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, -1/rho_val, -g2, 1],
        [dL_val/tau_val, 0, dL_val/tau_val, 0, -1/tau_val]
    ], dtype=float)

# =========================================================
# 3. NONLINEAR GAIN SLOPE (THIS IS WHAT MATTERS)
# =========================================================

def dL_value(alpha, delta, x_plus):
    """
    derivative of L(x+) at fixed point
    """
    denom = (delta + x_plus)**2 + 1
    return -2 * alpha * (delta + x_plus) / (denom**2)

# =========================================================
# 4. EIGENVALUE ANALYSIS
# =========================================================

def eigenvalues(rho_val, g1, g2, tau_val, dL_val):
    Jmat = J_numeric(rho_val, g1, g2, tau_val, dL_val)
    return np.linalg.eigvals(Jmat)

# =========================================================
# 5. HOPF DETECTION (CORRECT CRITERION)
# =========================================================

def is_hopf(eigs, tol=1e-6):
    """
    Hopf if a complex conjugate pair is ~pure imaginary
    """
    count = 0
    for ev in eigs:
        if abs(ev.real) < tol and abs(ev.imag) > tol:
            count += 1
    return count >= 2

# =========================================================
# 6. THRESHOLD SEARCH IN α
# =========================================================

def find_lasing_threshold(alpha_values, x_plus, rho, gamma1, gamma2, tau, delta):

    for alpha in alpha_values:

        # compute slope of gain at operating point
        dL = dL_value(alpha, delta, x_plus)

        # compute eigenvalues
        eigs = eigenvalues(rho, gamma1, gamma2, tau, dL)

        # check Hopf condition
        if is_hopf(eigs):

            print("\n=== HOPF DETECTED ===")
            print("alpha =", alpha)
            print("eigenvalues =", eigs)
            print("dL =", dL)

            return alpha

    return None

# =========================================================
# 7. OPTIONAL: RUN EXAMPLE
# =========================================================

if __name__ == "__main__":

    alpha_vals = np.linspace(0.1, 50, 500)

    threshold = find_lasing_threshold(
        alpha_vals,
        x_plus=1.0,     # operating point
        rho=1.5,
        gamma1=0.5,
        gamma2=0.5,
        tau=2.0,
        delta=0.2
    )

    print("\nFinal threshold:", threshold)