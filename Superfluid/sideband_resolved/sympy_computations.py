import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import numpy as np

# Define Jacobian
print("--- Compute characteristic polynomial ---")
lambda_ = sp.Symbol('lambda')
x, alpha, gamma, nu, delta, delta_eff = sp.symbols('x alpha gamma nu delta delta_eff', positive=True)

v_star = -sp.sqrt(x * (1 - x / alpha))

J = sp.Matrix([
    [0, 1, 0, 0],
    [-1, -gamma, 2 * x / sp.sqrt(alpha), 2 * v_star],
    [-v_star / nu, 0, -1 / nu, -(delta + x) / nu],
    [x / (nu * sp.sqrt(alpha)), 0, (delta + x) / nu, -1 / nu]
])
print(J)

# Compute characteristic polynomial: det(J - lambda*I) = 0
I = sp.eye(4)
raw_poly = (J - lambda_ * I).det()

# Scale by nu^2 to force c_4 = nu^2 and clear out the 1/nu fractions
scaled_poly = sp.expand(raw_poly * (nu**2))
poly_collected = sp.collect(scaled_poly, lambda_)

c4 = nu**2
c3 = poly_collected.coeff(lambda_, 3)
c2 = poly_collected.coeff(lambda_, 2)
c1 = poly_collected.coeff(lambda_, 1)
c0 = poly_collected.subs(lambda_, 0)

print(f"c4 = {sp.simplify(c4)}")
print(f"c3 = {sp.simplify(c3)}")
print(f"c2 = {sp.simplify(c2)}")
print(f"c1 = {sp.simplify(c1)}")
print(f"c0 = {sp.simplify(c0)}")


print("--- Routh-Hurwitz Condition ---")
# Compute the requested Routh-Hurwitz boundary condition expression
rh_cond = c4*c1**2 - (c1 * c2 * c3) + (c0 * c3**2)
rh_cond = sp.simplify(rh_cond)
print("--- Step 3: Extracting Polynomial Coefficients in terms of x^* ---")
# Convert the expression into a formal SymPy Polynomial in terms of x
# This guarantees precise extraction of each coefficient b_i
rh_poly = sp.Poly(rh_cond, x)

# The degrees will depend on the expansion, let's pull them dynamically
coeffs_dict = rh_poly.as_dict()

# Print out the non-zero coefficients matching their powers: b_i * (x^*)^i
for power, coeff in sorted(coeffs_dict.items(), reverse=True):
    # power is a tuple, e.g., (4,) for x^4
    p = power[0]
    print(f"b_{p} (coefficient of x^{p}) =")
    print(sp.simplify(coeff))
    print("-" * 50)

y = sp.Symbol('y')

# Substitute x = y - delta
rh_cond_y = sp.expand(rh_cond.subs(x, y - delta))

# Construct polynomial in y = x + delta
rh_poly_y = sp.Poly(rh_cond_y, y)

print(rh_poly_y)
print(rh_poly_y.all_coeffs())

coeffs_dict = rh_poly_y.as_dict()

# Print out the non-zero coefficients matching their powers: a_i * (x^*+\delta)^i
for power, coeff in sorted(coeffs_dict.items(), reverse=True):
    # power is a tuple, e.g., (4,) for x^4
    p = power[0]
    print(f"a_{p} (coefficient of (x+delta)^{p}) =")
    print(sp.simplify(coeff))
    print("-" * 50)