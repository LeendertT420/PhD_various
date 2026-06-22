import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import numpy as np

# 1. Define all necessary symbols
# lambda_ is the eigenvalue variable
# x, alpha, gamma, nu, delta, delta_eff are physical parameters
lambda_ = sp.Symbol('lambda')
x, alpha, gamma, nu, delta, delta_eff = sp.symbols('x alpha gamma nu delta delta_eff', positive=True)

v_star = sp.sqrt(x * (1 - x / alpha))

# Base Jacobian Matrix
J = sp.Matrix([
    [0, 1, 0, 0],
    [-1, -gamma, 2 * x / sp.sqrt(alpha), 2 * v_star],
    [-v_star / nu, 0, -1 / nu, -(delta + x) / nu],
    [x / (nu * sp.sqrt(alpha)), 0, (delta + x) / nu, -1 / nu]
])

# Compute raw characteristic polynomial: det(J - lambda*I) = 0
I = sp.eye(4)
raw_poly = (J - lambda_ * I).det()

# Scale by nu^2 to force c_4 = nu^2 and clear out the 1/nu fractions
scaled_poly = sp.expand(raw_poly * (nu**2))

# Align standard polynomial sign conventions for lambda^4
# (Since det(J-l*I) for 4x4 naturally starts with +lambda^4)
poly_collected = sp.collect(scaled_poly, lambda_)

c4 = nu**2
c3 = poly_collected.coeff(lambda_, 3)
c2 = poly_collected.coeff(lambda_, 2)
c1 = poly_collected.coeff(lambda_, 1)
c0 = poly_collected.subs(lambda_, 0)

# Display beautifully simplified coefficients
print(f"c4 = {sp.simplify(c4)}")
print(f"c3 = {sp.simplify(c3)}")
print(f"c2 = {sp.simplify(c2)}")
print(f"c1 = {sp.simplify(c1)}")
print(f"c0 = {sp.simplify(c0)}")


print("--- Step 2: Routh-Hurwitz Condition ---")
# Compute the requested Routh-Hurwitz boundary condition expression
rh_cond = c1 - (c1 * c2 * c3) + (c0 * c3**2)
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

# Print out the non-zero coefficients matching their powers: b_i * (x^*)^i
for power, coeff in sorted(coeffs_dict.items(), reverse=True):
    # power is a tuple, e.g., (4,) for x^4
    p = power[0]
    print(f"b_{p} (coefficient of (x+delta)^{p}) =")
    print(sp.simplify(coeff))
    print("-" * 50)

print("\n--- Step 4: Discriminant and Solving for nu ---")
# To find the discriminant safely, we build the polynomial expression explicitly
# from the extracted coefficients to drop any residual square roots if necessary
poly_expr = rh_poly_y.as_expr()

# Compute the discriminant of this polynomial with respect to x
disc = sp.discriminant(poly_expr, y)
print("Discriminant computed. Simplifying...")
disc_simplified = sp.simplify(disc)
print(disc_simplified)

# 1. Take the partial derivative of the discriminant with respect to delta
d_disc_d_delta = sp.diff(disc, delta)

# 2. Compute the Resultant to eliminate 'delta'
# This finds the polynomial constraint where both equations share a root
print("Computing the resultant to eliminate delta... (This may take a minute due to the high degree)")
poly_nu_min = sp.resultant(disc, d_disc_d_delta, delta)

# 3. Clean up and display the implicit polynomial expression for nu_min(gamma)
poly_nu_min_clean = sp.factor(poly_nu_min)
print("The polynomial expression defining nu_min(gamma) = 0 is:")
print(poly_nu_min_clean)
sp.pprint(poly_nu_min_clean)

coef, factors = sp.factor_list(poly_nu_min_clean)

relevant_factors = []
for factor, power in factors:
    # Filter out trivial monomial terms like gamma or nu, and non-physical roots
    if factor == gamma or factor == nu or factor == (gamma*nu + 2):
        continue
    relevant_factors.append(factor)
    print(f"\nFound Relevant Branch Polynomial (Power {power}):")
    print(factor)
    sp.pprint(factor)

print("\n" + "="*50)
print("--- Step 5: SymPy Verification & Limit Auditing ---")
print("="*50)

# Let's pull the primary relevant polynomial factor found by your resultant
# (Assuming 'factor' holds the large multi-variable expression in gamma and nu)
for factor, _ in factors:
    if factor not in [gamma, nu, gamma*nu + 2]:
        target_poly = factor
        break

print("\n1. Testing the direct substitution of gamma = 0 into the master polynomial:")
poly_at_gamma_0 = target_poly.subs(gamma, 0)
print(f"Polynomial at gamma=0: {poly_at_gamma_0} = 0")
roots_direct = sp.solve(poly_at_gamma_0, nu)
print(f"Direct roots for nu at gamma=0: {roots_direct}")

print("\n2. Performing leading-order asymptotic analysis as gamma -> 0:")
# We treat gamma as a small perturbation variable and take a Series expansion 
# of the master condition around gamma = 0 to capture the true boundary behavior.
try:
    # Expand the polynomial up to first order in gamma
    series_expansion = sp.series(target_poly, gamma, 0, 2).removeO()
    print(f"Series expansion near gamma=0:\n{series_expansion} = 0")
    
    # Isolate the leading-order coefficient driving the perturbation branch
    leading_coef = sp.collect(series_expansion, gamma)
    print(f"\nCollected by gamma:\n{leading_coef}")
    
except Exception as e:
    print(f"Could not compute direct series expansion due to complexity: {e}")

print("\n3. Verifying the specific cbrt(1/2) condition branch:")
# Let's explicitly test if the root nu = (1/2)**(1/3) balances the system
nu_target = (sp.Rational(1, 2))**(sp.Rational(1, 3))
substituted_target = target_poly.subs(nu, nu_target)
print(f"Substituting nu = (1/2)^(1/3) into the master polynomial yields:")
print(sp.simplify(substituted_target))

# Check if gamma factors out entirely when nu = cbrt(1/2)
print("\nCan gamma be factored out at this critical nu value?")
sp.pprint(sp.factor(substituted_target))

nu_sym, gamma_sym = sp.symbols('nu gamma', positive=True)

# Replace 'poly_nu_min_clean' with your actual resultant polynomial object
poly_func = sp.lambdify((nu_sym, gamma_sym), poly_nu_min_clean, 'numpy')

# --- Step 2: Set up the Logarithmic Gamma grid ---
# This creates 300 points distributed logarithmically from 10^-3 to 10^1
gamma_vals = np.logspace(-5, 0, 300)
nu_min_vals = []

def objective_func(nu, gamma):
    return poly_func(nu, gamma)

# --- Step 3: Track the root for each gamma ---
# Broad initial bracket search space for the starting point
current_guess_bracket = [0.001, 10.0] 

for g in gamma_vals:
    try:
        root = brentq(objective_func, current_guess_bracket[0], current_guess_bracket[1], args=(g,))
        nu_min_vals.append(root)
        
        # Dynamic bracket tracking to follow the curve continuously
        current_guess_bracket = [max(1e-5, root - 0.3), root + 0.3]
    except ValueError:
        # If the bracket fails, broaden the search area temporarily
        try:
            root = brentq(objective_func, 1e-5, 50.0, args=(g,))
            nu_min_vals.append(root)
            current_guess_bracket = [max(1e-5, root - 0.3), root + 0.3]
        except ValueError:
            nu_min_vals.append(np.nan)

nu_min_vals = np.array(nu_min_vals)

# --- Step 4: Plot the Curve with Logarithmic X-Axis ---
plt.figure(figsize=(8, 5))
plt.plot(gamma_vals, nu_min_vals, lw=2.5, color='darkblue', label=r'$\nu_{min}(\gamma)$')

# Enforce logarithmic scaling on the X-axis
plt.xscale('log')

plt.title(r'Minimal $\Omega/\kappa$ for limit cycles', fontsize=12)
plt.xlabel(r'$\Gamma/\Omega$', fontsize=11)
plt.ylabel(r'$\Omega/\kappa$', fontsize=11)

# Add minor grid lines for clear reading across log decades
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.xlim(gamma_vals.min(), gamma_vals.max())
#plt.legend(loc='best')

plt.tight_layout()
plt.show()