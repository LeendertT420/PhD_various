import sympy as sp
from sympy import symbols, Matrix, Poly, resultant, simplify, factor, collect

# 1. Define variables
rho, gamma1, gamma2, L, tau = symbols('rho gamma1 gamma2 L tau', real=True)
lam, x = symbols('lam x') # lam is for char poly, x represents omega**2

# 2. Define the Matrix
J = Matrix([
    [0, 1, 0, 0, 0],
    [-rho, -gamma1, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, -1/rho, -gamma2, 1],
    [L/tau, 0, L/tau, 0, -1/tau]
])

# 3. Calculate Characteristic Polynomial
char_poly = (lam * Matrix.eye(5) - J).det()

# 4. Extract Coefficients a1...a6
# P(lambda) = a1*lambda^5 + a2*lambda^4 + a3*lambda^3 + a4*lambda^2 + a5*lambda + a6
coeffs = Poly(char_poly, lam).all_coeffs()
a1, a2, a3, a4, a5, a6 = coeffs

# 5. Define Stability Equations (for lambda = i*omega)
# Real part (omega^4, omega^2, constant): a2*x^2 - a4*x + a6 = 0
# Imag part (omega^4, omega^2, constant): a1*x^2 - a3*x + a5 = 0
eq1 = a2*x**2 - a4*x + a6
eq2 = a1*x**2 - a3*x + a5

# 6. Resultant: Eliminate x (omega**2)
res = resultant(eq1, eq2, x)
print(res)
# 7. Collect coefficients by powers of L
# The resultant is a polynomial in L, e.g., C3*L^3 + C2*L^2 + C1*L + C0 = 0

# 2. Extract and format coefficients
res_poly = Poly(res, L)
coeffs_L = res_poly.all_coeffs()

print("--- Readable Coefficients ---")
for i, coeff in enumerate(coeffs_L):
    power = len(coeffs_L) - 1 - i
    # Apply factor() and then collect() to make it look clean
    clean_coeff = factor(coeff)
    print(f"C{power} (L^{power}):")
    print(clean_coeff)
    print("-" * 30)