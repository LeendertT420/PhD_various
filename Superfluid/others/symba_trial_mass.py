import sympy as sp
from sympy import symbols, Matrix, Poly, resultant, simplify, factor, collect

# 1. Define variables
m, g1, g2, L, t = symbols('m g1 g2 L t', real=True)
lam, x = symbols('lam x') # lam is for char poly, x represents omega**2

# 2. Define the Matrix
J = Matrix([
    [0, 1, 0, 0, 0],
    [-1, -g1, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, -m, -g2, m],
    [L/t, 0, L/t, 0, -1/t]
])

# characteristic polynomial
char_poly = J.charpoly(lam).as_expr()

# coefficients
coeffs = sp.Poly(char_poly, lam).all_coeffs()
a1,a2,a3,a4,a5,a6 = coeffs

# equations in x = omega^2
eq1 = a2*x**2 - a4*x + a6
eq2 = a1*x**2 - a3*x + a5

# eliminate x
res = sp.resultant(eq1, eq2, x)

# clean rational structure
res = sp.together(res)
num, den = sp.fraction(res)

# polynomial in L
P = sp.Poly(num, L).as_expr()

# discriminant wrt L
disc = sp.discriminant(P, L)

disc = sp.factor(disc)

print(disc)