from sympy import symbols, expand

# Define symbols
g1, g2, r, t, s = symbols('gamma1 gamma2 rho tau sigma')

# Define parameters
a1 = t
a2 = 1 + t * (g1 + g2)
a3 = g1 + g2 + t * (s + g1 * g2)
A4 = s + g1 * g2 + t * (g1 / r + g2 * r)
A5 = g1 / r + g2 * r + t
A6 = 1
S = g1 + g2
Z = 2 * t

# Intermediate Variables
U = a2 * A5 - a1 * A6
V = a2 * S - a1 * s
W = a2 * a3 - a1 * A4

# Coefficients
A = -S * Z**2
B = a1 * V**2 - a3 * V * Z + A5 * Z**2 + 2 * S * W * Z
C = -2 * a1 * U * V + a3 * (U * Z + V * W) - 2 * A5 * W * Z - S * W**2
D = a1 * U**2 - a3 * U * W + A5 * W**2

# To see the fully expanded form of a coefficient:
print(expand(A))
print(expand(B))