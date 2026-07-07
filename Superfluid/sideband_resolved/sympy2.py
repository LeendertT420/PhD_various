import sympy as sp

# 1. Setup variables
alpha, delta, gamma, nu = sp.symbols('alpha delta gamma nu', positive=True)
A2 = sp.symbols('A2', positive=True) # Work with A^2 directly
omega = sp.symbols('omega', real=True)
x0, u0, v0 = sp.symbols('x0 u0 v0', real=True)
Uc, Us, Vc, Vs = sp.symbols('Uc Us Vc Vs', real=True)

# 2. Define the 8 equations derived from Harmonic Balance
# We define them as f = 0
eqs = [
    x0 - (u0**2 + v0**2 + 0.5*(Uc**2 + Us**2 + Vc**2 + Vs**2)),
    (1 - omega**2)*sp.sqrt(A2) - 2*(u0*Uc + v0*Vc),
    (-gamma*omega)*sp.sqrt(A2) - 2*(u0*Us + v0*Vs),
    u0 + (delta + x0)*v0 + 0.5*sp.sqrt(A2)*Vc - sp.sqrt(alpha),
    v0 - (delta + x0)*u0 - 0.5*sp.sqrt(A2)*Uc,
    Uc + nu*omega*Us + (delta + x0)*Vc + v0*sp.sqrt(A2),
    Us - nu*omega*Uc + (delta + x0)*Vs,
    Vc + nu*omega*Vs - (delta + x0)*Uc - u0*sp.sqrt(A2)
]

# 3. Use Groebner Basis to eliminate variables (x0, u0, v0, Uc, Us, Vc, Vs, omega)
# We want to solve for A2 in terms of parameters.
# Note: This may be computationally heavy. 
vars_to_eliminate = [x0, u0, v0, Uc, Us, Vc, Vs, omega]
basis = sp.groebner(eqs, vars_to_eliminate, order='lex')

# The last element of the basis will be the polynomial purely in A2 and parameters
poly_A2 = basis[-1]

print("Polynomial in A^2:")
sp.pprint(poly_A2)

# 4. Calculate the discriminant
# The discriminant of a polynomial P(A^2) = a*A^4 + b*A^2 + c is b^2 - 4ac
# If poly_A2 is of the form a*A^4 + b*A^2 + c, extract coefficients
coeffs = sp.Poly(poly_A2, A2).coeffs()

if len(coeffs) == 3:
    a, b, c = coeffs
    discriminant = b**2 - 4*a*c
    print("\nDiscriminant (b^2 - 4ac):")
    sp.pprint(sp.simplify(discriminant))
else:
    print("\nPolynomial degree is not 2 in A^2; check truncation or variable order.")