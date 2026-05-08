import sympy as sp

# symbols
l, w = sp.symbols('l w', real=True)
z, t, d = sp.symbols('z t d', real=True)
x = sp.symbols('x', real=True)

# define L'(x*)
Lp = sp.symbols('Lp', real=True)

# characteristic polynomial
char_poly = (
    l**3
    + (2*z + 1/t)*l**2
    + (1 + 2*z/t)*l
    + (1 - Lp)/t
)

# substitute lambda = i*omega
char_iw = sp.expand(char_poly.subs(l, sp.I*w))

# separate real and imaginary parts
real_part = sp.simplify(sp.re(char_iw))
imag_part = sp.simplify(sp.im(char_iw))

# solve imaginary part → omega_H
omega_sol = sp.solve(imag_part, w)
omega_H = sp.simplify(omega_sol[1])  # positive root

# plug into real part → threshold condition
real_sub = sp.simplify(real_part.subs(w, omega_H))
Lp_sol = sp.solve(real_sub, Lp)[0]

# define epsilon
eps = sp.simplify(-Lp_sol/2)

# define L'(x*) from fixed point relation
Lp_expr = -2*x*(x + d)/((x + d)**2 + 1)

# equate and solve for x*
eq = sp.simplify(Lp_expr - Lp_sol)
x_sol = sp.solve(eq, x)

# simplify solutions
x_sol = [sp.simplify(s) for s in x_sol]

alpha_expr = x*((x + d)**2 + 1)
alpha_sol = [sp.simplify(alpha_expr.subs(x, s)) for s in x_sol]

# print results
print("omega_H =", omega_H)
print("L'(x*) =", Lp_sol)
print("epsilon =", eps)
print("x* solutions =", x_sol)
print("alpha_c solutions =", alpha_sol)



alpha_sol = []
for s in x_sol:
    a = sp.simplify(alpha_expr.subs(x, s))
    a = sp.factor(a)
    a = sp.together(a)
    a = sp.simplify(a)
    alpha_sol.append(a)

alpha_sol = [sp.simplify(sp.factor(s)) for s in alpha_sol]

print("alpha_c solutions (simplified) =", alpha_sol)

sp.init_printing()

alpha_sol_pretty = [
    sp.factor(sp.cancel(sp.together(sp.simplify(alpha_expr.subs(x, s)))))
    for s in x_sol
]

for i, a in enumerate(alpha_sol_pretty):
    print(f"branch {i+1}:")
    sp.pprint(a)