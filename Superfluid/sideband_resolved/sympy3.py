import sympy as sp

# Define symbols
z, t, d = sp.symbols('z t d')

# Define the expressions
xi = z * (2 * z + t + 1 / t)
y = ((1 - 2 * xi) * d + sp.sqrt(d**2 - 4 * xi * (xi - 1))) / (2 * xi - 2)
alpha = y * ((y + d)**2 + 1)

# Simplify alpha
# Note: simplify() can be computationally intensive; 
# simplify() or powsimp() are generally effective here.
simplified_alpha = sp.simplify(alpha)

print("Simplified expression for alpha:")
print(simplified_alpha)


E = -(d*(t - 2*z*(t*(t + 2*z) + 1)) + t*sp.sqrt((d**2*t**2 + 4*z*(t - z*(t*(t + 2*z) + 1))*(t*(t + 2*z) + 1))/t**2))*(4*(t - z*(t*(t + 2*z) + 1))**2 + (d*(t - 2*z*(t*(t + 2*z) + 1)) - 2*d*(t - z*(t*(t + 2*z) + 1)) + t*sp.sqrt((d**2*t**2 + 4*z*(t - z*(t*(t + 2*z) + 1))*(t*(t + 2*z) + 1))/t**2))**2)/(8*(t - z*(t*(t + 2*z) + 1))**3)

# 1. Try cancel() first: This handles rational functions efficiently
expr_canceled = sp.cancel(E)

# 2. Try simplify(): The general-purpose engine
expr_simplified = sp.simplify(expr_canceled)

# 3. If it's still messy, try collecting terms to see structure
# e.g., collect with respect to d
expr_collected = sp.collect(expr_simplified, d)

print(expr_collected)