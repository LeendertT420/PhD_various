import numpy as np

# -----------------------------
# bifurcation boundaries
# -----------------------------
def lower_boundary(d):
    s = np.sqrt(d**2 - 3)
    return -2/27 * (s - 2*d)**2 * (s + d)

def upper_boundary(d):
    s = np.sqrt(d**2 - 3)
    return  2/27 * (s + 2*d)**2 * (s - d)

# -----------------------------
# lasing threshold
# -----------------------------
def lasing_threshold(d, zeta, tau):
    B = zeta*(2*zeta*tau + tau + 1/tau)
    x = ((1 - 2*B)*d - np.sqrt(d**2 - 4*B*(B - 1))) / (2*B - 2)
    return x * ((x + d)**2 + 1)

def lasing_threshold2(d, zeta, tau):
    B = zeta*(2*zeta*tau + tau + 1/tau)
    x = ((1 - 2*B)*d + np.sqrt(d**2 - 4*B*(B - 1))) / (2*B - 2)
    return x * ((x + d)**2 + 1)

# -----------------------------
# fixed points
# -----------------------------
def x_star(alpha, delta):
    roots = np.roots([1, 2*delta, delta**2 + 1, -alpha])
    roots = np.real(roots[np.isreal(roots)])
    return roots

def L_prime(x, alpha, delta):
    return -2*alpha*(x + delta) / ((x + delta)**2 + 1)**2

# -----------------------------
# Jacobian
# -----------------------------
def Jacobian(alpha, delta, zeta, tau):
    roots = x_star(alpha, delta)

    J_list = []
    for x in roots:
        dLdt = L_prime(x, alpha, delta)
        J = np.array([
            [0,    1,        0],
            [-1,  -2*zeta,   1],
            [dLdt/tau, 0,   -1/tau]
        ])
        J_list.append(J)

    return J_list

def compute_eigs(alpha, delta, zeta, tau):
    return [np.linalg.eigvals(J) for J in Jacobian(alpha, delta, zeta, tau)]


# -----------------------------
# SYSTEM
# -----------------------------
def system(t, y, alpha, delta, zeta, tau):
    x, v, z = y
    dxdt = v
    dvdt = -2 * zeta * v - x + z
    dzdt = alpha / ((x+delta)**2+1)/tau - z/tau
    return [dxdt, dvdt, dzdt]