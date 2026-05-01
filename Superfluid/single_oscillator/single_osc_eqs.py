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

def L_prime(x_star, alpha, delta):
    return -2*alpha*(x_star + delta) / ((x_star + delta)**2 + 1)**2


# -----------------------------
# Jacobian
# -----------------------------
def Jacobian(x, alpha, delta, zeta, tau):
    dLdt = L_prime(x, alpha, delta)
    J = np.array([
        [0,    1,        0],
        [-1,  -2*zeta,   1],
        [dLdt/tau, 0,   -1/tau]
        ])
    return J

def compute_eigs(alpha, delta, zeta, tau):
    roots = x_star(alpha, delta)
    eigvals = []
    eigvecs = [[],[],[]]

    for i, root in enumerate(roots):
        l = np.linalg.eigvals(Jacobian(root, alpha, delta, zeta, tau))
        eigvals.append(l)
        for val in l:
            v = [1, val, L_prime(root, alpha, delta)/(val*tau+1)]
            eigvecs[i].append(v)

    return roots, eigvals, eigvecs


# -----------------------------
# SYSTEM
# -----------------------------
def system(t, y, alpha, delta, zeta, tau):
    x, v, z = y
    dxdt = v
    dvdt = -2 * zeta * v - x + z
    dzdt = alpha / ((x+delta)**2+1)/tau - z/tau
    return [dxdt, dvdt, dzdt]





def project_onto_plane(x, v1, v2):
    """
    Project vector x onto the plane spanned by v1 and v2.

    Parameters:
        x, v1, v2 : array-like (shape: (n,))
    
    Returns:
        projection of x onto span{v1, v2}
    """
    # Stack vectors as columns of A (n x 2 matrix)
    A = np.column_stack((v1, v2))
    
    # Compute projection: A (A^T A)^{-1} A^T x
    ATA_inv = np.linalg.inv(A.T @ A)
    projection = A @ ATA_inv @ A.T @ x
    
    return projection