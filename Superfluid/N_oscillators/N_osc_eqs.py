import numpy as np
from scipy.special import jn_zeros

# -----------------------------
# bifurcation boundaries
# -----------------------------
def lower_boundary(N, d):
    s = np.sqrt(d**2 - 3)
    return -2/27 * (s - 2*d)**2 * (s + d) / N

def upper_boundary(N, d):
    s = np.sqrt(d**2 - 3)
    return  2/27 * (s + 2*d)**2 * (s - d) / N


def zeta(i):
    return jn_zeros(0, i)[0]

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
def z_star(N, alpha, d):
    roots = np.roots([N**2, 2*N*d, d**2 + 1, -alpha])
    roots = np.real(roots[np.isreal(roots)])
    return roots


def L_prime_star(N, z_star, d):
    return -2*z_star*(N*z_star + d) / ((N*z_star + d)**2 + 1)


# -----------------------------
# Jacobian
# -----------------------------
def Jacobian(N, z_star, d, t, gs):
    L = L_prime_star(N, z_star, d)
    J = np.zeros( (2*N+1, 2*N+1) )
    for i in range(N):
        block = np.array([[0          , 1],
                          [-zeta(i+1)**2, -gs[i]]])
        J[2*i:2*i+2, 2*i:2*i+2] = block
        J[2*i+1,-1] = zeta(i+1)**2
    J[-1,:-1] = [L/t, 0]*N
    J[-1,-1] = -1/t

    return J

def compute_eigs(N, a, d, t, gs):
    roots = z_star(N, a, d)
    eigvals = []
    eigvecs = [[],[],[]]

    for i, root in enumerate(roots):
        vals, vecs = np.linalg.eig(Jacobian(N, root, d, t, gs))
        eigvals.append(vals)
        print(vals)
        eigvecs.append(vecs)
        print(vecs)
    for i in range(len(vals)):
        v = vecs[:, i]
        lam = vals[i]
        print(i, np.allclose(Jacobian(N, root, d, t, gs) @ v, lam * v))

    return roots, eigvals, eigvecs


# -----------------------------
# SYSTEM
# -----------------------------
def system(time, X, a, d, gs, t):
    N = (len(X)-1)//2
    x = X[:-1:2]
    v = X[1::2]
    z = X[-1]
    dX = np.zeros(2*N+1)
    for i, (x_i, v_i) in enumerate(zip(x, v)):
        dX[2*i] = v_i
        dX[2*i+1] = -1*gs[i]*v_i - zeta(i+1)**2*x_i + zeta(i+1)**2*z
    dX[-1] = a / ((np.sum(x)+d)**2+1)/t - z/t
    return dX





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


if __name__ == '__main__':
    N = 2
    gs = [.25, 1]
    a = 1
    t = 2
    d = 1
    z = z_star(N, a, d)
    print(z)
    print(Jacobian(N, z[0], d, t, gs))
    print(compute_eigs(N, a, d, t, gs))
    roots, vals, vecs = compute_eigs(N, a, d, t, gs)
    for i in range(len(vals)):
        v = vecs[0][:, i]
        lam = vals[i]
        print(i, np.allclose(A @ v, lam * v))