import numpy as np



def rotate(v):
    # assumes (x1, y1, x2, y2, z)
    # returns (x-, y-, x+, y+, z)
    return [(v[0]-v[2])/2, (v[1]-v[3])/2, (v[0]+v[2])/2, (v[1]+v[3])/2, v[4]]


# -----------------------------
# bifurcation boundaries
# -----------------------------
def lower_boundary(d):
    s = np.sqrt(d**2 - 3)
    return -1/27 * (s - 2*d)**2 * (s + d)


def upper_boundary(d):
    s = np.sqrt(d**2 - 3)
    return  1/27 * (s + 2*d)**2 * (s - d)

# -----------------------------
# lasing threshold
# -----------------------------


def lasing_threshold(d, t, m, g1, g2, epsilon=1e-6):
    b1 = -(m + 1)**2*(g1*m + g2)/t**3
    b2 = (g1**4*m**2*t**2 + g1**3*g2*m**2*t**2 + g1**3*g2*m*t**2 + g1**3*m**2*t - g1**3*m*t + 2*g1**2*g2**2*m*t**2 + g1**2*g2*m**2*t + 2*g1**2*g2*m*t - g1**2*g2*t + g1**2*m**3*t**2 - 4*g1**2*m**2*t**2 - g1**2*m*t**2 - g1**2*m + g1*g2**3*m*t**2 + g1*g2**3*t**2 - g1*g2**2*m**2*t + 2*g1*g2**2*m*t + g1*g2**2*t - g1*g2*m**3*t**2 - 5*g1*g2*m**2*t**2 - g1*g2*m**2 - 5*g1*g2*m*t**2 - g1*g2*t**2 - g1*g2 + 2*g1*m**3*t - 2*g1*m**2*t + g2**4*t**2 - g2**3*m*t + g2**3*t - g2**2*m**2*t**2 - 4*g2**2*m*t**2 - g2**2*m + g2**2*t**2 - 2*g2*m*t + 2*g2*t - m**3*t**2 + 2*m**2*t**2 - m*t**2)/t**4
    b3 = (g1**4*g2**2*m*t**2 + g1**4*g2*m**2*t**3 + g1**4*g2*m*t - g1**4*m**2*t**2 + g1**3*g2**3*m*t**2 + g1**3*g2**3*t**2 + g1**3*g2**2*m**2*t**3 + 2*g1**3*g2**2*m*t + g1**3*g2**2*t - 3*g1**3*g2*m*t**2 + g1**3*g2*m - g1**3*m**2*t**3 - g1**3*m**2*t + g1**2*g2**4*t**2 + g1**2*g2**3*m*t + g1**2*g2**3*t**3 + 2*g1**2*g2**3*t + 2*g1**2*g2**2*m**2*t**2 - 8*g1**2*g2**2*m*t**2 + g1**2*g2**2*m + 2*g1**2*g2**2*t**2 + g1**2*g2**2 + g1**2*g2*m**3*t**3 - 5*g1**2*g2*m**2*t**3 - 3*g1**2*g2*m*t**3 - 9*g1**2*g2*m*t + 2*g1**2*g2*t - g1**2*m**3*t**2 + 2*g1**2*m**2*t**2 - g1**2*m*t**2 + g1*g2**4*t**3 + g1*g2**4*t - 3*g1*g2**3*m*t**2 + g1*g2**3 - 3*g1*g2**2*m**2*t**3 + 2*g1*g2**2*m**2*t - 5*g1*g2**2*m*t**3 - 9*g1*g2**2*m*t + g1*g2**2*t**3 + 2*g1*g2*m**3*t**2 - 2*g1*g2*m**2*t**2 + 2*g1*g2*m**2 - 2*g1*g2*m*t**2 - 4*g1*g2*m + 2*g1*g2*t**2 + 2*g1*g2 - g1*m**3*t**3 - g1*m**3*t + 2*g1*m**2*t**3 + 2*g1*m**2*t - g1*m*t**3 - g1*m*t - g2**4*t**2 - g2**3*m*t**3 - g2**3*t - g2**2*m**2*t**2 + 2*g2**2*m*t**2 - g2**2*t**2 - g2*m**3*t**3 + 2*g2*m**2*t**3 - g2*m**2*t - g2*m*t**3 + 2*g2*m*t - g2*t)/t**4
    b4 = -g1*g2*(g1*t + t**2 + 1)*(g2*t + m*t**2 + 1)*(g1**2*m + g1*g2*m + g1*g2 + g2**2 + m**2 - 2*m + 1)/t**4

    dL_sols = np.roots([b1, b2, b3, b4])
    dL_sols = np.real(dL_sols[np.isreal(dL_sols)]) # get all real solutions
    #print('dL_sols:', dL_sols)
    print('SOLUTIONS', len(dL_sols))
    alphas = []
    
    s = 2
    
    for L in dL_sols:
        E = d**2 - s*L*(s*L + 2)
        for z in [( -d*(s*L + 1) + np.sqrt(E) ) / (s*(s*L + 2)),
                  ( -d*(s*L + 1) - np.sqrt(E) ) / (s*(s*L + 2))]:
            #print('z:', z)
            a = z*( (d + s*z)**2 + 1 )
            if np.all((a > 0) & np.isreal(a)):
                alphas.append(a)
            #print('alpha:', z*( (d + s*z)**2 + 1 ))
    alphas_sorted = sorted(alphas, key=lambda a: np.min(a))
    return alphas_sorted



def limit_cycle_bifurcation_lower(g, t):
    return (-2*g**3*t**2 - 3*g**2*t - g + t - np.sqrt(g*(4*g**5*t**4 + 12*g**4*t**3 - 8*g**3*t**4 + 13*g**3*t**2 - 20*g**2*t**3 + 6*g**2*t - 4*g*t**4 - 16*g*t**2 + g - 4*t**3 - 4*t)))/(t*(2*g*t + 1))

def limit_cycle_bifurcation_upper(g, t):
    return (-2*g**3*t**2 - 3*g**2*t - g + t + np.sqrt(g*(4*g**5*t**4 + 12*g**4*t**3 - 8*g**3*t**4 + 13*g**3*t**2 - 20*g**2*t**3 + 6*g**2*t - 4*g*t**4 - 16*g*t**2 + g - 4*t**3 - 4*t)))/(t*(2*g*t + 1))



# -----------------------------
# fixed points
# -----------------------------
def z_star(a, d):
    roots = np.roots([4, 4*d, d**2 + 1, -a])
    roots = np.real(roots[np.isreal(roots)])
    return roots

def dLdz(z, d):
    return -2*z*(2*z + d) / ((2*z + d)**2 + 1)

# -----------------------------
# Jacobian
# -----------------------------
def Jacobian(z, d, t, m, g1, g2):
    dL = dLdz(z, d)
    J = np.array([
            [0,      1,       0,      0,       0     ],
            [-1,   -g1, 0,      0,       1     ],
            [0,      0,       0,      1,       0     ], 
            [0,      0,       -m, -g2, m     ],
            [dL/t, 0,       dL/t, 0,       -1/t]
        ])

    return J


def compute_eigs(a, d, t, m, g1, g2):
    roots = z_star(a, d)
    eigvals = []
    eigvecs = []

    for i, root in enumerate(roots):
        vals, vecs = np.linalg.eig(Jacobian(root, d, t, m, g1, g2))
        eigvals.append(vals)
        eigvecs.append(vecs)

    return roots, eigvals, eigvecs


# -----------------------------
# SYSTEM
# -----------------------------
def system(t, y, a, d, tau, m, g1, g2):
    x1, v1, x2, v2, z = y
    dx1dt = v1
    dv1dt = -g1 * v1 - x1 + z
    dx2dt = v2
    dv2dt = -g2 * v2 - m*(x2 - z)
    dzdt = a / ((x1+x2+d)**2+1)/tau - z/tau
    return [dx1dt, dv1dt, dx2dt, dv2dt, dzdt]



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