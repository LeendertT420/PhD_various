import numpy as np



def rotate(v):
    # assumes (x1, y1, x2, y2, z)
    # returns (x-, y-, x+, y+, z)
    return [(v[0]-v[2])/2, (v[1]-v[3])/2, (v[0]+v[2])/2, (v[1]+v[3])/2, v[4]]


def sigma(rho):
    return rho + 1/rho

# -----------------------------
# bifurcation boundaries
# -----------------------------
def lower_boundary(d, rho):
    s = np.sqrt(d**2 - 3)
    return -2/27 * (s - 2*d)**2 * (s + d) / sigma(rho)


def upper_boundary(d, rho):
    s = np.sqrt(d**2 - 3)
    return  2/27 * (s + 2*d)**2 * (s - d) / sigma(rho)

# -----------------------------
# lasing threshold
# -----------------------------

def lasing_threshold_old(d, t, r, g1, g2, epsilon=1e-6):
    b1 = -4*(g1 + g2)/t**3
    b2 = (g1**4*r**2*t**2 + 2*g1**3*g2*r**2*t**2 + 2*g1**2*g2**2*r**2*t**2 + 2*g1**2*g2*r**2*t - 4*g1**2*r**3*t**2 - g1**2*r**2 + 2*g1*g2**3*r**2*t**2 + 2*g1*g2**2*r**2*t - 6*g1*g2*r**3*t**2 - 2*g1*g2*r**2 - 6*g1*g2*r*t**2 - 2*g1*r**3*t + 2*g1*r*t + g2**4*r**2*t**2 - g2**2*r**2 - 4*g2**2*r*t**2 + 2*g2*r**3*t - 2*g2*r*t - r**4*t**2 + 2*r**2*t**2 - t**2)/(r**2*t**4)
    b3 = (g1**4*g2**2*r**3*t**2 + g1**4*g2*r**3*t + g1**4*g2*r**2*t**3 - g1**4*r**2*t**2 + 2*g1**3*g2**3*r**3*t**2 + 3*g1**3*g2**2*r**3*t + g1**3*g2**2*r**2*t**3 - g1**3*g2*r**4*t**2 + g1**3*g2*r**3 - 2*g1**3*g2*r**2*t**2 - g1**3*r**3*t**3 - g1**3*r**2*t + g1**2*g2**4*r**3*t**2 + g1**2*g2**3*r**4*t**3 + 3*g1**2*g2**3*r**3*t - 2*g1**2*g2**2*r**4*t**2 + 2*g1**2*g2**2*r**3 - 2*g1**2*g2**2*r**2*t**2 - g1**2*g2*r**5*t**3 - 2*g1**2*g2*r**4*t - 6*g1**2*g2*r**3*t**3 - 5*g1**2*g2*r**2*t - g1**2*r**5*t**2 + 2*g1**2*r**3*t**2 - g1**2*r*t**2 + g1*g2**4*r**4*t**3 + g1*g2**4*r**3*t - 2*g1*g2**3*r**4*t**2 + g1*g2**3*r**3 - g1*g2**3*r**2*t**2 - 5*g1*g2**2*r**4*t - 6*g1*g2**2*r**3*t**3 - 2*g1*g2**2*r**2*t - g1*g2**2*r*t**3 + 2*g1*g2*r**5*t**2 - 4*g1*g2*r**3*t**2 + 2*g1*g2*r*t**2 - g1*r**6*t**3 - g1*r**5*t + 2*g1*r**4*t**3 + 2*g1*r**3*t - g1*r**2*t**3 - g1*r*t - g2**4*r**4*t**2 - g2**3*r**4*t - g2**3*r**3*t**3 - g2**2*r**5*t**2 + 2*g2**2*r**3*t**2 - g2**2*r*t**2 - g2*r**5*t - g2*r**4*t**3 + 2*g2*r**3*t + 2*g2*r**2*t**3 - g2*r*t - g2*t**3)/(r**3*t**4)
    b4 = -g1*g2*(g1*t + r*t**2 + 1)*(g2*r*t + r + t**2)*(g1**2*r + g1*g2*r**3 + g1*g2*r + g2**2*r**3 + r**4 - 2*r**2 + 1)/(r**3*t**4)

    dL_sols = np.roots([b1, b2, b3, b4])
    #dL_sols = np.real(dL_sols[np.isreal(dL_sols)]) # get all real solutions

    a1 = 1
    a2 = (g1*t + g2*t + 1)/t
    a3 = (g1*g2*r*t + g1*r + g2*r + r**2*t + t)/(r*t)

    alphas = []
    
    s = sigma(r)
    
    for L in dL_sols:
        
        a4 = (-2*L*r + g1*g2*r + g1*t + g2*r**2*t + r**2 + 1)/(r*t)
        a5 = (-L*g1*r - L*g2*r + g1 + g2*r**2 + r*t)/(r*t)
        a6 = (-L*r**2 - L + r)/(r*t)

        
        #test whether solution correspond to real solutions of omega 
        if (a3**2 >= 4*a1*a5 
            and a4**2 >= 4*a2*a6):

            omegas = [np.sqrt((a3 + np.sqrt(a3**2 - 4*a1*a5)) / (2*a1)),
                      np.sqrt((a3 - np.sqrt(a3**2 - 4*a1*a5)) / (2*a1)),
                      np.sqrt((a4 + np.sqrt(a4**2 - 4*a2*a6)) / (2*a2)),
                      np.sqrt((a4 - np.sqrt(a4**2 - 4*a2*a6)) / (2*a2))]
            print(f'omegas{omegas}')
            for w in omegas:
                if True:#(np.abs(a1*w**4 - a3*w**2 + a5) < epsilon
                    #and np.abs(a2*w**4 - a4*w**2 + a6) < epsilon):
                    print(w)
        E = d**2 - s*L*(s*L + 2)
        for z in [( -d*(s*L + 1) + np.sqrt(E) ) / (s*(s*L + 2)),
                  ( -d*(s*L + 1) - np.sqrt(E) ) / (s*(s*L + 2))]:
            alphas.append(z*( (d + s*z)**2 + 1 ))
    print(np.shape(alphas))        
    return alphas


def lasing_threshold(d, t, r, g1, g2, epsilon=1e-6):
    b1 = -4*(g1 + g2)/t**3
    b2 = (g1**4*r**2*t**2 + 2*g1**3*g2*r**2*t**2 + 2*g1**2*g2**2*r**2*t**2 + 2*g1**2*g2*r**2*t - 4*g1**2*r**3*t**2 - g1**2*r**2 + 2*g1*g2**3*r**2*t**2 + 2*g1*g2**2*r**2*t - 6*g1*g2*r**3*t**2 - 2*g1*g2*r**2 - 6*g1*g2*r*t**2 - 2*g1*r**3*t + 2*g1*r*t + g2**4*r**2*t**2 - g2**2*r**2 - 4*g2**2*r*t**2 + 2*g2*r**3*t - 2*g2*r*t - r**4*t**2 + 2*r**2*t**2 - t**2)/(r**2*t**4)
    b3 = (g1**4*g2**2*r**3*t**2 + g1**4*g2*r**3*t + g1**4*g2*r**2*t**3 - g1**4*r**2*t**2 + 2*g1**3*g2**3*r**3*t**2 + 3*g1**3*g2**2*r**3*t + g1**3*g2**2*r**2*t**3 - g1**3*g2*r**4*t**2 + g1**3*g2*r**3 - 2*g1**3*g2*r**2*t**2 - g1**3*r**3*t**3 - g1**3*r**2*t + g1**2*g2**4*r**3*t**2 + g1**2*g2**3*r**4*t**3 + 3*g1**2*g2**3*r**3*t - 2*g1**2*g2**2*r**4*t**2 + 2*g1**2*g2**2*r**3 - 2*g1**2*g2**2*r**2*t**2 - g1**2*g2*r**5*t**3 - 2*g1**2*g2*r**4*t - 6*g1**2*g2*r**3*t**3 - 5*g1**2*g2*r**2*t - g1**2*r**5*t**2 + 2*g1**2*r**3*t**2 - g1**2*r*t**2 + g1*g2**4*r**4*t**3 + g1*g2**4*r**3*t - 2*g1*g2**3*r**4*t**2 + g1*g2**3*r**3 - g1*g2**3*r**2*t**2 - 5*g1*g2**2*r**4*t - 6*g1*g2**2*r**3*t**3 - 2*g1*g2**2*r**2*t - g1*g2**2*r*t**3 + 2*g1*g2*r**5*t**2 - 4*g1*g2*r**3*t**2 + 2*g1*g2*r*t**2 - g1*r**6*t**3 - g1*r**5*t + 2*g1*r**4*t**3 + 2*g1*r**3*t - g1*r**2*t**3 - g1*r*t - g2**4*r**4*t**2 - g2**3*r**4*t - g2**3*r**3*t**3 - g2**2*r**5*t**2 + 2*g2**2*r**3*t**2 - g2**2*r*t**2 - g2*r**5*t - g2*r**4*t**3 + 2*g2*r**3*t + 2*g2*r**2*t**3 - g2*r*t - g2*t**3)/(r**3*t**4)
    b4 = -g1*g2*(g1*t + r*t**2 + 1)*(g2*r*t + r + t**2)*(g1**2*r + g1*g2*r**3 + g1*g2*r + g2**2*r**3 + r**4 - 2*r**2 + 1)/(r**3*t**4)

    dL_sols = np.roots([b1, b2, b3, b4])

    alphas = []
    
    s = sigma(r)
    
    for L in dL_sols:
        E = d**2 - s*L*(s*L + 2)
        for z in [( -d*(s*L + 1) + np.sqrt(E) ) / (s*(s*L + 2)),
                  ( -d*(s*L + 1) - np.sqrt(E) ) / (s*(s*L + 2))]:
            alphas.append(z*( (d + s*z)**2 + 1 ))
    print(np.shape(alphas))        
    return alphas






# -----------------------------
# fixed points
# -----------------------------
def z_star(alpha, delta, rho):
    roots = np.roots([sigma(rho)**2, 2*sigma(rho)*delta, delta**2 + 1, -alpha])
    roots = np.real(roots[np.isreal(roots)])
    return roots

def dLdz(z, delta, rho):
    return -2*z*(sigma(rho)*z + delta) / ((sigma(rho)*z + delta)**2 + 1)

# -----------------------------
# Jacobian
# -----------------------------
def Jacobian(z, delta, tau, rho, gamma1, gamma2):
    dL = dLdz(z, delta, rho)
    J = np.array([
            [0,      1,       0,      0,       0     ],
            [-rho,   -gamma1, 0,      0,       1     ],
            [0,      0,       0,      1,       0     ], 
            [0,      0,       -1/rho, -gamma2, 1     ],
            [dL/tau, 0,       dL/tau, 0,       -1/tau]
        ])

    return J


def compute_eigs(alpha, delta, tau, rho, gamma1, gamma2):
    roots = z_star(alpha, delta, rho)
    eigvals = []
    eigvecs = []

    for i, root in enumerate(roots):
        vals, vecs = np.linalg.eig(Jacobian(root, delta, tau, rho, gamma1, gamma2))
        eigvals.append(vals)
        eigvecs.append(vecs)

    return roots, eigvals, eigvecs


# -----------------------------
# SYSTEM
# -----------------------------
def system(t, y, alpha, delta, tau, rho, gamma1, gamma2):
    x1, v1, x2, v2, z = y
    dx1dt = v1
    dv1dt = -gamma1 * v1 - rho*x1 + z
    dx2dt = v2
    dv2dt = -gamma2 * v2 - 1/rho*x2 + z
    dzdt = alpha / ((x1+x2+delta)**2+1)/tau - z/tau
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