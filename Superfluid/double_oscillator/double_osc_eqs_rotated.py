import numpy as np


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

def lasing_threshold(delta, tau, rho, gamma1, gamma2, epsilon=1e-9):
    b1 = -4*(gamma1 + gamma2)/tau**3
    b2 = (gamma1**4*rho**2*tau**2 + 2*gamma1**3*gamma2*rho**2*tau**2 + 2*gamma1**2*gamma2**2*rho**2*tau**2 + 2*gamma1**2*gamma2*rho**2*tau - 4*gamma1**2*rho**3*tau**2 - gamma1**2*rho**2 + 2*gamma1*gamma2**3*rho**2*tau**2 + 2*gamma1*gamma2**2*rho**2*tau - 6*gamma1*gamma2*rho**3*tau**2 - 2*gamma1*gamma2*rho**2 - 6*gamma1*gamma2*rho*tau**2 - 2*gamma1*rho**3*tau + 2*gamma1*rho*tau + gamma2**4*rho**2*tau**2 - gamma2**2*rho**2 - 4*gamma2**2*rho*tau**2 + 2*gamma2*rho**3*tau - 2*gamma2*rho*tau - rho**4*tau**2 + 2*rho**2*tau**2 - tau**2)/(rho**2*tau**4)
    b3 = (gamma1**4*gamma2**2*rho**3*tau**2 + gamma1**4*gamma2*rho**3*tau + gamma1**4*gamma2*rho**2*tau**3 - gamma1**4*rho**2*tau**2 + 2*gamma1**3*gamma2**3*rho**3*tau**2 + 3*gamma1**3*gamma2**2*rho**3*tau + gamma1**3*gamma2**2*rho**2*tau**3 - gamma1**3*gamma2*rho**4*tau**2 + gamma1**3*gamma2*rho**3 - 2*gamma1**3*gamma2*rho**2*tau**2 - gamma1**3*rho**3*tau**3 - gamma1**3*rho**2*tau + gamma1**2*gamma2**4*rho**3*tau**2 + gamma1**2*gamma2**3*rho**4*tau**3 + 3*gamma1**2*gamma2**3*rho**3*tau - 2*gamma1**2*gamma2**2*rho**4*tau**2 + 2*gamma1**2*gamma2**2*rho**3 - 2*gamma1**2*gamma2**2*rho**2*tau**2 - gamma1**2*gamma2*rho**5*tau**3 - 2*gamma1**2*gamma2*rho**4*tau - 6*gamma1**2*gamma2*rho**3*tau**3 - 5*gamma1**2*gamma2*rho**2*tau - gamma1**2*rho**5*tau**2 + 2*gamma1**2*rho**3*tau**2 - gamma1**2*rho*tau**2 + gamma1*gamma2**4*rho**4*tau**3 + gamma1*gamma2**4*rho**3*tau - 2*gamma1*gamma2**3*rho**4*tau**2 + gamma1*gamma2**3*rho**3 - gamma1*gamma2**3*rho**2*tau**2 - 5*gamma1*gamma2**2*rho**4*tau - 6*gamma1*gamma2**2*rho**3*tau**3 - 2*gamma1*gamma2**2*rho**2*tau - gamma1*gamma2**2*rho*tau**3 + 2*gamma1*gamma2*rho**5*tau**2 - 4*gamma1*gamma2*rho**3*tau**2 + 2*gamma1*gamma2*rho*tau**2 - gamma1*rho**6*tau**3 - gamma1*rho**5*tau + 2*gamma1*rho**4*tau**3 + 2*gamma1*rho**3*tau - gamma1*rho**2*tau**3 - gamma1*rho*tau - gamma2**4*rho**4*tau**2 - gamma2**3*rho**4*tau - gamma2**3*rho**3*tau**3 - gamma2**2*rho**5*tau**2 + 2*gamma2**2*rho**3*tau**2 - gamma2**2*rho*tau**2 - gamma2*rho**5*tau - gamma2*rho**4*tau**3 + 2*gamma2*rho**3*tau + 2*gamma2*rho**2*tau**3 - gamma2*rho*tau - gamma2*tau**3)/(rho**3*tau**4)
    b4 = -gamma1*gamma2*(gamma1*tau + rho*tau**2 + 1)*(gamma2*rho*tau + rho + tau**2)*(gamma1**2*rho + gamma1*gamma2*rho**3 + gamma1*gamma2*rho + gamma2**2*rho**3 + rho**4 - 2*rho**2 + 1)/(rho**3*tau**4)

    dL_sols = np.roots([b1, b2, b3, b4])
    dL_sols = np.real(dL_sols[np.isreal(dL_sols)]) # get all real solutions

    a1 = 1
    a2 = 1/tau + gamma1 + gamma2
    a3 = (gamma1 + gamma2)/tau + sigma(rho) + gamma1*gamma2

    alphas = []
    print(f'dL:{dL_sols}')
    for dL in dL_sols:
        
        a4 = (sigma(rho)+gamma1*gamma2)/tau + gamma1/rho + gamma2*rho - 2*dL/tau
        a5 = (gamma1/rho + gamma2*rho)/tau + 1 - (gamma1+gamma2)*dL/tau
        a6 = 1-sigma(rho)*dL/tau

        
        #test whether solution correspond to real solutions of omega 
        if (a3**2 >= 4*a1*a5 
            and a4**2 >= 4*a2*a6):

            omegas = [np.sqrt((a3 + np.sqrt(a3**2 - 4*a1*a5)) / (2*a1)),
                      np.sqrt((a3 - np.sqrt(a3**2 - 4*a1*a5)) / (2*a1)),
                      np.sqrt((a4 + np.sqrt(a4**2 - 4*a2*a6)) / (2*a2)),
                      np.sqrt((a4 - np.sqrt(a4**2 - 4*a2*a6)) / (2*a2))]
            
            print(f'omegas:{omegas}')
            for w in omegas:
                print(np.abs(a1*w**4 - a3*w**2 + a5))
                print(np.abs(a2*w**4 - a4*w**2 + a6))
                if (np.abs(a1*w**4 - a3*w**2 + a5) < epsilon
                    and np.abs(a2*w**4 - a4*w**2 + a6) < epsilon):

                    E = 2*delta + sigma(rho)*dL
                    F = E**2 - 8*sigma(rho)*(delta+1)*dL
                    for z in [( -1*E + np.sqrt(F) ) / (4*sigma(rho)),
                            ( -1*E - np.sqrt(F) ) / (4*sigma(rho))]:
                        alphas.append(z*( (delta + sigma(rho)*z)**2 + 1 ))
            
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
    eigvecs = [[],[],[]]

    for i, root in enumerate(roots):
        l = np.linalg.eigvals(Jacobian(root, delta, tau, rho, gamma1, gamma2))
        eigvals.append(l)
        for val in l:
            v = [1, val, dLdz(root, delta, rho)/(val*tau+1)]
            eigvecs[i].append(v)

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