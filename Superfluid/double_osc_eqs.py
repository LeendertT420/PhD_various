import numpy as np


def rho(w1, w2):
    return 1/w1**2 + 1/w2**2

# -----------------------------
# bifurcation boundaries
# -----------------------------
def lower_boundary(d, w1, w2):
    s = np.sqrt(d**2 - 3)
    return -2/27 * (s - 2*d)**2 * (s + d) / rho(w1, w2)

def upper_boundary(d, w1, w2):
    s = np.sqrt(d**2 - 3)
    return  2/27 * (s + 2*d)**2 * (s - d) / rho(w1, w2)

# -----------------------------
# lasing threshold
# -----------------------------
def lasing_threshold(d, t, w1, w2, g1, g2, branch='upper', epsilon=1e-10):
    r = rho(w1, w2)
    sw = w1**2 + w2**2
    sg = g1 + g2

    a1 = t
    a2 = 1 + a1*sg
    a3 = sg + a1*(sw + g1*g2)

    b6 = (w1*w2)**2
    b5 = g1*w2**2 + g2*w1**2 + a1*b6
    b4 = sw + g1*g2

    A = -4*a1**2*sg

    B = a1*(a2*sg - a1*sw)**2 - 2*a1*a3*(a2*sg - a1*sw)
    B+= 4*a1**2*b5 - 4*a1*sg*(a1*b4 - a2*a3)

    C = 2*a1*(a1*b6 - a2*b5)*(a2*sg - a1*sw)
    C-= a3*( 2*a1*(a1*b6 - a2*b5) + (a2*sg - a1*sw)*(a1*b4 - a2*a3) )
    C+= 4*a1*b5*(a1*b4 - a2*a3) - sg*(a1*b4 - a2*a3)**2

    D = a1*(a1*b6 - a2*b5)**2 - a3*(a1*b6 - a2*b5)*(a1*b4 - a2*a3)
    D+= b5*(a1*b4 - a2*a3)**2

    dL_sols = np.roots([A, B, C, D])
    dL_sols = np.real(dL_sols[np.isreal(dL_sols)])
    print(dL_sols)
    valid_sols = []
    for sol in dL_sols:
        P = a3**2 - 4*a1*(b5-sg*sol)
        Q = (b4+2*sol)**2 - 4*a2*(b6-sw*sol)
        print(f'P:{P}')
        print(f'Q:{Q}')
        if P >= 0 and Q >= 0:
            p1 = (a3 + np.sqrt(P)) / (2*a1)
            p2 = (a3 - np.sqrt(P)) / (2*a1)
            q1 = (b4 + 2*sol + np.sqrt(Q)) / (2*a2)
            q2 = (b4 + 2*sol - np.sqrt(Q)) / (2*a2)
            print(np.abs(a1*p1**2 - a3*p1 + b5 - sg*sol), np.abs(a1*p2**2 - a3*p2 + b5 - sg*sol))
            print(np.abs(a2*q1**2 - (b4+2*sol)*q1 + b6-sw*sol), np.abs(a2*q2**2 - (b4+2*sol)*q2 + b6-sw*sol))
            if ( (np.abs(a1*p1**2 - a3*p1 + b5 - sg*sol) < epsilon
                  or np.abs(a1*p2**2 - a3*p2 + b5 - sg*sol) < epsilon)
                and (np.abs(a2*q1**2 - (b4+2*sol)*q1 + b6-sw*sol) < epsilon
                     or np.abs(a2*q2**2 - (b4+2*sol)*q2 + b6-sw*sol) < epsilon)):

                valid_sols.append(sol)

    alphas = []
    for dL in valid_sols:
        E = 2*d + r*dL
        F = E**2 - 8*r*(d+1)*dL
        for z in [( -1*E + np.sqrt(F) ) / (4*r),
                  ( -1*E - np.sqrt(F) ) / (4*r)]:
            alphas.append(z*( (d + r*z)**2 + 1 ))
    
    return alphas


# -----------------------------
# fixed points
# -----------------------------
def z_star(alpha, delta, w1, w2):
    r = rho(w1, w2)
    roots = np.roots([r**2, 2*r*delta, delta**2 + 1, -alpha])
    roots = np.real(roots[np.isreal(roots)])
    return roots

def dLdz(z, delta, w1, w2):
    r = rho(w1, w2)
    return -2*z*(r*z + delta) / ((r*z + delta)**2 + 1)

# -----------------------------
# Jacobian
# -----------------------------
def Jacobian(alpha, delta, tau, w1, w2, g1, g2):
    roots = z_star(alpha, delta, w1, w2)

    J_list = []
    for z in roots:
        dL = dLdz(z, delta, w1, w2)
        J = np.array([
            [0,      1,   0,      0,   0     ],
            [-w1**2, -g1, 0,      0,   1     ],
            [0,      0,   0,      1,   0     ], 
            [0,      0,   -w2**2, -g2, 1     ],
            [dL/tau, 0,   dL/tau, 0,   -1/tau]
        ])
        J_list.append(J)

    return J_list

def compute_eigs(alpha, delta, tau, w1, w2, g1, g2):
    return [np.linalg.eigvals(J) for J in Jacobian(alpha, delta, tau, w1, w2, g1, g2)]


# -----------------------------
# SYSTEM
# -----------------------------
def system(t, y, alpha, delta, tau, w1, w2, g1, g2):
    x1, v1, x2, v2, z = y
    dx1dt = v1
    dv1dt = -g1 * v1 - w1**2*x1 + z
    dx2dt = v2
    dv2dt = -g2 * v2 - w2**2*x2 + z
    dzdt = alpha / ((x1+x2+delta)**2+1)/tau - z/tau
    return [dx1dt, dv1dt, dx2dt, dv2dt, dzdt]