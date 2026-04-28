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
def lasing_threshold_older(d, t, w1, w2, g1, g2, branch='upper', epsilon=1e-10):
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


def lasing_threshold_old(delta, tau, rho, gamma1, gamma2):
    mu = gamma1/rho + gamma2*rho
    xi = gamma1 + gamma2
    chi = gamma1*gamma2 + sigma(rho)

    A = 2*(1+tau*xi)*(xi+tau*chi) - 2*tau*(chi+tau*mu)
    B = (xi+tau*chi)**2 - 4*tau*(mu+tau)
    C = (1+tau*xi)*(xi+tau*chi)**2 + (1+tau*xi)*B - 2*tau*(xi+tau*chi)*(chi+tau*mu) + 4*tau**2
    D = 4*(1+tau*xi)*tau*xi + 4*tau*(xi+tau*chi) - 4*tau**2*sigma(rho)

    b1 = 64*tau**3*xi
    b2 = 32*tau**2*A*xi + 16*tau**2*B - D**2
    b3 = 4*A**2*tau*xi + 8*tau*A*B -2*C*D
    b4 = A**2*B - C**2

    dL_sols = np.roots([b1, b2, b3, b4])
    dL_sols = np.real(dL_sols[np.isreal(dL_sols)])
    print(dL_sols)

    alphas = []
    for dL in dL_sols:
        E = 2*delta + rho*dL
        F = E**2 - 8*rho*(delta+1)*dL
        for z in [( -1*E + np.sqrt(F) ) / (4*rho),
                  ( -1*E - np.sqrt(F) ) / (4*rho)]:
            alphas.append(z*( (delta + rho*z)**2 + 1 ))
    
    return alphas

def lasing_threshold(delta, tau, rho, gamma1, gamma2):
    b1 = -4*(gamma1 + gamma2)/tau**3
    b2 = (gamma1**4*rho**2*tau**2 + 2*gamma1**3*gamma2*rho**2*tau**2 + 2*gamma1**2*gamma2**2*rho**2*tau**2 + 2*gamma1**2*gamma2*rho**2*tau - 4*gamma1**2*rho**3*tau**2 - gamma1**2*rho**2 + 2*gamma1*gamma2**3*rho**2*tau**2 + 2*gamma1*gamma2**2*rho**2*tau - 6*gamma1*gamma2*rho**3*tau**2 - 2*gamma1*gamma2*rho**2 - 6*gamma1*gamma2*rho*tau**2 - 2*gamma1*rho**3*tau + 2*gamma1*rho*tau + gamma2**4*rho**2*tau**2 - gamma2**2*rho**2 - 4*gamma2**2*rho*tau**2 + 2*gamma2*rho**3*tau - 2*gamma2*rho*tau - rho**4*tau**2 + 2*rho**2*tau**2 - tau**2)/(rho**2*tau**4)
    b3 = (gamma1**4*gamma2**2*rho**3*tau**2 + gamma1**4*gamma2*rho**3*tau + gamma1**4*gamma2*rho**2*tau**3 - gamma1**4*rho**2*tau**2 + 2*gamma1**3*gamma2**3*rho**3*tau**2 + 3*gamma1**3*gamma2**2*rho**3*tau + gamma1**3*gamma2**2*rho**2*tau**3 - gamma1**3*gamma2*rho**4*tau**2 + gamma1**3*gamma2*rho**3 - 2*gamma1**3*gamma2*rho**2*tau**2 - gamma1**3*rho**3*tau**3 - gamma1**3*rho**2*tau + gamma1**2*gamma2**4*rho**3*tau**2 + gamma1**2*gamma2**3*rho**4*tau**3 + 3*gamma1**2*gamma2**3*rho**3*tau - 2*gamma1**2*gamma2**2*rho**4*tau**2 + 2*gamma1**2*gamma2**2*rho**3 - 2*gamma1**2*gamma2**2*rho**2*tau**2 - gamma1**2*gamma2*rho**5*tau**3 - 2*gamma1**2*gamma2*rho**4*tau - 6*gamma1**2*gamma2*rho**3*tau**3 - 5*gamma1**2*gamma2*rho**2*tau - gamma1**2*rho**5*tau**2 + 2*gamma1**2*rho**3*tau**2 - gamma1**2*rho*tau**2 + gamma1*gamma2**4*rho**4*tau**3 + gamma1*gamma2**4*rho**3*tau - 2*gamma1*gamma2**3*rho**4*tau**2 + gamma1*gamma2**3*rho**3 - gamma1*gamma2**3*rho**2*tau**2 - 5*gamma1*gamma2**2*rho**4*tau - 6*gamma1*gamma2**2*rho**3*tau**3 - 2*gamma1*gamma2**2*rho**2*tau - gamma1*gamma2**2*rho*tau**3 + 2*gamma1*gamma2*rho**5*tau**2 - 4*gamma1*gamma2*rho**3*tau**2 + 2*gamma1*gamma2*rho*tau**2 - gamma1*rho**6*tau**3 - gamma1*rho**5*tau + 2*gamma1*rho**4*tau**3 + 2*gamma1*rho**3*tau - gamma1*rho**2*tau**3 - gamma1*rho*tau - gamma2**4*rho**4*tau**2 - gamma2**3*rho**4*tau - gamma2**3*rho**3*tau**3 - gamma2**2*rho**5*tau**2 + 2*gamma2**2*rho**3*tau**2 - gamma2**2*rho*tau**2 - gamma2*rho**5*tau - gamma2*rho**4*tau**3 + 2*gamma2*rho**3*tau + 2*gamma2*rho**2*tau**3 - gamma2*rho*tau - gamma2*tau**3)/(rho**3*tau**4)
    b4 = -gamma1*gamma2*(gamma1*tau + rho*tau**2 + 1)*(gamma2*rho*tau + rho + tau**2)*(gamma1**2*rho + gamma1*gamma2*rho**3 + gamma1*gamma2*rho + gamma2**2*rho**3 + rho**4 - 2*rho**2 + 1)/(rho**3*tau**4)

    dL_sols = np.roots([b1, b2, b3, b4])
    dL_sols = np.real(dL_sols[np.isreal(dL_sols)])
    print(dL_sols)

    alphas = []
    for dL in dL_sols:
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
def Jacobian(alpha, delta, tau, rho, gamma1, gamma2):
    roots = z_star(alpha, delta, rho)

    J_list = []
    for z in roots:
        dL = dLdz(z, delta, rho)
        J = np.array([
            [0,      1,       0,      0,       0     ],
            [-rho,   -gamma1, 0,      0,       1     ],
            [0,      0,       0,      1,       0     ], 
            [0,      0,       -1/rho, -gamma2, 1     ],
            [dL/tau, 0,       dL/tau, 0,       -1/tau]
        ])
        J_list.append(J)

    return J_list

def compute_eigs(alpha, delta, tau, rho, gamma1, gamma2):
    return [np.linalg.eigvals(J) for J in Jacobian(alpha, delta, tau, rho, gamma1, gamma2)]


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