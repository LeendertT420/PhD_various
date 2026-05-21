import numpy as np
import sympy as sp
from scipy.special import jn_zeros
from scipy.optimize import root_scalar
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

verbose = True
np.set_printoptions(precision=5)

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
    return jn_zeros(1, i)[-1]

def mu_spectrum(i):
    return ( jn_zeros(1, i) / jn_zeros(1, 1) )**2


# -----------------------------
# lasing threshold
# -----------------------------
def lasing_threshold(N, threshold_polys, t, d):
    if verbose: print('CALCULATING LASING THRESHOLD')
    print(f'KANKER: {threshold_polys['omega2_solutions']}')
    w2_sols = extract_real_entries(threshold_polys['omega2_solutions'])
    #print(threshold_polys['threshold_polynomial'])
    if verbose: print(f'\tw^2: {w2_sols} {np.shape(w2_sols)}')
    
    w2_sols = np.real(w2_sols[np.isreal(w2_sols)])
    w2_sols = w2_sols[w2_sols>=0]
    if verbose: print(f'\tw^2 (after filtering): {w2_sols} {np.shape(w2_sols)}')

    dL_sols = []

    for sol in w2_sols:
        dL_sols.append(-t * np.polyval(threshold_polys['g_Re(x)'], sol) / np.polyval(threshold_polys['h_Re(x)'], sol))
    
    if verbose: print(f'\tdL solutions: {dL_sols} {np.shape(dL_sols)}')

    z_sols = []

    for sol in dL_sols:
        D = d**2 - sol*(sol+2)
        z_sols.append( (-d*(sol+1) + np.sqrt(D)) / (N*(sol+2)) )
        z_sols.append( (-d*(sol+1) - np.sqrt(D)) / (N*(sol+2)) )

    z_sols = np.array(z_sols)
    if verbose: print(f'\tz solutions shape:{np.shape(z_sols)}')

    thresholds = z_sols * ((N*z_sols + d)**2 + 1)
    if verbose: print(f'\tthresholds shape:{np.shape(thresholds)}')
    
    filtered_thresholds = thresholds#filter_arrays(thresholds)
    #print(thresholds, filtered_thresholds)
    if verbose: print(f'\tthresholds shape (after filtering):{np.shape(filtered_thresholds)}')

    alphas_sorted = sorted(filtered_thresholds, key=lambda a: np.min(a))
    print(alphas_sorted)
    return alphas_sorted

def find_pure_imag_crossings(mus, gs, t,
                              dL_min, dL_max,
                              num_scan_points=500):

    J0 = construct_J0(mus, gs, t, symbolic=False)

    def eigen_decomposition(dL):
        J = J0.copy()
        J[-1, 0] = dL / t
        eigvals = np.linalg.eigvals(J)

        # IMPORTANT: ordering is by imaginary part (physical spectrum)
        idx = np.argsort(np.imag(eigvals))
        return eigvals[idx]

    def real_parts(dL):
        return np.real(eigen_decomposition(dL))

    dLs = np.linspace(dL_min, dL_max, num_scan_points)

    roots = []

    prev = real_parts(dLs[0])

    for i in range(1, len(dLs)):
        curr = real_parts(dLs[i])

        for k in range(len(prev)):

            f1 = prev[k]
            f2 = curr[k]

            # detect crossing of imaginary axis
            if f1 * f2 < 0 or f1 == 0 or f2 == 0:

                def f(dL, k=k):
                    vals = real_parts(dL)[k]
                    return vals

                sol = root_scalar(
                    f,
                    bracket=[dLs[i - 1], dLs[i]],
                    method='brentq'
                )

                if sol.converged:
                    roots.append(sol.root)

        prev = curr

    return np.unique(np.round(roots, 10))



def lasing_threshold2(N, d, t, mus, gs, num_scan_points=500):
    if isinstance(d, (np.ndarray, list)):
        d_max = np.max(np.abs([d[0], d[-1]]))
    else:
        d_max = d

    dL_min = (1+np.sqrt(1+d_max**2))/2
    dL_max = (1-np.sqrt(1+d_max**2))/2

    dL_sols = find_pure_imag_crossings(mus, gs, t, dL_min, dL_max, num_scan_points=num_scan_points)

    
    if verbose: print(f'\tdL solutions: {dL_sols} {np.shape(dL_sols)}')

    z_sols = []

    for sol in dL_sols:
        D = d**2 - sol*(sol+2)
        z_sols.append( (-d*(sol+1) + np.sqrt(D)) / (N*(sol+2)) )
        z_sols.append( (-d*(sol+1) - np.sqrt(D)) / (N*(sol+2)) )

    z_sols = np.array(z_sols)
    if verbose: print(f'\tz solutions shape:{np.shape(z_sols)}')

    thresholds = z_sols * ((N*z_sols + d)**2 + 1)
    if verbose: print(f'\tthresholds shape:{np.shape(thresholds)}')
    
    filtered_thresholds = filter_arrays(thresholds)
    #print(thresholds, filtered_thresholds)
    if verbose: print(f'\tthresholds shape (after filtering):{np.shape(filtered_thresholds)}')

    alphas_sorted = sorted(filtered_thresholds, key=lambda a: np.min(a))
    print(alphas_sorted)
    return alphas_sorted


def filter_arrays(arr_list):
    """
    Remove arrays that:
    - are entirely negative
    - consist only of NaN values
    """
    filtered = []

    for arr in arr_list:
        arr = np.asarray(arr)

        # skip arrays with only NaNs
        if np.all(np.isnan(arr)):
            continue

        # skip arrays that are entirely negative
        if np.all(arr < 0):
            continue

        filtered.append(arr)

    return filtered


# -----------------------------
# fixed points
# -----------------------------
def z_star(N, a, d):
    roots = np.roots([N**2, 2*N*d, d**2 + 1, -a])
    roots = np.real(roots[np.isreal(roots)])
    print(f'roots: {roots}')
    return roots


def dL_star(N, z, d):
    return -2*N*z*(N*z + d) / ((N*z + d)**2 + 1)


# -----------------------------
# Jacobian
# -----------------------------
def construct_J0(mu_vals, gamma_vals, tau, symbolic=True):
    """
    Construct J0:
    
        J = J0 + p q^T
    
    where the (last row, first column) element dL/tau
    has been omitted from J0.

    Variable ordering:

        [X, Y,
         u_2, v_2,
         u_3, v_3,
         ...
         u_N, v_N,
         z]
    """

    N = len(mu_vals)

    if symbolic:
        mu_vals = list(map(sp.sympify, mu_vals))
        gamma_vals = list(map(sp.sympify, gamma_vals))
        tau = sp.sympify(tau)
    else:
        mu_vals = np.asarray(mu_vals, dtype=float)
        gamma_vals = np.asarray(gamma_vals, dtype=float)

    # ------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------

    mu_bar = sum(mu_vals)/N
    gamma_bar = sum(gamma_vals)/N

    mu_tilde = [mu_vals[i] - mu_bar for i in range(N)]
    gamma_tilde = [gamma_vals[i] - gamma_bar for i in range(N)]

    mu_hat = [(mu_vals[i] - mu_vals[0])/N for i in range(N)]
    gamma_hat = [(gamma_vals[i] - gamma_vals[0])/N for i in range(N)]

    # ------------------------------------------------------------
    # Matrix size
    # ------------------------------------------------------------

    dim = 2*N + 1

    if symbolic:
        J0 = sp.zeros(dim, dim)
    else:
        J0 = np.zeros((dim, dim))

    z_idx = dim - 1

    # ============================================================
    # Row 1: Xdot = Y
    # ============================================================

    J0[0, 1] = 1

    # ============================================================
    # Row 2: Ydot
    # ============================================================

    J0[1, 0] = -mu_bar
    J0[1, 1] = -gamma_bar

    # Coupling to all (u_j, v_j)
    for j in range(1, N):

        uj = 2*j
        vj = uj + 1

        J0[1, uj] = -mu_hat[j]
        J0[1, vj] = -gamma_hat[j]

    # coupling to z
    J0[1, z_idx] = mu_bar

    # ============================================================
    # Rows for each (u_i, v_i)
    # ============================================================

    for i in range(1, N):

        ui = 2*i
        vi = ui + 1

        # --------------------------------------------------------
        # u_i dot = v_i
        # --------------------------------------------------------

        J0[ui, vi] = 1

        # --------------------------------------------------------
        # v_i equation
        # --------------------------------------------------------

        J0[vi, 0] = -mu_tilde[i]
        J0[vi, 1] = -gamma_tilde[i]

        # Dense coupling to ALL u_j,v_j
        for j in range(1, N):

            uj = 2*j
            vj = uj + 1

            if i == j:

                J0[vi, uj] = mu_hat[j] - mu_vals[j]
                J0[vi, vj] = gamma_hat[j] - gamma_vals[j]

            else:

                J0[vi, uj] = mu_hat[j]
                J0[vi, vj] = gamma_hat[j]

        # coupling to z
        J0[vi, z_idx] = mu_tilde[i]

    J0[z_idx, z_idx] = -1/tau

    return J0


def clear_denominators(expr, x):
    # combine all rational terms into single fraction
    expr = sp.together(expr)

    # optionally simplify cancellations
    expr = sp.cancel(expr)

    # split numerator / denominator
    num, den = sp.fraction(expr)

    # return polynomial numerator (if valid)
    num = sp.expand(num)
    den = sp.expand(den)

    # enforce polynomial in x
    num = sp.Poly(num, x).as_expr()

    return num, den


def derive_threshold_polynomials(mu_vals, gamma_vals, tau):
    """
    Derive the omega^2 polynomial equation:
    
        g_Re*h_Im + g_Im*h_Re = 0
    
    and solve for omega^2.
    """

    lam = sp.symbols('lam')
    omega = sp.symbols('omega', real=True)
    x = sp.symbols('x', real=True)

    J0 = construct_J0(mu_vals, gamma_vals, tau, symbolic=True)
    J0 = np.array(J0, dtype=float)
    J0 = sp.Matrix(J0)
    dim = J0.shape[0]

    M = J0 - lam * sp.eye(dim)
    print(M)
    B = M.copy()
    B.row_del(dim-1)
    B.col_del(0)
    print(B)
    # IMPORTANT: keep charpoly ONLY in lam
    print('M:', sp.expand(M.det().as_expr()))
    print('B:', sp.expand(B.det().as_expr()))
    p, q = clear_denominators(sp.expand(M.det().as_expr()), lam)
    s, t = clear_denominators(sp.expand(B.det().as_expr()), lam)
    print(f'p:{p}')
    print(f'q:{q}')
    print(f's:{s}')
    print(f't:{t}')

    g_lam = p*t
    h_lam = s*q

    # expand BEFORE substitution
    g_lam = sp.expand(g_lam)
    h_lam = sp.expand(h_lam)

    # substitute AFTER expansion (critical fix)
    g = g_lam.xreplace({lam: sp.I * omega})
    h = h_lam.xreplace({lam: sp.I * omega})
    g = sp.expand(sp.N(g))
    h = sp.expand(sp.N(h))

    # real/imag split
    g_re, g_im = sp.re(g), sp.im(g)
    h_re, h_im = sp.re(h), sp.im(h)

    g_im = sp.cancel(g_im / omega)
    h_im = sp.cancel(h_im / omega)

    # omega^2 substitution
    subs = {omega**2: x}

    g_re = sp.expand(g_re.subs(subs))
    g_im = sp.expand(g_im.subs(subs))
    h_re = sp.expand(h_re.subs(subs))
    h_im = sp.expand(h_im.subs(subs))

    print(f'gre: {g_re}')
    print(f'gim: {g_im}')
    print(f'hre: {h_re}')
    print(f'him: {h_im}')

    expr = sp.expand(g_re * h_im - g_im * h_re)

    poly = sp.Poly(expr, x)

    coeffs = np.array([complex(sp.N(c)) for c in poly.all_coeffs()], dtype=complex)/1e6
    print(coeffs)
    x_solutions = np.roots(coeffs)
    print('XSOLS:', x_solutions)

    return {
        "g_Re(x)": np.array(sp.poly(g_re).all_coeffs(), dtype=float),
        "g_Im(x)": np.array(sp.poly(g_im).all_coeffs(), dtype=float),
        "h_Re(x)": np.array(sp.poly(h_re).all_coeffs(), dtype=float),
        "h_Im(x)": np.array(sp.poly(h_im).all_coeffs(), dtype=float),
        "threshold_polynomial": np.array(sp.poly(expr).all_coeffs(), dtype=float),
        "omega2_solutions": np.array(x_solutions, dtype=complex),
    }


def extract_real_entries(arr, epsilon=1e-5):
    """
    Return all entries whose imaginary part is smaller than epsilon.
    Returned values are converted to real floats.
    """
    arr = np.asarray(arr, dtype=complex)

    mask = np.abs(arr.imag) < epsilon

    return arr.real[mask]


def Jacobian(N, z, d, t, mus, gs):
    J0 = construct_J0(mus, gs, t, symbolic=False)

    dL = dL_star(N, z, d)
    J = J0.copy()
    J[-1, 0] = dL / t
    if verbose: print(f'Jacobian:{J}')
    return J



def compute_eigs(N, mus, a, d, t, gs):
    roots = z_star(N, a, d)
    eigvals = []
    eigvecs = []
    if verbose: print('EIGENVALUES AND EIGENVECTORS:')
    for i, root in enumerate(roots):
        vals, vecs = np.linalg.eig(Jacobian(N, root, d, t, mus, gs))
        eigvals.append(vals)
        eigvecs.append(vecs)
        print(f'\troot {i}')
        if verbose:
            for j, (val, vec) in enumerate(zip(vals, vecs)):
                print(f'\t\tvalue {j}:{val}')
                print(f'\t\tvector {j}:{vec}')

    return roots, eigvals, eigvecs



# -----------------------------
# SYSTEM
# -----------------------------
def system(time, X, a, d, mus, gs, t):
    N = (len(X)-1)//2
    x = X[:-1:2]
    y = X[1::2]
    z = X[-1]
    dX = np.zeros(2*N+1)
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        dX[2*i] = y_i
        dX[2*i+1] = -1*gs[i]*y_i - mus[i]*x_i + mus[i]*z
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
    gs = np.array([1, 1,2, 6, 7, 7])
    mus = np.array([1, 2,6, 6, 9, 10])
    a = 1
    t = 2
    d = 1
    z = z_star(N, a, d)[0]
    # symbolic
    J0_sym = construct_J0(mus, gs, t, symbolic=True)

    # numeric numpy array
    J0_np = construct_J0(mus, gs, t, symbolic=False)

    print(type(J0_sym), J0_sym)
    print(type(J0_np), J0_np)

    result = derive_threshold_polynomials(
        mus,
        gs,
        t
    )
    print(result)