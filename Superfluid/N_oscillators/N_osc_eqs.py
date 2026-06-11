import warnings
import numpy as np
import sympy as sp
from scipy.special import jn_zeros
from scipy.optimize import root_scalar
from numba import njit


warnings.filterwarnings('ignore')  # Suppress all warnings

verbose = False

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


def mu_spectrum_harmonic(i):
    return np.sqrt(np.arange(1, i+1))


# -----------------------------
# lasing threshold
# -----------------------------
def find_pure_imag_crossings(mus, gs, t,
                              dL_min, dL_max,
                              num_scan_points=250):

    J0 = construct_J0(mus, gs, t)

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



def lasing_threshold(N, d, t, mus, gs, num_scan_points=250, as_func_off='delta', delta_effs=None):

    if isinstance(d, (np.ndarray, list)):
        d_max = np.max(np.abs([d[0], d[-1]]))
    else:
        d_max = d

    dL_min = (1-np.sqrt(1+d_max**2))/2

    dL_max = 0

    dL_sols = find_pure_imag_crossings(mus, gs, t, dL_min, dL_max, num_scan_points=num_scan_points)
    if verbose: print(f'\tdL solutions: {dL_sols} {np.shape(dL_sols)}')

    if len(dL_sols) == 0:
        return []

    z_sols = []

    if as_func_off=='delta':
        for sol in dL_sols:
            D = d**2 - sol*(sol+2)
            z_sols.append( (-d*(sol+1) + np.sqrt(D)) / (N*(sol+2)) )
            z_sols.append( (-d*(sol+1) - np.sqrt(D)) / (N*(sol+2)) )

        z_sols = np.array(z_sols)

        if len(z_sols) == 0:
            return []

        thresholds = z_sols * ((N*z_sols + d)**2 + 1)

    elif as_func_off=='delta_eff':
        for sol in dL_sols:
            z_sols.append(sol*(delta_effs**2+1) / (-2*N*delta_effs))
            
        z_sols = np.array(z_sols)

        if len(z_sols) == 0:
            return []

        thresholds = z_sols * (delta_effs**2 + 1)

    if verbose: print(f'\tz solutions shape:{np.shape(z_sols)}')    
    if verbose: print(f'\tthresholds shape:{np.shape(thresholds)}')
    
    thresholds_filtered = thresholds#filter_arrays(thresholds)
    if verbose: print(f'\tthresholds shape (after filtering):{np.shape(thresholds_filtered)}')
    if len(thresholds_filtered) == 0:
        return []

    thresholds_sorted = sorted(thresholds_filtered, key=lambda a: np.min(a))

    return thresholds_sorted


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
    return roots

def z_star_eff(a, d_eff):
    return a / (d_eff**2 + 1)


@njit
def z_star_numba(N, a, d):
    # 1. Divide by N^2 to get a monic cubic: x^3 + b*x^2 + c*x + d_val = 0
    b = (2.0 * d) / N
    c = (d**2 + 1.0) / (N**2)
    d_val = -a / (N**2)
    
    # 2. Change variables (x = y - b/3) to get depressed cubic: y^3 + p*y + q = 0
    p = c - (b**2) / 3.0
    q = (2.0 * b**3) / 27.0 - (b * c) / 3.0 + d_val
    
    # 3. Calculate the discriminant
    discriminant = (q / 2.0)**2 + (p / 3.0)**3
    shift = b / 3.0
    
    if discriminant > 0:
        # One real root, two complex roots
        sqrt_disc = np.sqrt(discriminant)
        u = np.cbrt(-q / 2.0 + sqrt_disc)
        v = np.cbrt(-q / 2.0 - sqrt_disc)
        
        # Return the single real root
        return np.array([u + v - shift])
        
    elif discriminant == 0:
        # All roots are real, and at least two are equal
        if p == 0 and q == 0:
            return np.array([-shift])
        
        root1 = 3.0 * q / p - shift
        root2 = -1.5 * q / p - shift
        # Return unique real roots
        if abs(root1 - root2) < 1e-12:
            return np.array([root1])
        return np.array([root1, root2])
        
    else:
        # discriminant < 0: Three distinct real roots (Trigonometric solution)
        r = np.sqrt(-(p**3) / 27.0)
        phi = np.arccos(-q / (2.0 * r))
        
        root1 = 2.0 * np.cbrt(r) * np.cos(phi / 3.0) - shift
        root2 = 2.0 * np.cbrt(r) * np.cos((phi + 2.0 * np.pi) / 3.0) - shift
        root3 = 2.0 * np.cbrt(r) * np.cos((phi + 4.0 * np.pi) / 3.0) - shift
        
        return np.array([root1, root2, root3])


def dL_star(N, z, d):
    return -2*N*z*(N*z + d) / ((N*z + d)**2 + 1)


# -----------------------------
# Jacobian
# -----------------------------
def construct_J0(mu_vals, gamma_vals, tau):
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



def extract_real_entries(arr, epsilon=1e-5):
    """
    Return all entries whose imaginary part is smaller than epsilon.
    Returned values are converted to real floats.
    """
    arr = np.asarray(arr, dtype=complex)

    mask = np.abs(arr.imag) < epsilon

    return arr.real[mask]


def Jacobian(N, z, d, t, mus, gs):
    J0 = construct_J0(mus, gs, t)

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
        if verbose: print(f'\troot {i}')
        if verbose:
            for j, (val, vec) in enumerate(zip(vals, vecs)):
                print(f'\t\tvalue {j}:{val}')
                print(f'\t\tvector {j}:{vec}')

    return np.array(roots), np.array(eigvals), np.array(eigvecs)



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


@njit
def system_core(X, a, d, mus, gs, t, out):
    N = (len(X)-1)//2

    x = X[0:2*N:2]
    y = X[1:2*N:2]
    z = X[-1]

    for i in range(N):
        out[2*i] = y[i]
        out[2*i+1] = -gs[i]*y[i] - mus[i]*x[i] + mus[i]*z

    s = 0.0
    for i in range(N):
        s += x[i]

    out[-1] = a / ((s + d)*(s + d) + 1)/t - z/t


def system_numba(t, X, a, d, mus, gs, tau):
    out = np.zeros_like(X)
    system_core(X, a, d, mus, gs, tau, out)
    return out



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


def project_onto_line(x, v):
    """
    Project vector x onto the line spanned by vector v.

    Parameters
    ----------
    x : array-like
        Vector to be projected
    v : array-like
        Direction vector of the line

    Returns
    -------
    numpy.ndarray
        Projection of x onto span(v)
    """
    x = np.asarray(x)
    v = np.asarray(v)

    return (np.dot(x, v) / np.dot(v, v)) * v


def transform_matrix(N):
    """
    Construct the (2N+1)x(2N+1) transformation matrix for

    (x1,y1,...,xN,yN,z) -> (X,Y,u2,v2,...,uN,vN,z)

    where
        X = (1/N) sum_i x_i
        Y = (1/N) sum_i y_i
        u_i = x_i - X
        v_i = y_i - Y
    """
    M = np.zeros((2*N + 1, 2*N + 1))

    # Row for X
    for i in range(N):
        M[0, 2*i] = 1

    # Row for Y
    for i in range(N):
        M[1, 2*i + 1] = 1

    # Rows for u_i, v_i (i = 2,...,N)
    for i in range(1, N):   # zero-based: i=1 corresponds to u2,v2
        row_u = 2*i
        row_v = 2*i + 1

        # u_i = x_i - X = x_i - (1/N) sum_j x_j
        for j in range(N):
            M[row_u, 2*j] = -1
        M[row_u, 2*i] = N - 1

        # v_i = y_i - Y = y_i - (1/N) sum_j y_j
        for j in range(N):
            M[row_v, 2*j + 1] = -1
        M[row_v, 2*i + 1] = N - 1

    # z unchanged
    M[-1, -1] = N

    return M / N


def inverse_transform_matrix(N):
    """
    Construct the inverse transformation matrix for

    (X,Y,u2,v2,...,uN,vN,z) -> (x1,y1,...,xN,yN,z)

    Returns a (2N+1)x(2N+1) matrix.
    """
    M = np.zeros((2*N + 1, 2*N + 1))

    # x1 = X - sum_{i=2}^N u_i
    M[0, 0] = 1
    for i in range(1, N):
        M[0, 2*i] = -1

    # y1 = Y - sum_{i=2}^N v_i
    M[1, 1] = 1
    for i in range(1, N):
        M[1, 2*i + 1] = -1

    # xi = X + ui, yi = Y + vi  for i=2,...,N
    for i in range(1, N):
        row_x = 2*i
        row_y = 2*i + 1

        M[row_x, 0] = 1          # X contribution
        M[row_x, 2*i] = 1        # ui contribution

        M[row_y, 1] = 1          # Y contribution
        M[row_y, 2*i + 1] = 1    # vi contribution

    # z unchanged
    M[-1, -1] = 1

    return M
            




if __name__ == '__main__':
    N = 2
    gs = np.array([1, 1,2, 6, 7, 7])
    mus = np.array([1, 2,6, 6, 9, 10])
    a = 1
    t = 2
    d = 1
    z = z_star(N, a, d)[0]