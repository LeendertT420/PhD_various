import warnings
import numpy as np
import sympy as sp
from scipy.special import jn_zeros
from scipy.optimize import root_scalar
from numba import njit

from scipy.integrate import solve_ivp
from scipy.optimize import root

warnings.filterwarnings('ignore')  # Suppress all warnings

verbose = False

np.set_printoptions(precision=5)



@njit(fastmath=True)
def ode_vector_field(t, state, N, gamma, mu, tau, alpha, delta, sigma, chi):
    """
    Highly optimized, raw math vector field. 
    State layout: [x_1...x_N, y_1...y_N, z] (Size: 2N + 1)
    """
    # Unpack state
    x = state[0:N]
    y = state[N:2*N]
    z = state[2*N]
    
    # Preallocate derivatives
    derivatives = np.empty(2 * N + 1)
    dx = derivatives[0:N]
    dy = derivatives[N:2*N]
    
    # 1. dx_i/dt = y_i
    dx[:] = y
    
    # 2. Compute the quadratic interaction tensor term: (mu_i / sigma) * sum(chi_ijk * x_j * x_k)
    # Fast nested loops are completely fine because Numba compiles them to raw C loops.
    tensor_term = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            # Only loop over unique pairs j <= k
            # j == k case (appears 1 time)
            s += chi[i, j, j] * x[j] * x[j]
            
            # j < k case (appears 2 times: jk and kj)
            for k in range(j + 1, N):
                s += 2.0 * chi[i, j, k] * x[j] * x[k]
                
        tensor_term[i] = (mu[i] / sigma) * s

    # 3. dy_i/dt
    for i in range(N):
        dy[i] = -gamma[i]*y[i] - mu[i]*x[i] + mu[i]*z + tensor_term[i]
        
    # 4. dz/dt
    sum_x = np.sum(x)
    dz = (1.0 / tau) * ((alpha / ((delta + sum_x)**2 + 1.0)) - z)
    derivatives[2*N] = dz
    
    return derivatives



def fixed_points_num(params, num_tries=30):
    """
    Finds fixed points numerically and filters them based on physical bounds:
    0 < x_i < sigma  and  0 < z < sigma (with y_i = 0).
    """
    N = params['N']
    sigma = params['sigma']
    
    def steady_state_objective(vars_xz):
        x_val = vars_xz[0:N]
        z_val = vars_xz[-1]
        U = np.concatenate([x_val, np.zeros(N), [z_val]])
        derivs = system(0, U, params)
        return np.concatenate([derivs[N:2*N], [derivs[-1]]])

    valid_solutions = []

    for x in fixed_points_0th_order(params):
        initial_guess = np.concatenate([np.full(N, x), [x]])
        
        res = root(steady_state_objective, initial_guess, method='lm')
        
        if res.success and np.linalg.norm(res.fun) < 1e-10:
            sol_x = res.x[0:N]
            sol_z = res.x[-1]
            
            if np.all(sol_x >= 0) and np.all(sol_x < 10*sigma):
                if (sol_z >= 0) and (sol_z < 10*sigma):
                    full_fixed_point = np.concatenate([sol_x, np.zeros(N), [sol_z]])
                    
                    if not any(np.allclose(full_fixed_point, sol, atol=1e-8) for sol in valid_solutions):
                        valid_solutions.append(full_fixed_point)
                        
    return valid_solutions



def fixed_points_0th_order(params):
    N = params['N']
    d = params['delta']
    a = params['alpha']

    roots = np.roots([N**2, 2*N*d, d**2 + 1, -a])
    roots = np.real(roots[np.isreal(roots)])

    return roots


def fixed_points_1st_order(params):
    points_0th_order = fixed_points_0th_order(params)

    N = params['N']
    d = params['delta']
    chi = params['chi'][:N,:N,:N]
    Lambda = np.sum(chi, axis=(1, 2))
    eps = 1/params['sigma']

    Lambda_bar = np.sum(chi)

    points_1st_order = []
    for x in points_0th_order:
        dL_bar = (2*x*(N*x+d)*Lambda_bar) / ((3*N*x+d)*(N*x+d)+1)

        x_star = x + eps * x**2 * ( Lambda - dL_bar )
        z_star = x - eps * x**2 * dL_bar

        points_1st_order.append(np.concatenate([x_star, [z_star]]))
    return points_1st_order




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



def dL_star(x_star, params):
    N = params['N']
    alpha = params['alpha']
    tau = params['tau']
    delta = params['delta']

    x = x_star[:N]
    q = np.sum(x) + delta

    return (-2*alpha/tau) * q / ( q**2 + 1 )**2




def construct_jacobian(x_star, params, modecoupling=True, opticalcoupling=True):
    """
    Construct the block Jacobian matrix

        J = [[0, I, 0],
             [A, Γ, μ],
             [L, 0, -1/tau]]

    Parameters
    ----------
    gamma : (N,) array_like
        Vector γ_i

    mu : (N,) array_like
        Vector μ_i

    x_star : (N,) array_like
        Fixed point vector x*_i

    chi : (N, N, N) array_like
        Tensor χ_ijk

    sigma : float
        Scalar σ

    tau : float
        Scalar τ

    dL : (N,) array_like
        Vector of partial derivatives:
            dL_j = ∂L*/∂x_j / τ

    Returns
    -------
    J : (2N+1, 2N+1) ndarray
        Full Jacobian matrix
    """
    N = params['N']
    mu = params['mu']
    gamma = params['gamma']
    chi = params['chi'][:N,:N,:N]
    sigma = params['sigma']
    tau = params['tau']

    if opticalcoupling:
        dL = dL_star(x_star, params)

    # --- Block matrices ---

    # 0 block
    O = np.zeros((N, N))

    # Identity block
    I = np.eye(N)

    # Gamma block
    Gamma = -np.diag(gamma)

    # mu column block
    mu_col = mu.reshape(N, 1)

    # L row block
    if opticalcoupling:
        L_row = np.full((1, N), dL/tau)
    elif not opticalcoupling:
        L_row = np.full((1, N), 0)

    # A block
    A = np.zeros((N, N))

    if modecoupling:
        for i in range(N):
            for j in range(N):
                interaction_sum = np.sum(chi[i, j, :] * x_star[:N])

                A[i, j] = (
                    -mu[i] * (i == j)
                    + (2.0 * mu[i] / sigma) * interaction_sum
                )
    elif not modecoupling:
        for i in range(N):
            A[i, i] = -mu[i]

    # --- Assemble full Jacobian ---

    top = np.hstack([O, I, np.zeros((N, 1))])

    middle = np.hstack([A, Gamma, mu_col])

    bottom = np.hstack([L_row, np.zeros((1, N)), np.array([[-1.0 / tau]])])

    J = np.vstack([top, middle, bottom])

    return J


# -----------------------------
# lasing threshold
# -----------------------------
def find_pure_imag_crossings(params, dL_min, dL_max, num_scan_points=250):

    J0 = construct_jacobian(None, params, modecoupling=False, opticalcoupling=False)

    N = params['N']
    tau = params['tau']

    def eigen_decomposition(dL):
        N = int((np.shape(J0)[0]-1)/2)
        J = J0.copy()
        J[-1, 0:N] = np.full((1, N), dL/tau)
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



def lasing_threshold(params, deltas=None, num_scan_points=250, as_func_off='delta', delta_effs=None):
    N = params['N']
    d = params['delta']

    if isinstance(deltas, (np.ndarray, list)):
        d_max = np.max(np.abs([deltas[0], deltas[-1]]))
    else:
        d_max = d

    dL_min = (1-np.sqrt(1+d_max**2))/2
    dL_max = 0

    dL_sols = find_pure_imag_crossings(params, dL_min, dL_max, num_scan_points=num_scan_points)
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
# Jacobian
# -----------------------------

def extract_real_entries(arr, epsilon=1e-5):
    """
    Return all entries whose imaginary part is smaller than epsilon.
    Returned values are converted to real floats.
    """
    arr = np.asarray(arr, dtype=complex)

    mask = np.abs(arr.imag) < epsilon

    return arr.real[mask]

def compute_eigs(params):
    roots = fixed_points_num(params)

    eigvals = []
    eigvecs = []
    if verbose: print('EIGENVALUES AND EIGENVECTORS:')
    for i, root in enumerate(roots):

        dL = dL_star(root, params)
        vals, vecs = np.linalg.eig(construct_jacobian(root, params, dL))
        eigvals.append(vals)
        eigvecs.append(vecs)
        if verbose: print(f'\troot {i}')
        if verbose:
            for j, (val, vec) in enumerate(zip(vals, vecs)):
                print(f'\t\tvalue {j}:{val}')
                print(f'\t\tvector {j}:{vec}')

    return np.array(roots), np.array(eigvals), np.array(eigvecs)



