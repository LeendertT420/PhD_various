import warnings
from typing import Any, List, Literal

import numpy as np
from numpy.typing import NDArray
from numba import njit
import numba

from scipy.optimize import root, root_scalar
from scipy.special import jn_zeros, jnp_zeros, jn

warnings.filterwarnings('ignore')  # Suppress all warnings

verbose = False
#np.set_printoptions(precision=5)


# -----------------------------
# Frequency spectrum
# -----------------------------

def zeta(i : int) -> float:
    return jn_zeros(1, i)[-1]


def mu_spectrum(i : int) -> NDArray[np.float64]:
    return ( jn_zeros(1, i) / jn_zeros(1, 1) )**2


def mu_spectrum_harmonic(i : int) -> NDArray[np.float64]:
    return np.sqrt(np.arange(1, i+1))


def get_bessel_profile(i, N_points):
    zeta_i = jnp_zeros(0, i)[-1]
    r = np.linspace(0, 1, N_points)
    return r, jn(0, zeta_i * r) / jn(0, zeta_i)


def prepare_spatial_vdw_weights(r):
    """
    Precomputes mode matrices and spatial integration weights once.
    """
    # Integration weights w(r) * r
    w = np.empty_like(r)
    w[0] = 0.5 * (r[1] - r[0])
    w[-1] = 0.5 * (r[-1] - r[-2])
    w[1:-1] = 0.5 * (r[2:] - r[:-2])
    W_r = w * r
    
    W3 = 4 * W_r
    W4 = 20/3 * W_r
    
    return W3, W4




# -----------------------------
# System
# -----------------------------

def system(t : float,
           state : NDArray[np.float64],
           params : dict[str, int | float | NDArray[np.float64]],
           use_3d : bool = True,
           use_4d : bool = True,
           delta_sweep : bool = False,
           delta_start : float = None,
           delta_stop : float = None,
           sweepspeed : float = None) -> NDArray[np.float64]:
    '''
    Computes the ODE system with conditional quadratic and cubic non-linear mode interactions.

        x_dot_i = y_i

        y_dot_i = -gamma_i y_i
                  - mu_i x_i
                  + mu_i z
                  + [optional] (mu_i / sigma) * sum_{j,k} chi_ijk x_j x_k
                  - [optional] (mu_i / sigma^2) * sum_{j,k,l} chi_ijkl x_j x_k x_l

        z_dot = (1/tau) * (
                    alpha / ((delta + sum_i x_i)^2 + 1)
                    - z
                )

    Parameters
    ----------
    t : float
        Time (unused, included for solve_ivp compatibility)

    state : ndarray, shape (2N + 1,)
        State vector:
            state = [x_1,...,x_N, y_1,...,y_N, z]

    N : int
        Number of modes
    gamma : ndarray, shape (N,)
    mu : ndarray, shape (N,)
    tau : float
    alpha : float
    delta : float
    sigma : float
    chi_ijk : ndarray, shape (N,N,N)
        3D cubic mode interaction tensor
    chi_ijkl : ndarray, shape (N,N,N,N)
        4D quartic mode interaction tensor
    use_3d : bool, default True
        Flag to include or ignore the cubic potential interaction (3D tensor contraction)
    use_4d : bool, default True
        Flag to include or ignore the quartic potential interaction (4D tensor contraction)

    Returns
    -------
    dstate_dt : ndarray, shape (2N + 1,)
    '''

    N = params['N']

    # unpack state
    x = state[:N]
    y = state[N:2*N]
    z = state[-1]

    # --- x equations ---
    x_dot = y

    # --- nonlinear interaction terms ---
    if use_4d:
        use_3d = True
        interaction_4D = np.einsum('ijkl,j,k,l->i', params['chi_ijkl'][:N, :N, :N, :N], x, x, x)
    else:
        interaction_4D = np.zeros(N)

    if use_3d:
        interaction_3D = np.einsum('ijk,j,k->i', params['chi_ijk'][:N, :N, :N], x, x)
    else:
        interaction_3D = np.zeros(N)
        
    
    # --- y equations ---
    y_dot = (
        -params['gamma'] * y
        -params['mu'] * x
        +params['mu'] * params['xi'] * z
        +(params['mu']  / params['sigma'] ) * interaction_3D
        -(params['mu']  / (params['sigma'] **2)) * interaction_4D
    )
    #print('dissipation:', -params['gamma'] * y)
    #print('linear spring:', -params['mu'] * x)
    #print('driving:', +params['mu'] * params['xi'] * z)
    #print('3D:', (params['mu']  / params['sigma'] ) * interaction_3D)

    # --- z equation ---
    if delta_sweep:

        current_delta = delta_start - sweepspeed * t
        
        if current_delta < delta_stop:
            delta = delta_stop
        else:
            delta = current_delta
    else:
        delta = params['delta']

    z_dot = (1.0 / params['tau']) * (
        params['alpha'] / ((delta + np.sum(x))**2 + 1.0) - z
    )

    # concatenate into one vector
    dstate_dt = np.concatenate([
        x_dot,
        y_dot,
        np.array([z_dot])
    ])

    return dstate_dt



def compute_full_optomechanical_energy(sol_y, params, use_3d=True, use_4d=True):
    """
    Computes Mechanical, Optical, and Interaction Energies over time.
    
    sol_y : shape (2*N + 1, N_points)
            sol_y[:N]     -> x_i (displacements)
            sol_y[N:2*N]  -> y_i (velocities)
            sol_y[2*N]    -> z   (optical mode / cavity field)
    """
    N = params['N']
    x = sol_y[:N, :]
    y = sol_y[N:2*N, :]
    z = sol_y[2*N, :]  # Optical mode variable
    
    mu = params['mu'][:, None]
    
    # 1. Mechanical Energies
    T = 0.5 * np.sum(y**2, axis=0)                  # Kinetic
    V_0 = 0.5 * np.sum(mu * (x**2), axis=0)         # Linear Potential
    
    # Non-linear mechanical potentials
    V_3 = (1.0 / 3.0) * np.einsum('ijk,it,jt,kt->t', params['chi_ijk'], x, x, x) if use_3d else 0.0
    V_4 = (1.0 / 4.0) * np.einsum('ijkl,it,jt,kt,lt->t', params['chi_ijkl'], x, x, x, x) if use_4d else 0.0
    
    E_mech = T + V_0 + V_3 + V_4
    
    # 2. Optical Energy
    # Assuming z is the complex field amplitude (or real photon number if z >= 0)
    photon_number = np.abs(z)**2 if np.iscomplexobj(z) else z**2
    
    # In unitless form: E_opt = detuning * photon_number (or bare cavity energy)
    delta_eff = params.get('delta', 1.0)
    E_opt = -delta_eff * photon_number
    
    # 3. Optomechanical Interaction Energy
    # g_i is the coupling vector (e.g., params['xi'] or optomechanical coupling g)
    g_i = params.get('xi', np.ones(N))[:, None]
    V_int = 0#-np.sum(g_i * x, axis=0) * photon_number
    
    # Total Energy
    E_total = E_mech + E_opt #+ V_int
    
    return {
        'E_total': E_total,
        'E_mech': E_mech,
        'E_opt': E_opt,
        'V_int': V_int,
        'photons': photon_number}



def system_non_adiabatic(t : float,
           state : NDArray[np.float64],
           params : dict[str, int | float | NDArray[np.float64]],
           use_3d : bool = True,
           use_4d : bool = True,
           delta_sweep : bool = False,
           delta_start : float = None,
           delta_stop : float = None,
           sweepspeed : float = None) -> NDArray[np.float64]:
    '''
    Computes the ODE system with conditional quadratic and cubic non-linear mode interactions.

        x_dot_i = y_i

        y_dot_i = -gamma_i y_i
                  - mu_i x_i
                  + mu_i z
                  + [optional] (mu_i / sigma) * sum_{j,k} chi_ijk x_j x_k
                  - [optional] (mu_i / sigma^2) * sum_{j,k,l} chi_ijkl x_j x_k x_l

        z_dot = (1/tau) * (
                    alpha / ((delta + sum_i x_i)^2 + 1)
                    - z
                )

    Parameters
    ----------
    t : float
        Time (unused, included for solve_ivp compatibility)

    state : ndarray, shape (2N + 1,)
        State vector:
            state = [x_1,...,x_N, y_1,...,y_N, z]

    N : int
        Number of modes
    gamma : ndarray, shape (N,)
    mu : ndarray, shape (N,)
    tau : float
    alpha : float
    delta : float
    sigma : float
    chi_ijk : ndarray, shape (N,N,N)
        3D cubic mode interaction tensor
    chi_ijkl : ndarray, shape (N,N,N,N)
        4D quartic mode interaction tensor
    use_3d : bool, default True
        Flag to include or ignore the cubic potential interaction (3D tensor contraction)
    use_4d : bool, default True
        Flag to include or ignore the quartic potential interaction (4D tensor contraction)

    Returns
    -------
    dstate_dt : ndarray, shape (2N + 1,)
    '''

    N = params['N']

    # unpack state
    x = state[:N]
    y = state[N:2*N]
    u = state[2*N]
    v = state[2*N+1]
    z = state[2*N+2]

    # --- x equations ---
    x_dot = y

    # --- nonlinear interaction terms ---
    if use_4d:
        use_3d = True
        interaction_4D = np.einsum('ijkl,j,k,l->i', params['chi_ijkl'][:N, :N, :N, :N], x, x, x)
    else:
        interaction_4D = np.zeros(N)

    if use_3d:
        interaction_3D = np.einsum('ijk,j,k->i', params['chi_ijk'][:N, :N, :N], x, x)
    else:
        interaction_3D = np.zeros(N)
        
    
    # --- y equations ---
    y_dot = (
        -params['gamma'] * y
        -params['mu'] * x
        +params['mu'] * params['xi'] * z
        +(params['mu']  / params['sigma'] ) * interaction_3D
        -(params['mu']  / (params['sigma'] **2)) * interaction_4D
    )

    # --- z equation ---
    if delta_sweep:

        current_delta = delta_start - sweepspeed * t
        
        if current_delta < delta_stop:
            delta = delta_stop
        else:
            delta = current_delta
    else:
        delta = params['delta']

    sum_x = np.sum(x)
    eff_detuning = params['delta'] + sum_x
    
    u_dot = (1.0 / params['nu']) * (-u - eff_detuning * v + np.sqrt(params['alpha']))
    v_dot = (1.0 / params['nu']) * (-v + eff_detuning * u)
    z_dot = (1.0 / params['tau']) * (u**2 + v**2 - z)

    # concatenate into one vector (2N + 3)
    dstate_dt = np.concatenate([
        x_dot,
        y_dot,
        np.array([u_dot, v_dot, z_dot])
    ])

    return dstate_dt


@njit(fastmath=True)
def _system_numba_core(t : float,
                       state : NDArray[np.float64],
                       N : int,
                       gamma : NDArray[np.float64],
                       mu : NDArray[np.float64],
                       tau : float,
                       alpha : float,
                       delta : float,
                       sigma : float,
                       chi_ijk : NDArray[np.float64],
                       chi_ijkl : NDArray[np.float64],
                       xi : NDArray[np.float64],
                       use_3d : bool = True,
                       use_4d : bool = True) -> NDArray[np.float64]:
    '''
    Core numerical backend compiled to raw machine code.
    '''
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
    
    # 2. Compute the interaction terms
    tensor_term = np.zeros(N)
    sigma_sq = sigma * sigma
    
    for i in range(N):
        # --- 3D Tensor Contraction (Cubic Energy Term) ---
        s_3d = 0.0
        if use_3d:
            for j in range(N):
                s_3d += chi_ijk[i, j, j] * x[j] * x[j]
                for k in range(j + 1, N):
                    s_3d += 2.0 * chi_ijk[i, j, k] * x[j] * x[k]
                
        # --- 4D Tensor Contraction (Quartic Energy Term) ---
        s_4d = 0.0
        if use_4d:
            for j in range(N):
                s_4d += chi_ijkl[i, j, j, j] * x[j] * x[j] * x[j]
                
                for k in range(j + 1, N):
                    s_4d += 3.0 * chi_ijkl[i, j, j, k] * x[j] * x[j] * x[k]
                    s_4d += 3.0 * chi_ijkl[i, j, k, k] * x[j] * x[k] * x[k]
                    
                    for l in range(k + 1, N):
                        s_4d += 6.0 * chi_ijkl[i, j, k, l] * x[j] * x[k] * x[l]
                    
        # Combine non-linear forces with their respective scaling factors
        tensor_term[i] = (mu[i] / sigma) * s_3d - (mu[i] / sigma_sq) * s_4d

    # 3. dy_i/dt
    for i in range(N):
        dy[i] = -gamma[i]*y[i] - mu[i]*x[i] + mu[i]*xi[i]*z + tensor_term[i]
        
    # 4. dz/dt
    sum_x = np.sum(x)
    dz = (1.0 / tau) * ((alpha / ((delta + sum_x)**2 + 1.0)) - z)
    derivatives[2*N] = dz
    
    return derivatives


def system_numba(t : float,
                 state : NDArray[np.float64],
                 params : dict[str, int | float | NDArray[np.float64]],
                 use_3d : bool = True,
                 use_4d : bool = True) -> NDArray[np.float64]:
    '''
    User-facing ODE function that accepts the params dictionary 
    and handles solve_ivp compatibility cleanly.
    '''
    N = params['N']
    if use_4d:
        use_3d = True
    if not use_3d:
        use_4d = False
    
    return _system_numba_core(
        t=t,
        state=state,
        N=N,
        gamma=params['gamma'],
        mu=params['mu'],
        tau=params['tau'],
        alpha=params['alpha'],
        delta=params['delta'],
        sigma=params['sigma'],
        chi_ijk=params['chi_ijk'][:N, :N, :N],
        chi_ijkl=params['chi_ijkl'][:N, :N, :N, :N],
        xi=params['xi'],
        use_3d=use_3d,
        use_4d=use_4d
    )



@njit(fastmath=True)
def _system_numba_hidde_core(t : float,
                       state : NDArray[np.float64],
                       N : int,
                       gamma : NDArray[np.float64],
                       mu : NDArray[np.float64],
                       tau : float,
                       alpha : float,
                       delta : float,
                       sigma : float,
                       xi : NDArray[np.float64],
                       profiles,
                       W3,
                       W4,
                       use_3d : bool = True,
                       use_4d : bool = True) -> NDArray[np.float64]:
    '''
    Core numerical backend compiled to raw machine code.
    '''
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
    
    # 2. Compute the interaction terms
    sigma_sq = sigma * sigma
    
    X = profiles @ x
    M = profiles.shape[0]
    
    # Vectorized element-wise operations
    X2_W3 = (X**2) * W3
    X3_W4 = (X**3) * W4

    # 4. Accelerations dv/dt
    for i in range(N):

        # 3rd Order Van der Waals Force Projection
        if use_3d:
            f3_i = 0.0
            for m in range(M):
                f3_i += profiles[m, i] * X2_W3[m]

        # 4th Order Van der Waals Force Projection
        if use_4d:
            f4_i = 0.0
            for m in range(M):
                f4_i += profiles[m, i] * X3_W4[m]

        dy[i] = -gamma[i]*y[i] - mu[i]*x[i] + mu[i]*xi[i]*z + (mu[i] / sigma) * f3_i - (mu[i] / sigma_sq) * f4_i
        
    # 4. dz/dt
    sum_x = np.sum(x)
    dz = (1.0 / tau) * ((alpha / ((delta + sum_x)**2 + 1.0)) - z)
    derivatives[2*N] = dz
    
    return derivatives


def system_numba_hidde(t : float,
                 state : NDArray[np.float64],
                 params : dict[str, int | float | NDArray[np.float64]],
                 profiles, W3, W4,
                 use_3d : bool = True,
                 use_4d : bool = True) -> NDArray[np.float64]:
    '''
    User-facing ODE function that accepts the params dictionary 
    and handles solve_ivp compatibility cleanly.
    '''
    N = params['N']
    if use_4d:
        use_3d = True
    if not use_3d:
        use_4d = False
    
    return _system_numba_hidde_core(
        t=t,
        state=state,
        N=N,
        gamma=params['gamma'],
        mu=params['mu'],
        tau=params['tau'],
        alpha=params['alpha'],
        delta=params['delta'],
        sigma=params['sigma'],
        xi=params['xi'],
        profiles=profiles,
        W3=W3,
        W4=W4,
        use_3d=use_3d,
        use_4d=use_4d
    )



@njit(fastmath=True)
def _system_numba_core_fast(
    t: float,
    state: NDArray[np.float64],
    derivatives: NDArray[np.float64],  # Pre-allocated output array passed in!
    N: int,
    gamma: NDArray[np.float64],
    mu: NDArray[np.float64],
    tau: float,
    alpha: float,
    delta: float,
    sigma: float,
    chi_ijk: NDArray[np.float64],
    chi_ijkl: NDArray[np.float64],
    xi: NDArray[np.float64],
    use_3d: bool = True,
    use_4d: bool = True
) -> None:
    """
    Zero-allocation Numba core that computes RHS directly into `derivatives`.
    """
    # Unpack state pointers without slicing allocations
    z = state[2 * N]
    sigma_inv = 1.0 / sigma
    sigma_sq_inv = sigma_inv * sigma_inv

    # 1. dx_i/dt = y_i
    for i in range(N):
        derivatives[i] = state[N + i]

    # Compute sum_x inline for z equation
    sum_x = 0.0
    for i in range(N):
        sum_x += state[i]

    # 2 & 3. Compute dy_i/dt directly without intermediate tensor_term array
    for i in range(N):
        s_3d = 0.0
        if use_3d:
            for j in range(N):
                xj = state[j]
                s_3d += chi_ijk[i, j, j] * xj * xj
                for k in range(j + 1, N):
                    s_3d += 2.0 * chi_ijk[i, j, k] * xj * state[k]

        s_4d = 0.0
        if use_4d:
            for j in range(N):
                xj = state[j]
                xj_sq = xj * xj
                s_4d += chi_ijkl[i, j, j, j] * xj_sq * xj
                
                for k in range(j + 1, N):
                    xk = state[k]
                    s_4d += 3.0 * chi_ijkl[i, j, j, k] * xj_sq * xk
                    s_4d += 3.0 * chi_ijkl[i, j, k, k] * xj * xk * xk
                    
                    for l in range(k + 1, N):
                        s_4d += 6.0 * chi_ijkl[i, j, k, l] * xj * xk * state[l]

        # Combine nonlinear terms directly into dy[i]
        mu_i = mu[i]
        tensor_term_i = (mu_i * sigma_inv) * s_3d - (mu_i * sigma_sq_inv) * s_4d
        
        # dy_i/dt
        derivatives[N + i] = -gamma[i] * state[N + i] - mu_i * state[i] + mu_i * xi[i] * z + tensor_term_i

    # 4. dz/dt
    denom = delta + sum_x
    derivatives[2 * N] = (1.0 / tau) * ((alpha / (denom * denom + 1.0)) - z)



def system_numba_fast(t : float,
                 state : NDArray[np.float64],
                 params : dict[str, int | float | NDArray[np.float64]],
                 use_3d : bool = True,
                 use_4d : bool = True) -> NDArray[np.float64]:
    '''
    User-facing ODE function that accepts the params dictionary 
    and handles solve_ivp compatibility cleanly.
    '''
    N = params['N']
    if use_4d:
        use_3d = True
    if not use_3d:
        use_4d = False
    
    return _system_numba_core_fast(
        t=t,
        state=state,
        N=N,
        gamma=params['gamma'],
        mu=params['mu'],
        tau=params['tau'],
        alpha=params['alpha'],
        delta=params['delta'],
        sigma=params['sigma'],
        chi_ijk=params['chi_ijk'][:N, :N, :N],
        chi_ijkl=params['chi_ijkl'][:N, :N, :N, :N],
        xi=params['xi'],
        use_3d=use_3d,
        use_4d=use_4d
    )


# -----------------------------
# Fixed points
# -----------------------------

def fixed_points_num(system_func, 
                     args,
                     num_tries : int = 30,
                     tolerance : float = 1e-10) -> list[float]:
    '''
    Finds fixed points numerically and filters them based on physical bounds:
    0 < x_i < sigma  and  0 < z < sigma (with y_i = 0).
    '''
    params = args[0]
    N = params['N']
    
    def steady_state_objective(vars_xz):
        x_val = vars_xz[0:N]
        z_val = vars_xz[-1]
        U = np.concatenate([x_val, np.zeros(N), [z_val]])

        derivs = system_func(0, U, *args)
        return np.concatenate([derivs[N:2*N], [derivs[-1]]])

    valid_solutions = []

    for x in fixed_points_0th_order(params):
        initial_guess = np.concatenate([np.full(N, x), [x]])
        
        res = root(steady_state_objective, initial_guess, method='lm')
        
        if res.success and np.linalg.norm(res.fun) < tolerance:
            sol_x = res.x[0:N]
            sol_z = res.x[-1]
            
            if np.all(sol_x >= 0) and np.all(sol_x < 2*params['sigma']):
                if (sol_z >= 0) and (sol_z < 2*params['sigma']):
                    full_fixed_point = np.concatenate([sol_x, np.zeros(N), [sol_z]])
                    
                    if not any(np.allclose(full_fixed_point, sol, atol=1e-8) for sol in valid_solutions):
                        valid_solutions.append(full_fixed_point)
                        
    return valid_solutions


def fixed_points_0th_order(params : dict[str, int | float | NDArray[np.float64]]) -> list[float]:
    N = params['N']
    d = params['delta']
    a = params['alpha']

    roots = np.roots([N**2, 2*N*d, d**2 + 1, -a])
    roots = np.real(roots[np.isreal(roots)])

    return roots


def fixed_points_1st_order(params : dict[str, int | float | NDArray[np.float64]]) -> list[float]:
    points_0th_order = fixed_points_0th_order(params)

    N = params['N']
    d = params['delta']
    Lambda = np.sum(params['chi_ijk'][:N,:N,:N], axis=(1, 2))
    eps = 1/params['sigma']

    Lambda_bar = np.sum(params['chi_ijk'][:N,:N,:N])

    points_1st_order = []
    for x in points_0th_order:
        dL_bar = (2*x*(N*x+d)*Lambda_bar) / ((3*N*x+d)*(N*x+d)+1)

        x_star = x + eps * x**2 * ( Lambda - dL_bar )
        z_star = x - eps * x**2 * dL_bar

        points_1st_order.append(np.concatenate([x_star, [z_star]]))

    return points_1st_order



# -----------------------------
# Bifurcation boundaries
# -----------------------------

def lower_boundary(N : int,
                   d : float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    s = np.sqrt(d**2 - 3)
    return -2/27 * (s - 2*d)**2 * (s + d) / N


def upper_boundary(N : int,
                   d : float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    s = np.sqrt(d**2 - 3)
    return  2/27 * (s + 2*d)**2 * (s - d) / N


def cusp(N : int) -> tuple[float, float]:
    return -np.sqrt(3), 8*np.sqrt(3)/(9*N)



# -----------------------------
# Jacobian
# -----------------------------

def dL(x : float | NDArray[np.float64],
       params : dict[str, int | float | NDArray[np.float64]]) -> float | NDArray[np.float64]:
    N = params['N']
    x = x[:N]
    q = np.sum(x) + params['delta']
    return (-2*params['alpha']) * q / ( q**2 + 1 )**2


def Jacobian(t : float,
             x : float,
             params : dict[str, int | float | NDArray[np.float64]],
             use_3d : bool = True,
             use_4d : bool = True,
             use_optical_coupling : bool = True) -> NDArray[np.float64]:
    '''
    Construct the block Jacobian matrix

        J = [[0, I, 0],
             [A, Γ, μ],
             [L, 0, -1/tau]]

    Parameters
    ----------
    x_star : (N,) array_like
        Fixed point vector x*_i
    params : dict
        Dictionary containing keys 'N', 'mu', 'gamma', 'chi_ijk', 'chi_ijkl', 'sigma', 'tau'
    use_3d : bool, default True
        Whether to include the 3D tensor chi_ijk derivative in the A matrix
    use_4d : bool, default True
        Whether to include the 4D tensor chi_ijkl derivative in the A matrix
    opticalcoupling : bool, default True
        Whether to include the dL/tau optical coupling block

    Returns
    -------
    J : (2N+1, 2N+1) ndarray
        Full Jacobian matrix
    '''
    N = params['N']
    mu = params['mu']
    tau = params['tau']

    use_4d = use_4d and use_3d

    if use_4d:
        chi_ijkl = params['chi_ijkl'][:N, :N, :N, :N]

    if use_3d:
        sigma = params['sigma']
        chi_ijk = params['chi_ijk'][:N, :N, :N]


    # --- Block matrices ---

    # 0 block
    O = np.zeros((N, N))

    # Identity block
    I = np.eye(N)

    # Gamma block
    Gamma = -np.diag(params['gamma'])

    # mu column block
    mu_col = mu.reshape(N, 1)

    # L row block
    if use_optical_coupling:
        dL_val = dL(x, params)
        L_row = np.full((1, N), dL_val / tau)
    else:
        L_row = np.zeros((1, N))
        

    # A block initialization
    A = np.zeros((N, N))

    # Evaluate the structural coordinate stiffness contributions
    for i in range(N):
        for j in range(N):
            # 1. Base linear frequency term (diagonal)
            A[i, j] = -mu[i] * (i == j)
            
            # 2. 3D interaction tensor contribution
            if use_3d:
                # sum_k chi_ijk * x_k
                interaction_sum_3d = np.sum(chi_ijk[i, j, :] * x[:N])
                A[i, j] += (2.0 * mu[i] / sigma) * interaction_sum_3d
                
            # 3. 4D interaction tensor contribution
            if use_4d:
                # sum_{k,l} chi_ijkl * x_k * x_l
                # Outer product matrix x_k * x_l maps directly onto the last two dimensions
                x_outer = np.outer(x[:N], x[:N])
                interaction_sum_4d = np.sum(chi_ijkl[i, j, :, :] * x_outer)
                A[i, j] -= (3.0 * mu[i] / (sigma**2)) * interaction_sum_4d

    # --- Assemble full Jacobian ---

    top = np.hstack([O, I, np.zeros((N, 1))])

    middle = np.hstack([A, Gamma, mu_col])

    bottom = np.hstack([L_row, np.zeros((1, N)), np.array([[-1.0 / tau]])])

    J = np.vstack([top, middle, bottom])

    return J


@njit(fastmath=True)
def _jacobian_numba_core(t: float, 
                         x : NDArray[np.float64],
                         N : int,
                         gamma : NDArray[np.float64],
                         mu : NDArray[np.float64],
                         tau : float,
                         sigma : float,
                         chi_ijk : NDArray[np.float64],
                         chi_ijkl : NDArray[np.float64],
                         xi : NDArray[np.float64],
                         dL_val : float,
                         use_3d : bool = True,
                         use_4d : bool = True,
                         use_optical_coupling : bool = True) -> NDArray[np.float64]:
    '''
    Core numerical backend for Jacobian assembly compiled to raw machine code.
    '''
    # Preallocate the complete full Jacobian matrix
    total_dim = 2 * N + 1
    J = np.zeros((total_dim, total_dim))
    
    # Define clean, zero-allocation views onto the block segments of J
    # Top Row Block: [ O_block , I_block , zero_col ]
    I_block = J[0:N, N:2*N]
    
    # Middle Row Block: [ A_block , Gamma_block , mu_col ]
    A_block = J[N:2*N, 0:N]
    Gamma_block = J[N:2*N, N:2*N]
    mu_col = J[N:2*N, 2*N:2*N+1]
    
    # Bottom Row Block: [ L_row , zero_row , constant_cell ]
    L_row = J[2*N:2*N+1, 0:N]
    
    # --- Populating the trivial block matrix sections ---
    # 1. Identity block
    for i in range(N):
        I_block[i, i] = 1.0
        
    # 2. Gamma block (damping)
    for i in range(N):
        Gamma_block[i, i] = -gamma[i]
        
    # 3. Mu column block
    for i in range(N):
        mu_col[i, 0] = mu[i]*xi[i]
        
    # 4. L row block & final cell
    if use_optical_coupling:
        for j in range(N):
            L_row[0, j] = dL_val / tau
            
    J[2*N, 2*N] = -1.0 / tau
    
    # --- Populating the coordinate stiffness block (A) ---
    sigma_sq = sigma * sigma
    
    for i in range(N):
        for j in range(N):
            # Base linear frequency term (diagonal element matching)
            if i == j:
                A_val = -mu[i]
            else:
                A_val = 0.0
            
            # 3D interaction tensor contribution
            if use_3d:
                interaction_sum_3d = 0.0
                for k in range(N):
                    interaction_sum_3d += chi_ijk[i, j, k] * x[k]
                A_val += (2.0 * mu[i] / sigma) * interaction_sum_3d
                
            # 4D interaction tensor contribution
            if use_4d:
                interaction_sum_4d = 0.0
                for k in range(N):
                    for l in range(N):
                        interaction_sum_4d += chi_ijkl[i, j, k, l] * x[k] * x[l]
                A_val -= (3.0 * mu[i] / sigma_sq) * interaction_sum_4d
                
            A_block[i, j] = A_val
            
    return J


def Jacobian_numba(t : float, 
                   x : NDArray[np.float64],
                   params : dict[str, int | float | NDArray[np.float64]],
                   use_3d : bool = True,
                   use_4d : bool = True,
                   use_optical_coupling : bool = True) -> NDArray[np.float64]:
    '''
    User-facing Jacobian function that safely handles the params dictionary
    and invokes the compiled Numba numerical engine.
    '''
    N = params['N']
    
    # Compute the optical derivative scalar if flag is toggled
    if use_optical_coupling:
        # Assuming dL(x, params) computes your analytical partial float value
        dL_val = dL(x, params)
    else:
        dL_val = 0.0
        
    return _jacobian_numba_core(
        t=t,
        x=x,
        N=N,
        gamma=params['gamma'],
        mu=params['mu'],
        tau=params['tau'],
        sigma=params['sigma'],
        chi_ijk=params['chi_ijk'][:N, :N, :N],
        chi_ijkl=params['chi_ijkl'][:N, :N, :N, :N],
        xi=params['xi'],
        dL_val=dL_val,
        use_3d=use_3d,
        use_4d=use_4d,
        use_optical_coupling=use_optical_coupling
    )


def compute_eigs(params : dict[str, int | float | NDArray[np.float64]],
                 numba : bool = False,
                 use_3d : bool = True,
                 use_4d : bool = True,
                 use_optical_coupling : bool = True,
                 verbose : bool = False) -> tuple[NDArray[np.float64],
                                                  NDArray[np.float64],
                                                  NDArray[np.float64]]:
    
    roots = fixed_points_num(params, use_3d=use_3d, use_4d=use_4d, numba=numba)

    eigvals = []
    eigvecs = []

    if verbose: print('EIGENVALUES AND EIGENVECTORS:')

    for i, root in enumerate(roots):

        if numba:
            J_func = Jacobian_numba
        else:
            J_func = Jacobian
            
        vals, vecs = np.linalg.eig(J_func(0.0, root, params, use_3d=use_3d, use_4d=use_4d,
                                          use_optical_coupling=use_optical_coupling))
        eigvals.append(vals)
        eigvecs.append(vecs)

        if verbose:
            print(f'\troot {i}')
            for j, (val, vec) in enumerate(zip(vals, vecs)):
                print(f'\t\tvalue {j}:{val}')
                print(f'\t\tvector {j}:{vec}')

    return np.array(roots), np.array(eigvals), np.array(eigvecs)



# -----------------------------
# Lasing threshold
# -----------------------------

def find_pure_imag_crossings(params : dict[str, int | float | NDArray[np.float64]],
                             dL_min : float,
                             dL_max : float,
                             num_scan_points : int = 250) -> NDArray[np.float64]:

    N = params['N']
    tau = params['tau']

    J0 = Jacobian(0, np.zeros(2*N+1), params, use_3d=False, use_4d=False, use_optical_coupling=False)

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


def lasing_threshold(params : dict[str, int | float | NDArray[np.float64]],
                     deltas : NDArray[np.float64] = None,
                     num_scan_points : int = 250,
                     as_func_off : Literal['delta', 'delta_eff'] = 'delta',
                     delta_effs : NDArray[np.float64] = None,
                     return_all : bool = True,
                     verbose : bool = False) -> list[NDArray[np.float64]]:
    
    N = params['N']

    if isinstance(deltas, (np.ndarray, list)):
        d_max = np.max(np.abs([deltas[0], deltas[-1]]))
        d = deltas
    else:
        d = params['delta']
        d_max = d

    dL_min = (1-np.sqrt(1+d_max**2))/2
    dL_max = 0

    dL_sols = N*find_pure_imag_crossings(params, dL_min, dL_max, num_scan_points=num_scan_points)
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
    
    thresholds_filtered = _filter_arrays(thresholds)
    if verbose: print(f'\tthresholds shape (after filtering):{np.shape(thresholds_filtered)}')
    if len(thresholds_filtered) == 0:
        return []

    thresholds_sorted = sorted(thresholds_filtered, key=lambda a: np.min(a))

    if return_all:
        return thresholds_sorted
    elif not return_all:
        return thresholds_sorted[0]


def _filter_arrays(arr_list : list[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
    '''
    Remove arrays that:
    - are entirely negative
    - consist only of NaN values
    '''
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


def _print_dict(dic):
    for key, val in dic.items():
        print(f"{key:<15} : {val}")


def to_SI(params,
          rho = 145, # kg/m3
          rho_s = 145,  # kg/m^3
          a_vdw = 2.6e-24, # m^5/s^2
          kappa = 11.45e6, # Hz, FWHM
          G = 2e16, # Hz/m
          R = 3e-3, # m
          beta = 1.5e6,
          f = 2.818e14, # Hz 
          kappa_ex = None, # Hz, FWHM
          verbose = True):
    
    if kappa_ex == None:
        kappa_ex = kappa/2 # assumes critical coupling
    
    kappa_rad_HWHM = np.pi*kappa # rad/s, HWHM
    kappa_ex_rad_HWHM = np.pi*kappa_ex
    G_rad = 2*np.pi*G
    omega = 2*np.pi*f

    params_SI = {}
    if 'N' in params: params_SI['N'] = params['N']
    params_SI['d'] =  params['sigma'] * kappa_rad_HWHM / G_rad
    c3 = np.sqrt(3*rho_s*a_vdw / (rho * params_SI['d']**3))
    Omega1 = jn_zeros(1, 1)[0] * c3 / R
    m0 = np.pi*R**4*rho**2 / (jn_zeros(1, 1)[0]**2 * params_SI['d'] * rho_s)

    if 'tau' in params: params_SI['tau'] = params['tau'] / Omega1 # s
    if 'mu' in params: params_SI['freqs'] = np.sqrt(params['mu'])/(2*np.pi) * Omega1 # Hz
    if 'mu' in params: params_SI['masses'] = m0 / params['mu'] # kg
    if 'gamma' in params: params_SI['Gammas'] = params['gamma']/(2*np.pi) * Omega1 # Hz
    if 'delta' in params: params_SI['detuning'] = kappa*params['delta']/2 # Hz
    if 'alpha' in params: params_SI['power'] = params['alpha'] / (2*beta * kappa_ex_rad_HWHM * G_rad**2 * params_SI['d']**4 / (3*np.pi*a_vdw*rho*omega*kappa_rad_HWHM**3*R**2)) # W

    if verbose:
        _print_dict(params_SI)
    
    return params_SI


def to_unitless(params_SI,
                rho = 145, # kg/m3
                rho_s = 145,  # kg/m^3
                a_vdw = 2.6e-24, # m^5/s^2
                kappa = 11.45e6, # Hz, FWHM
                G = 2e16, # Hz/m
                R = 3e-3, # m
                beta = 1.5e6,
                f = 2.818e14, # Hz 
                kappa_ex = None, # Hz, FWHM
                verbose = True):
    
    if kappa_ex == None:
        kappa_ex = kappa/2 # assumes critical coupling
    
    kappa_rad_HWHM = np.pi*kappa # rad/s, HWHM
    kappa_ex_rad_HWHM = np.pi*kappa_ex
    G_rad = 2*np.pi*G
    omega = 2*np.pi*f

    params = {}
    params['N'] = params_SI['N']
    params['mu'] = mu_spectrum(params_SI['N'])
    params['sigma'] = params_SI['d'] * G_rad / kappa_rad_HWHM
    c3 = np.sqrt(3*rho_s*a_vdw / (rho * params_SI['d']**3))
    Omega1 = jn_zeros(1, 1)[0] * c3 / R
    params['Omega1'] = Omega1

    if 'tau' in params_SI: params['tau'] = params_SI['tau'] * Omega1
    if 'Gammas' in params_SI: params['gamma'] = params_SI['Gammas']*(2*np.pi) / Omega1
    if 'detuning' in params_SI: params['delta'] = 2/kappa*params_SI['detuning']
    if 'power' in params_SI: params['alpha'] = params_SI['power'] * (2*beta * kappa_ex_rad_HWHM * G_rad**2 * params_SI['d']**4 / (3*np.pi*a_vdw*rho*omega*kappa_rad_HWHM**3*R**2))

    params['nu'] = Omega1/kappa_rad_HWHM

    if verbose:
        _print_dict(params)

    return params
        





    
