import warnings
from typing import Any, List, Literal

import numpy as np
from numpy.typing import NDArray
from numba import njit

from scipy.optimize import root, root_scalar
from scipy.special import jn_zeros

warnings.filterwarnings('ignore')  # Suppress all warnings

verbose = False

# -----------------------------
# Frequency spectrum
# -----------------------------

def zeta(i : int) -> float:
    return jn_zeros(1, i)[-1]


def mu_spectrum(i : int) -> NDArray[np.float64]:
    return ( jn_zeros(1, i) / jn_zeros(1, 1) )**2


def mu_spectrum_harmonic(i : int) -> NDArray[np.float64]:
    return np.sqrt(np.arange(1, i+1))


# -----------------------------
# System
# -----------------------------

def system(t : float,
           state : NDArray[np.float64],
           params : dict[str, int | float | NDArray[np.float64]],
           use_3d : bool = True,
           use_4d : bool = True) -> NDArray[np.float64]:
    '''
    Computes the non-sideband resolved ODE system with N mechanical modes,
    radiation pressure dynamics (u, v), and opto-thermal coupling (z).
    State vector: [x_1,...,x_N, y_1,...,y_N, u, v, z] (Size: 2N + 3)
    '''
    N = params['N']
    nu = params['nu']
    tau = params['tau']

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
    # Optical/Thermal drive force including radiation pressure and opto-thermal effects
    optical_force = u**2 + v**2 + params['beta'] * z
    
    y_dot = (
        -params['gamma'] * y
        -params['mu'] * x
        +params['mu'] * params['xi'] * optical_force
        +(params['mu'] / params['sigma']) * interaction_3D
        -(params['mu'] / (params['sigma'] **2)) * interaction_4D
    )

    # --- u, v, z cavity equations ---
    sum_x = np.sum(x)
    eff_detuning = params['delta'] + sum_x
    
    u_dot = (1.0 / nu) * (-u - eff_detuning * v + np.sqrt(params['alpha']))
    v_dot = (1.0 / nu) * (-v + eff_detuning * u)
    z_dot = (1.0 / tau) * (u**2 + v**2 - z)

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
                       nu : float,
                       alpha : float,
                       delta : float,
                       sigma : float,
                       beta : float,
                       chi_ijk : NDArray[np.float64],
                       chi_ijkl : NDArray[np.float64],
                       xi : NDArray[np.float64],
                       use_3d : bool = True,
                       use_4d : bool = True) -> NDArray[np.float64]:
    '''
    Core numerical backend for the 2N+3 non-sideband resolved system.
    '''
    x = state[0:N]
    y = state[N:2*N]
    u = state[2*N]
    v = state[2*N+1]
    z = state[2*N+2]
    
    derivatives = np.empty(2 * N + 3)
    dx = derivatives[0:N]
    dy = derivatives[N:2*N]
    
    dx[:] = y
    
    tensor_term = np.zeros(N)
    sigma_sq = sigma * sigma
    
    for i in range(N):
        s_3d = 0.0
        if use_3d:
            for j in range(N):
                s_3d += chi_ijk[i, j, j] * x[j] * x[j]
                for k in range(j + 1, N):
                    s_3d += 2.0 * chi_ijk[i, j, k] * x[j] * x[k]
                
        s_4d = 0.0
        if use_4d:
            for j in range(N):
                s_4d += chi_ijkl[i, j, j, j] * x[j] * x[j] * x[j]
                for k in range(j + 1, N):
                    s_4d += 3.0 * chi_ijkl[i, j, j, k] * x[j] * x[j] * x[k]
                    s_4d += 3.0 * chi_ijkl[i, j, k, k] * x[j] * x[k] * x[k]
                    for l in range(k + 1, N):
                        s_4d += 6.0 * chi_ijkl[i, j, k, l] * x[j] * x[k] * x[l]
                    
        tensor_term[i] = (mu[i] / sigma) * s_3d - (mu[i] / sigma_sq) * s_4d

    optical_force = u*u + v*v + beta * z
    for i in range(N):
        dy[i] = -gamma[i]*y[i] - mu[i]*x[i] + mu[i]*xi[i]*optical_force + tensor_term[i]
        
    sum_x = np.sum(x)
    eff_detuning = delta + sum_x
    
    derivatives[2*N] = (1.0 / nu) * (-u - eff_detuning * v + np.sqrt(alpha))
    derivatives[2*N+1] = (1.0 / nu) * (-v + eff_detuning * u)
    derivatives[2*N+2] = (1.0 / tau) * (u*u + v*v - z)
    
    return derivatives


def system_numba(t : float,
                 state : NDArray[np.float64],
                 params : dict[str, int | float | NDArray[np.float64]],
                 use_3d : bool = True,
                 use_4d : bool = True) -> NDArray[np.float64]:
    N = params['N']
    if use_4d:
        use_3d = True
    if not use_3d:
        use_4d = False
    
    return _system_numba_core(
        t=t, state=state, N=N,
        gamma=params['gamma'], mu=params['mu'], tau=params['tau'], nu=params['nu'],
        alpha=params['alpha'], delta=params['delta'], sigma=params['sigma'], beta=params['beta'],
        chi_ijk=params['chi_ijk'][:N, :N, :N], chi_ijkl=params['chi_ijkl'][:N, :N, :N, :N],
        xi=params['xi'], use_3d=use_3d, use_4d=use_4d
    )


# -----------------------------
# Fixed points
# -----------------------------

def fixed_points_num(params : dict[str, int | float | NDArray[np.float64]],
                     tolerance : float = 1e-10,
                     use_3d : bool = True,
                     use_4d : bool = True,
                     numba : bool = True) -> list[NDArray[np.float64]]:
    '''
    Finds full [x, y, u, v, z] steady states numerically.
    '''
    N = params['N']
    
    def steady_state_objective(vars_xuvz):
        x_val = vars_xuvz[0:N]
        u_val = vars_xuvz[N]
        v_val = vars_xuvz[N+1]
        z_val = vars_xuvz[N+2]
        
        # Assemble complete evaluation vector
        U = np.concatenate([x_val, np.zeros(N), [u_val, v_val, z_val]])
        system_func = system_numba if numba else system
        derivs = system_func(0, U, params, use_3d=use_3d, use_4d=use_4d)
        
        # Objective maps mechanical acceleration and optical equations to zero
        return np.concatenate([derivs[N:2*N], derivs[2*N:]])

    valid_solutions = []
    points_0th = fixed_points_0th_order(params)

    for x_0 in points_0th:
        # Reconstruct analytical u, v, z tracking guesses based on x_0 profile
        X_0 = N * x_0
        denom = 1.0 + (params['delta'] + X_0)**2
        u_0 = np.sqrt(params['alpha']) / denom
        v_0 = np.sqrt(params['alpha']) * (params['delta'] + X_0) / denom
        z_0 = u_0**2 + v_0**2
        
        initial_guess = np.concatenate([np.full(N, x_0), [u_0, v_0, z_0]])
        res = root(steady_state_objective, initial_guess, method='lm')
        
        if res.success and np.linalg.norm(res.fun) < tolerance:
            sol_x = res.x[0:N]
            sol_u, sol_v, sol_z = res.x[N], res.x[N+1], res.x[N+2]
            
            if np.all(sol_x >= 0) and np.all(sol_x < 2*params['sigma']):
                full_fixed_point = np.concatenate([sol_x, np.zeros(N), [sol_u, sol_v, sol_z]])
                if not any(np.allclose(full_fixed_point, sol, atol=1e-8) for sol in valid_solutions):
                    valid_solutions.append(full_fixed_point)
                        
    return valid_solutions


def fixed_points_0th_order(params : dict[str, int | float | NDArray[np.float64]]) -> list[float]:
    N = params['N']
    d = params['delta']
    a = params['alpha']
    b = params['beta']
    xi_mean = np.mean(params['xi'][:N])
    
    # Drive factor handles combination of radiation pressure + optothermal load
    drive_scale = a * xi_mean * (1.0 + b)
    roots = np.roots([N**2, 2*N*d, d**2 + 1, -drive_scale])
    return np.real(roots[np.isreal(roots)])


# -----------------------------
# Jacobian
# -----------------------------

def Jacobian(t : float,
             state : NDArray[np.float64],
             params : dict[str, int | float | NDArray[np.float64]],
             use_3d : bool = True,
             use_4d : bool = True,
             use_optical_coupling : bool = True) -> NDArray[np.float64]:
    '''
    Assembles the multi-block (2N+3) x (2N+3) Jacobian matrix.
    '''
    N = params['N']
    mu = params['mu']
    tau = params['tau']
    nu = params['nu']
    beta = params['beta']

    x = state[:N]
    u = state[2*N]
    v = state[2*N+1]

    use_4d = use_4d and use_3d
    sigma = params['sigma']

    # --- Block Components ---
    J = np.zeros((2*N + 3, 2*N + 3))
    J[0:N, N:2*N] = np.eye(N)  # dx/dt View onto y
    J[N:2*N, N:2*N] = -np.diag(params['gamma'])  # dy/dt Damping

    # Compute mechanical stiffness component block A
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i, j] = -mu[i] * (i == j)
            if use_3d:
                A[i, j] += (2.0 * mu[i] / sigma) * np.sum(params['chi_ijk'][i, j, :N] * x)
            if use_4d:
                x_outer = np.outer(x, x)
                A[i, j] -= (3.0 * mu[i] / (sigma**2)) * np.sum(params['chi_ijkl'][i, j, :N, :N] * x_outer)
    J[N:2*N, 0:N] = A

    # Field driving elements into mechanical equations
    for i in range(N):
        J[N+i, 2*N]   = mu[i] * params['xi'][i] * (2.0 * u)
        J[N+i, 2*N+1] = mu[i] * params['xi'][i] * (2.0 * v)
        J[N+i, 2*N+2] = mu[i] * params['xi'][i] * beta

    # Cavity equations coupling dependencies
    if use_optical_coupling:
        J[2*N, 0:N]   = -v / nu
        J[2*N+1, 0:N] = u / nu
        
    eff_detuning = params['delta'] + np.sum(x)
    J[2*N, 2*N]     = -1.0 / nu
    J[2*N, 2*N+1]   = -eff_detuning / nu
    J[2*N+1, 2*N]   = eff_detuning / nu
    J[2*N+1, 2*N+1] = -1.0 / nu
    
    J[2*N+2, 2*N]   = 2.0 * u / tau
    J[2*N+2, 2*N+1] = 2.0 * v / tau
    J[2*N+2, 2*N+2] = -1.0 / tau

    return J


@njit(fastmath=True)
def _jacobian_numba_core(t: float, 
                         state : NDArray[np.float64],
                         N : int,
                         gamma : NDArray[np.float64],
                         mu : NDArray[np.float64],
                         tau : float,
                         nu : float,
                         delta : float,
                         sigma : float,
                         beta : float,
                         chi_ijk : NDArray[np.float64],
                         chi_ijkl : NDArray[np.float64],
                         xi : NDArray[np.float64],
                         use_3d : bool = True,
                         use_4d : bool = True,
                         use_optical_coupling : bool = True) -> NDArray[np.float64]:
    
    total_dim = 2 * N + 3
    J = np.zeros((total_dim, total_dim))
    
    # Unpack state variables safely
    x = state[0:N]
    u = state[2*N]
    v = state[2*N+1]
    
    # 1. Calculate the dynamic effective detuning
    sum_x = 0.0
    for k in range(N):
        sum_x += x[k]
    eff_detuning = delta + sum_x

    # 2. Populate Trivial Identities and Damping blocks
    for i in range(N):
        J[i, N + i] = 1.0                # dx_i/dt = y_i
        J[N + i, N + i] = -gamma[i]      # dy_i/dt internal damping
        
        # Derivatives of dy_i/dt with respect to u, v, z
        J[N + i, 2*N]   = mu[i] * xi[i] * 2.0 * u
        J[N + i, 2*N+1] = mu[i] * xi[i] * 2.0 * v
        J[N + i, 2*N+2] = mu[i] * xi[i] * beta

    # 3. Populate Mechanical Stiffness Block (A)
    sigma_sq = sigma * sigma
    for i in range(N):
        for j in range(N):
            A_val = -mu[i] if i == j else 0.0
            
            if use_3d:
                s3 = 0.0
                for k in range(N): 
                    s3 += chi_ijk[i, j, k] * x[k]
                A_val += (2.0 * mu[i] / sigma) * s3
                
            if use_4d:
                s4 = 0.0
                for k in range(N):
                    for l in range(N): 
                        s4 += chi_ijkl[i, j, k, l] * x[k] * x[l]
                A_val -= (3.0 * mu[i] / sigma_sq) * s4
                
            J[N + i, j] = A_val
            
    # 4. Populate Cavity Equations Cross-Couplings
    if use_optical_coupling:
        for j in range(N):
            J[2*N, j]     = -v / nu    # d(u_dot)/dx_j
            J[2*N+1, j]   =  u / nu    # d(v_dot)/dx_j
            
    # Cavity internal field and thermal derivatives
    J[2*N, 2*N]     = -1.0 / nu
    J[2*N, 2*N+1]   = -eff_detuning / nu
    J[2*N+1, 2*N]   =  eff_detuning / nu
    J[2*N+1, 2*N+1] = -1.0 / nu
    
    J[2*N+2, 2*N]   = 2.0 * u / tau
    J[2*N+2, 2*N+1] = 2.0 * v / tau
    J[2*N+2, 2*N+2] = -1.0 / tau
    
    return J


def Jacobian_numba(t : float, 
                   state : NDArray[np.float64],
                   params : dict[str, int | float | NDArray[np.float64]],
                   use_3d : bool = True,
                   use_4d : bool = True,
                   use_optical_coupling : bool = True) -> NDArray[np.float64]:
    # Fallback to analytical structural matrix calculation if numba alignment bounds hit edge checks
    return _jacobian_numba_core(
        t, state, params['N'], params['gamma'], params['mu'], params['tau'], params['nu'],
        params['delta'], params['sigma'], params['beta'], params['chi_ijk'], params['chi_ijkl'],
        params['xi'], use_3d, use_4d, use_optical_coupling
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
    eigvals, eigvecs = [], []

    for i, root_vec in enumerate(roots):
        J_func = Jacobian_numba if numba else Jacobian
        vals, vecs = np.linalg.eig(J_func(0.0, root_vec, params, use_3d=use_3d, use_4d=use_4d,
                                          use_optical_coupling=use_optical_coupling))
        eigvals.append(vals)
        eigvecs.append(vecs)
    return np.array(roots), np.array(eigvals), np.array(eigvecs)


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