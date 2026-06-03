import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
import multiprocessing as mp
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from equations import *

# =====================================================================
# 1. OPTIMIZED PHYSICS ENGINE (Symmetry + Machine Code Compilation)
# =====================================================================
@njit(fastmath=True)
def ode_vector_field(t, state, N, gamma, mu, tau, alpha, delta, sigma, chi):
    '''
    Optimized ODE vector field exploiting the full 3-way symmetry of chi_ijk.
    State layout: [x_1...x_N, y_1...y_N, z]
    '''
    x = state[0:N]
    y = state[N:2*N]
    z = state[2*N]
    
    derivatives = np.empty(2 * N + 1)
    dx = derivatives[0:N]
    dy = derivatives[N:2*N]
    
    # dx_i/dt = y_i
    dx[:] = y
    
    # Exploit 3-way tensor symmetry: collapses loop complexity
    tensor_term = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            # Case j == k (appears 1 time)
            s += chi[i, j, j] * x[j] * x[j]
            
            # Case j < k (appears 2 times: jk and kj match)
            for k in range(j + 1, N):
                s += 2.0 * chi[i, j, k] * x[j] * x[k]
                
        tensor_term[i] = (mu[i] / sigma) * s

    # dy_i/dt
    for i in range(N):
        dy[i] = -gamma[i]*y[i] - mu[i]*x[i] + mu[i]*z + tensor_term[i]
        
    # dz/dt
    sum_x = np.sum(x)
    derivatives[2*N] = (1.0 / tau) * ((alpha / ((delta + sum_x)**2 + 1.0)) - z)
    
    return derivatives


# =====================================================================
# 2. OBJECT-ORIENTED INTERFACE AND EXECUTORS
# =====================================================================
class OscillatorSystem:
    def __init__(self, N, gamma, mu, tau, alpha, delta, sigma, chi):
        self.N = N
        self.gamma = np.atleast_1d(gamma)
        self.mu = np.atleast_1d(mu)
        self.tau = tau
        self.alpha = alpha
        self.delta = delta
        self.sigma = sigma
        self.chi = chi 


def _parallel_worker(task_args):
    '''Isolated worker function designed for multi-core serialization with

    strict shape protection on numerical blow-ups.
    '''
    sys_params, t_span, u0, t_eval, coords, alpha_threshold = task_args
    
    if sys_params['alpha'] < alpha_threshold:
        return {
            'coords': coords,
            'above_threshold': False,
            'exploded': False,
            'final_single_state': u0, # Pass along incoming hot-start unaltered
            'raw_tail': np.full((2 * sys.N + 1, len(t_eval)), np.nan), # Clean NaN array block
            'roots': None,
            'eigvals': None,
            'eigvecs': None
        }
    
    sys = OscillatorSystem(**sys_params)
    sol = solve_ivp(
        fun=ode_vector_field,
        t_span=t_span,
        y0=u0,
        args=(sys.N, sys.gamma, sys.mu, sys.tau, sys.alpha, sys.delta, sys.sigma, sys.chi),
        method='BDF',
        t_eval=t_eval,
        jac=jac_for_solver,
        rtol=1e-6, atol=1e-9
    )

    roots, vals, vecs = compute_eigs(sys_params)

    exploded = not sol.success

    # --- ENFORCE RIGID SHAPE OUTPUTS ---
    if not exploded:
        raw_tail = sol.y
        final_single_state = sol.y[:, -1]
    else:
        # If it failed or blew up, fill a pristine matrix matching the exact expected shape with NaNs
        raw_tail = np.full((2 * sys.N + 1, len(t_eval)), np.nan)
        # Fall back to default initial conditions if this vector is pulled as a hot-start next layer
        final_single_state = np.concatenate([
            np.zeros(sys.N), 
            np.zeros(sys.N),                     
            [0.1]                                                        
        ])
        
    return {
        'coords': coords,
        'above_threshold': True,
        'exploded': exploded,
        'final_single_state': final_single_state,
        'raw_tail': raw_tail,
        'roots': roots,
        'eigvals': vals,
        'eigvecs': vecs
    }


class LayeredParallelSmartSweeper:
    '''Slices a 3D parameter grid into 2D layers, evaluating each layer in parallel

    while passing the final state from the geometrically closest completed grid element as a hot start.
    '''
    def __init__(self, base_config, t_span, default_u0, t_span_eval, t_res_eval=0.1):
        self.base_config = base_config
        self.t_span = t_span
        self.default_u0 = default_u0
        self.t_eval = np.linspace(t_span_eval[0], t_span_eval[1], int(abs(t_span_eval[1] - t_span_eval[0])/t_res_eval))
        
        self.history_coords = []
        self.history_states = []

    def _get_closest_initial_state(self, target_coords):
        '''Finds the closest available historical state vector using Euclidean distance.'''
        if not self.history_coords:
            return self.default_u0
        
        distances = np.linalg.norm(np.array(self.history_coords) - np.array(target_coords), axis=1)
        closest_index = np.argmin(distances)
        return self.history_states[closest_index]

    def run_layered_parallel_sweep(self, p1_name, p1_vals, p2_name, p2_vals, p3_name, p3_vals, lthreshold=None):
        results = []
        num_cores = mp.cpu_count()
        
        # Outer Progress Bar (Iterates sequentially through slices)
        outer_pbar = tqdm(p1_vals, desc=f'Overall {p1_name} layers', position=0, leave=True)
        
        for v1 in outer_pbar:
            outer_pbar.set_postfix_str(f'Current {p1_name}={v1:.2f}')
            
            tasks = []
            # Build the parallel tasks batch for the current 2D plane slice
            for i, v2 in enumerate(p2_vals):
                
                if lthreshold is not None:
                    alpha_threshold = lthreshold[i]
                else:
                    alpha_threshold = 0

                for v3 in p3_vals:
                    current_coords = (v1, v2, v3)
                    
                    # Pull best starting guess vector out of history
                    u0_hot = self._get_closest_initial_state(current_coords)
                    
                    run_config = self.base_config.copy()
                    run_config[p1_name] = v1
                    run_config[p2_name] = v2
                    run_config[p3_name] = v3
                    
                    tasks.append((run_config, self.t_span, u0_hot, self.t_eval, current_coords, alpha_threshold))
            
            # Execute the 2D slice concurrently across your CPU cores
            layer_results = process_map(
                _parallel_worker, 
                tasks, 
                max_workers=num_cores,
                desc=' └─ Parallel Slice Processing',
                position=1, 
                leave=False
            )
            
            # Harvest and save clean records to history to prepare for the next layer step
            for res in layer_results:
                if res['above_threshold'] and not res['exploded']:
                    self.history_coords.append(res['coords'])
                    self.history_states.append(res['final_single_state'])
                
                results.append(res)
                
        return results


# =====================================================================
# 3. PIPELINE EXECUTION ENGINE
# =====================================================================
if __name__ == '__main__':
    N = 15

    # Correct 3D array index parsing logic
    chi_data = np.load('./chi_ijk.npy')[:N, :N, :N]

    base_config = {
        'N': N,
        'gamma': np.ones(N) * 0.05,
        'mu': mu_spectrum(N),
        'tau': 1.0,
        'alpha': 1.0,
        'delta': 0.0,
        'sigma': 20.0,
        'chi': chi_data
    }
    default_u0 = np.concatenate([
        np.random.uniform(-0.5, 0.5, N), 
        np.zeros(N),                     
        [0.1]                                                        
    ])
    
    t_span = (0.0, 1000.0)
    Dt_eval = 2*np.pi*5 / np.sqrt(base_config['mu'][0]) # span ensures 5 oscillations of the heaviest (slowest) oscillator
    t_span_eval = (t_span[1]-Dt_eval, t_span[1]) # evaluate on the last part of the simulation
    t_res_eval = 2*np.pi / (np.sqrt(base_config['mu'][-1]) * 20) # resolution ensures 10 points per oscillation for the lightest (fastest) oscillator

    
    # Instantiate Sweeper
    sweeper = LayeredParallelSmartSweeper(base_config, t_span, default_u0, t_span_eval, t_res_eval=t_res_eval)
    
    # Define parameters vectors grid resolution
    deltas = np.linspace(-1, 1, 5)
    lthreshold = np.array(lasing_threshold(base_config, deltas=deltas))
    alphas = np.linspace(0.9*np.min(lthreshold), 2, 5)
    sigmas = np.array([30, 40, 50])
    
    
    # Execute the Pipeline
    sweep_data = sweeper.run_layered_parallel_sweep(
        p1_name='alpha', p1_vals=alphas,
        p2_name='delta', p2_vals=deltas,
        p3_name='sigma', p3_vals=sigmas # if lthreshold is given, p1 should be alpha and p2 should be delta
    )
    
    # Map the compiled records to a highly accessible 5-dimensional NumPy Array Data Block
    print('\nWriting data...')
    dim = 2 * N + 1
    state_tensor = np.zeros((len(alphas), len(deltas), len(sigmas), dim, int(abs(t_span_eval[1] - t_span_eval[0])/t_res_eval)))
    roots_tensor = np.full((len(alphas), len(deltas), len(sigmas), 3, dim), np.nan)
    eigvals_tensor = np.full((len(alphas), len(deltas), len(sigmas), 3, dim), np.nan, dtype=complex)
    eigvecs_tensor = np.full((len(alphas), len(deltas), len(sigmas), 3, dim, dim), np.nan, dtype=complex)
    above_threshold_tensor = np.full((len(alphas), len(deltas), len(sigmas)), True, dtype=bool)
    exploded_tensor = np.full((len(alphas), len(deltas), len(sigmas)), False, dtype=bool)

    
    idx = 0
    for i in tqdm(range(len(alphas))):
        for j in range(len(deltas)):
            for k in range(len(sigmas)):
                res = sweep_data[idx]
                state_tensor[i, j, k, :, :] = res['raw_tail']

                above_threshold_tensor[i,j,k] = res['above_threshold']
                exploded_tensor[i,j,k] = res['exploded']

                if res['above_threshold'] and not res['exploded'] and len(res['roots']) > 0:
                    num_found = len(res['roots'])
                    
                    # Assign the found data up to the actual count; remaining slots stay NaN
                    roots_tensor[i, j, k, :num_found, :] = res['roots']
                    eigvals_tensor[i, j, k, :num_found, :] = res['eigvals']
                    eigvecs_tensor[i, j, k, :num_found, :, :] = res['eigvecs']

                idx += 1

    # Save to disk
    filename = f'sweep_results_N={N}.npz'

    time = np.linspace(t_span_eval[0], t_span_eval[1], int(Dt_eval/t_res_eval))

    np.savez_compressed(filename, time=time, 
                        alphas=alphas, deltas=deltas, sigmas=sigmas, states=state_tensor,
                        roots=roots_tensor, eigvals=eigvals_tensor, eigvecs=eigvecs_tensor,
                        above_threshold=above_threshold_tensor, exploded=exploded_tensor)
    
    print(f'Successfully saved simulation outputs to {filename}')


    print(f'total number of simulations : {exploded_tensor.size}')
    print(f'number of failed simulations: {np.sum(exploded_tensor)}')