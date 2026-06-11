import numpy as np
import multiprocessing as mp
import os
import shutil

from scipy.integrate import solve_ivp

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import copy
from equations import * 

# =====================================================================
# 1. HARDENED PARALLEL WORKER EXECUTOR
# =====================================================================
def _parallel_worker(task_args):
    '''Isolated worker function designed for multi-core serialization with
    strict memory boundaries and explicit tracking data isolation.
    '''
    sys_params, t_span, u0, t_eval, coords, use_3d, use_4d, alpha_threshold = task_args
    
    # 1. Unpack and forcefully isolate memory blocks to prevent cross-core leaks
    sys_params_local = sys_params
    N_local = sys_params_local['N']
    
    dim = 2 * N_local + 1
    
    # 2. Solver Execution Block
    # Utilizing 'Radau' to natively leverage your analytical Jacobian matrix.

    sol = solve_ivp(
        fun=system_numba,
        t_span=t_span,
        y0=u0,  # Actively utilizing the hot-start trajectory vector
        args=(sys_params_local, use_3d, use_4d),
        method='RK45',              # STIFF SOLVER FIX: Activates analytical Jacobian
        t_eval=t_eval,
        jac=Jacobian_numba,          # Passed Jacobian function
        rtol=1e-6, atol=1e-8,
        first_step=1e-5
    )

    roots, vals, vecs = compute_eigs(sys_params)


    # 3. Enforce Rigid Shape Outputs
    if sol.success:
        raw_tail = sol.y
        final_single_state = sol.y[:, -1]
    else:
        print(sol)
        # Step-size failure tracking fallback
        if len(sol.t) == 0:
            timestep_fail = 0
            state_fail = np.ones(2*N_local+1)
        else:
            timestep_fail = sol.t[-1]
            state_fail = sol.y[:,-1]
        print(sol.message)
        print(f'failed at timestep {timestep_fail}')
        print(f'derivatives: {system(timestep_fail, state_fail, sys_params_local, use_3d, use_4d)}')

        raw_tail = np.full((dim, len(t_eval)), np.nan)
        final_single_state = np.concatenate([
            np.zeros(N_local), 
            np.zeros(N_local),                     
            [0.1]                                                                                   
        ])
        
    return {
        'coords': coords,
        'above_threshold': True,
        'exploded': not sol.success,
        'final_single_state': final_single_state,
        'raw_tail': raw_tail,
        'roots': roots,
        'eigvals': vals,
        'eigvecs': vecs
    }


# =====================================================================
# 2. LAYERED GEOMETRIC SMART SWEEPER
# =====================================================================
class LayeredParallelSmartSweeper:
    '''Slices a 3D parameter grid into 2D layers, evaluating each layer in parallel
    while passing the final state from the geometrically closest completed grid element as a hot start.
    '''
    def __init__(self, base_config, t_span, default_u0, t_span_eval, use_3d, use_4d, N_points : int = 250):
        self.base_config = base_config
        self.t_span = t_span
        self.default_u0 = default_u0
        self.use_3d = use_3d
        self.use_4d = use_4d
        self.t_eval = np.linspace(t_span_eval[0], t_span_eval[1], N_points)
        
        self.history_coords = []
        self.history_states = []

    def _get_closest_initial_state(self, target_coords):
        '''Finds the completed historical state vector whose delta value 
        is closest to the target delta.
        
        Coordinates map as: (alpha, delta, sigma) -> indices (0, 1, 2)
        '''
        if not self.history_coords:
            return self.default_u0
        
        # 1. Isolate the target delta parameter
        target_delta = target_coords[1]
        
        # 2. Extract only the historical delta values (index 1 of each tuple)
        history_deltas = np.array([coords[1] for coords in self.history_coords])
        
        # 3. Compute absolute distances along the delta axis exclusively
        distances = np.abs(history_deltas - target_delta)
        
        # 4. Find the index of the minimum distance element
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
                alpha_threshold = lthreshold[i] if lthreshold is not None else 0

                for v3 in p3_vals:
                    current_coords = (v1, v2, v3)
                    
                    # Extract historical attractor trajectory
                    u0_hot = self._get_closest_initial_state(current_coords)
                    
                    # Isolate configuration dictionaries explicitly
                    run_config = copy.deepcopy(self.base_config)
                    run_config[p1_name] = v1
                    run_config[p2_name] = v2
                    run_config[p3_name] = v3
                    
                    tasks.append((run_config, self.t_span, u0_hot, self.t_eval, current_coords, self.use_3d, self.use_4d, alpha_threshold))
            
            # Execute the 2D slice concurrently across CPU cores
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
    
    def clear_history(self):
        '''Flushes the historical states to free up RAM between separate sigma runs.'''
        self.history_coords = []
        self.history_states = []


# =====================================================================
# 3. PIPELINE EXECUTION ENGINE
# =====================================================================
# =====================================================================
# 3. PIPELINE EXECUTION ENGINE
# =====================================================================
if __name__ == '__main__':
    import gc # Import garbage collector for explicit RAM flushing

    N = 15
    use_3d = True
    use_4d = True

    base_config = {'N': N,
                   'gamma': np.ones(N) * 0.05,
                   'mu': mu_spectrum(N),
                   'tau': 1.0,
                   'alpha': 1.0,
                   'delta': 0.0,
                   'sigma': 20.0,
                   'chi_ijk': np.load('./tensors/chi_ijk.npy')[:N, :N, :N],
                   'chi_ijkl': np.load('./tensors/chi_ijkl.npy')[:N, :N, :N, :N],
                   'xi': np.ones(N)}
    
    default_u0 = np.concatenate([
        np.random.uniform(-0.5, 0.5, N),
        np.zeros(N),                     
        [0.1]
    ])
    
    # Precompute time evaluation boundaries
    Dt_sim = 1000.0
    Dt_eval = 2*np.pi*25 / np.sqrt(base_config['mu'][0])
    t_span = (0.0, Dt_sim + Dt_eval)
    t_span_eval = (Dt_sim, Dt_sim + Dt_eval)
    N_points = int(10*25*np.sqrt(base_config['mu'][-1]/base_config['mu'][0]))

    # =====================================================================
    # NUMBA COMPILATION RACE-CONDITION SAFEGUARD
    # =====================================================================
    print("Pre-compiling Numba physics engine on main thread...")
    dummy_y = np.zeros(2 * N + 1)
    _ = system_numba(0.0, dummy_y, base_config)
    _ = Jacobian_numba(0.0, dummy_y, base_config)
    print("Pre-compilation finished successfully.\n")

    # Instantiate Sweeper
    sweeper = LayeredParallelSmartSweeper(base_config, t_span, default_u0, t_span_eval, use_3d, use_4d, N_points=N_points)
    
    # Define parameters vectors grid resolution
    deltas = np.linspace(-5, 3, 100)[::-1]
    lthreshold = np.array(lasing_threshold(base_config, deltas=deltas, return_all=False))
    alphas = np.linspace(0, 6, 100)
    sigmas = np.array([50, 75]) #np.arange(40, 60, 20)
    
    dim = 2 * N + 1
    deltas_increasing = deltas[::-1]
    time_array = np.linspace(t_span_eval[0], t_span_eval[1], N_points)

    # Ensure results directory exists
    os.makedirs('./results', exist_ok=True)

    # =====================================================================
    # CHUNKED EXECUTION: Loop over Sigma to constrain RAM
    # =====================================================================
    for k, sigma in enumerate(sigmas):
        print(f"\n=======================================================")
        print(f"Sweeping parameter space for sigma = {sigma} ({k+1}/{len(sigmas)})")
        print(f"=======================================================")
        
        # Execute Pipeline for a SINGLE sigma value
        sweep_data = sweeper.run_layered_parallel_sweep(
            p1_name='alpha', p1_vals=alphas,
            p2_name='delta', p2_vals=deltas,
            p3_name='sigma', p3_vals=[sigma], # Isolate to current sigma
            lthreshold=lthreshold
        )
        
        print(f'\nAllocating and writing data for sigma={sigma}...')
        
        # 1. Preallocate tensors for THIS SIGMA ONLY (Removing the sigma dimension)
        state_tensor = np.zeros((len(alphas), len(deltas), dim, N_points))
        roots_tensor = np.full((len(alphas), len(deltas), 3, dim), np.nan)
        eigvals_tensor = np.full((len(alphas), len(deltas), 3, dim), np.nan, dtype=complex)
        eigvecs_tensor = np.full((len(alphas), len(deltas), 3, dim, dim), np.nan, dtype=complex)
        above_threshold_tensor = np.full((len(alphas), len(deltas)), True, dtype=bool)
        exploded_tensor = np.full((len(alphas), len(deltas)), False, dtype=bool)

        # 2. Map data to memory block
        idx = 0
        for i in tqdm(range(len(alphas)), desc="Mapping Arrays"):
            ix = len(alphas) - 1 - i if alphas[-1] < alphas[0] else i

            for j in range(len(deltas)):
                jx = len(deltas) - 1 - j if deltas[-1] < deltas[0] else j

                res = sweep_data[idx]
                
                state_tensor[ix, jx, :, :] = res['raw_tail']
                above_threshold_tensor[ix, jx] = res['above_threshold']
                exploded_tensor[ix, jx] = res['exploded']

                if res['above_threshold'] and not res['exploded'] and res['roots'] is not None and len(res['roots']) > 0:
                    num_found = len(res['roots'])
                    roots_tensor[ix, jx, :num_found, :] = res['roots']
                    eigvals_tensor[ix, jx, :num_found, :] = res['eigvals']
                    eigvecs_tensor[ix, jx, :num_found, :, :] = res['eigvecs']

                idx += 1

        # 3. Save chunk to disk
        filename = f'./results/sweep_results_N={N}_sigma={sigma}.npz'
        np.savez_compressed(
            filename, 
            time=time_array, 
            alphas=alphas, 
            deltas=deltas_increasing,  
            sigma=sigma, # Save individual scalar
            states=state_tensor,
            roots=roots_tensor, 
            eigvals=eigvals_tensor, 
            eigvecs=eigvecs_tensor,
            base_config=base_config,
            above_threshold=above_threshold_tensor, 
            exploded=exploded_tensor
        )
        
        print(f'Saved chunk to {filename}')
        
        # 4. Optional Error Audit for this chunk
        failed_indices = np.argwhere(exploded_tensor == True)
        if len(failed_indices) > 0:
            print(f"Audit: {len(failed_indices)} failed simulations in this chunk.")
            # Your audit logic can be kept here if you want to inspect step-zeros

        # 5. AGGRESSIVE RAM FLUSH
        # Delete local heavy references
        del sweep_data
        del state_tensor
        del roots_tensor
        del eigvals_tensor
        del eigvecs_tensor
        del above_threshold_tensor
        del exploded_tensor
        
        # Clear sweeper hot-start memory to prevent accumulating old states
        sweeper.clear_history() 
        
        # Force garbage collector to release memory to OS immediately
        gc.collect()