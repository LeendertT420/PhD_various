import numpy as np
import multiprocessing as mp
import sys
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import copy
from equations import * # =====================================================================
# 1. HARDENED PARALLEL WORKER EXECUTOR (REFACTORED FOR SEQUENTIAL DELTA SWEEPS)
# =====================================================================
def _parallel_worker(task_args):
    '''Isolated worker function running a sequential delta sweep for a fixed alpha/sigma.
    This architecture enables hot starting across the delta trajectory.
    '''
    base_config, T_sim, x0_init, T_eval, use_3d, use_4d, N_points, p1_name, v1, p2_name, v2, p3_name, p3_vals = task_args
    
    N_local = base_config['N']
    current_x0 = np.copy(x0_init) # Initialize hot-start tracking state
    worker_results = []

    # Iterate sequentially through the delta axis to maintain the hot start
    for v3 in p3_vals:
        current_coords = (v1, v2, v3)
        
        # Isolate configuration dictionaries explicitly
        run_config = copy.deepcopy(base_config)
        run_config[p1_name] = v1
        run_config[p2_name] = v2
        run_config[p3_name] = v3
        
        t_eval = np.linspace(T_sim, T_sim + T_eval, N_points)

        sol = solve_ivp(
            fun=system_numba,
            t_span=(0, T_sim + T_eval),
            y0=current_x0, # <-- HOT START: Uses the previous final state
            args=(run_config, use_3d, use_4d),
            method='RK45',    
            t_eval=t_eval,
            jac=Jacobian_numba,       
            rtol=1e-6, atol=1e-8,
            first_step=1e-5
        )

        # Initialize fallback variables to prevent NameErrors
        classification = 'NOT CLASSIFIED'
        roots, vals, vecs = None, None, None
        peaks, fft_freqs, fft_vals = np.array([]), np.array([]), np.array([])

        if sol.success:
            # Update hot start state for the NEXT iteration in this delta loop
            #current_x0 = sol.y[:, -1]
            
            # CRITICAL FIX: sum across axis=0 to preserve time-series length
            x_total = np.sum(sol.y[:N_local, :], axis=0)
            x_ac = x_total - np.mean(x_total)

            if np.max(x_ac) < 1e-3:
                classification = 'BELOW THRESHOLD'
            else:
                roots, vals, vecs = compute_eigs(run_config)
                
                # FIND PEAKS
                fft_freqs = np.fft.rfftfreq(N_points, d=T_eval / N_points)
                  
                sigma_x = np.std(x_ac)
                if sigma_x < 1e-12:
                    classification = 'NO VARIANCE'
                else:
                    x_normalized = x_ac / sigma_x
                    fft_vals = (2.0 / N_points) * np.abs(np.fft.rfft(x_normalized))
                    peaks, _ = find_peaks(fft_vals, prominence=0.1, height=0.05)
   
                    

                    # FIND LYAPUNOV EXPONENT
                    x0_2 = sol.y[:, 0] + 1e-9 * np.ones(2 * N_local + 1)

                    sol_2 = solve_ivp(
                        fun=system_numba,
                        t_span=(T_sim, T_sim + T_eval),
                        y0=x0_2, 
                        args=(run_config, use_3d, use_4d),
                        method='RK45',    
                        t_eval=t_eval,
                        jac=Jacobian_numba,       
                        rtol=1e-6, atol=1e-8,
                        first_step=1e-5
                    )
                    
                    distance = np.linalg.norm(sol_2.y - sol.y, axis=0)
                    
                    if len(np.where(distance >= 1)[0]) == 0:
                        # CLASSIFY
                        if len(peaks) == 0:
                            classification = 'NOT CLASSIFIED'
                        elif len(peaks) == 1:
                            classification = 'SINGLE MODE LASING'
                        else:
                            df = fft_freqs[1] - fft_freqs[0]
                            active_freqs = np.sort(fft_freqs[peaks])
                            
                            f0 = active_freqs[0]
                            margin = 4 * df
                            is_mode_locked = f0 > margin
                            
                            if is_mode_locked:
                                for f in active_freqs[1:]:
                                    nearest_multiple = round(f / f0) * f0
                                    if np.abs(f - nearest_multiple) > margin:
                                        is_mode_locked = False
                                        break
                                        
                            if is_mode_locked:
                                classification = 'MODE LOCKED'
                            else:
                                classification = 'MULTI MODE LASING'
                    else:
                        log_distance = np.log(distance)
                        time = sol.t

                        sat_indices = np.where(distance >= 1)[0]
                        saturation_idx = sat_indices[0] 
                        fit_mask = time <= time[saturation_idx]

                        time_fit = time[fit_mask]
                        log_dist_fit = log_distance[fit_mask]

                        if len(time_fit) > 1:
                            mle_estimate, _ = np.polyfit(time_fit, log_dist_fit, 1)
                            classification = 'CHAOTIC' if mle_estimate > 0 else 'NOT CLASSIFIED'
                        else:
                            classification = 'NOT CLASSIFIED'
        else:
            print(f'\nIntegration failed at coordinates: {current_coords}')
            if len(sol.t) > 0:
                current_x0 = np.copy(sol.y[:, -1]) # Attempt to hold state anyway

        # Build output structure for this single parameter point
        worker_results.append({
            'coords': current_coords,
            'success': sol.success,
            'classification': classification,
            'roots': roots,
            'eigvals': vals,
            'eigvecs': vecs,
            'peaks_freqs': fft_freqs[peaks] if len(peaks) > 0 else np.array([]),
            'peaks_amps': fft_vals[peaks] if len(peaks) > 0 else np.array([]),
            'final_state': sol.y[:, -1] if len(sol.t) > 0 else np.zeros(2 * N_local + 1)
        })

    return worker_results


# =====================================================================
# 2. LAYERED GEOMETRIC SMART SWEEPER
# =====================================================================
class LayeredParallelSmartSweeper:
    '''Slices a 3D parameter grid into 2D layers, evaluating the rows in parallel
    while passing the sequential history downstream along the delta axis for hot starting.
    '''
    def __init__(self, base_config_SI, T_sim, x0, T_eval, use_3d, use_4d, N_points: int = 250):
        self.base_config_SI = base_config_SI
        self.T_sim = T_sim
        self.x0 = x0
        self.use_3d = use_3d
        self.use_4d = use_4d
        self.T_eval = T_eval
        self.N_points = N_points

    def run_layered_parallel_sweep(self, p1_name, p1_vals, p2_name, p2_vals, p3_name, p3_vals):
        results = []
        num_cores = mp.cpu_count()
        
        outer_pbar = tqdm(p1_vals, desc=f'Overall {p1_name} layers', position=0, leave=True)
        
        for v1 in outer_pbar:
            tasks = []

            self.base_config_SI['d'] = to_SI({'sigma': v1})['d']

            base_config = to_unitless(self.base_config_SI)
            base_config['chi_ijk'] = np.load('./tensors/chi_ijk.npy')[:N, :N, :N]
            base_config['chi_ijkl'] = np.load('./tensors/chi_ijkl.npy')[:N, :N, :N]
            base_config['xi'] = np.ones(N)

            outer_pbar.set_postfix_str(f'Current: {p1_name}={v1:.2f}, gamma={base_config['gamma'][0]:.4f}, tau={base_config['tau']:.4f}')
            # We parallelize over parameter 2 (alpha). Each worker gets the entire p3 (delta) array.
            for v2 in p2_vals:
                tasks.append((
                    base_config, self.T_sim, self.x0, self.T_eval, 
                    self.use_3d, self.use_4d, self.N_points,
                    p1_name, v1, p2_name, v2, p3_name, p3_vals
                ))
            
            layer_results = process_map(
                _parallel_worker, 
                tasks, 
                max_workers=num_cores,
                desc=' └─ Parallel Alpha-Row Processing (Delta Sweep)',
                position=1, 
                leave=False
            )
            
            for res_list in layer_results:
                results.extend(res_list)
                
        return results


# =====================================================================
# 3. PIPELINE EXECUTION ENGINE
# =====================================================================
if __name__ == '__main__':
    N = 15
    use_3d = True
    use_4d = True


    base_config_SI = {'N': N,
           'Gammas': np.ones(N)*10,
           'tau': 1/5000,
           'power': 0,
           'detuning': 0,
           'd': 18e-9}
           
    base_config = to_unitless(base_config_SI)

    base_config['chi_ijk'] = np.load('./tensors/chi_ijk.npy')[:N, :N, :N]
    base_config['chi_ijkl'] = np.load('./tensors/chi_ijkl.npy')[:N, :N, :N]
    base_config['xi'] = np.ones(N)
    
    default_x0 = np.zeros(2 * N + 1)
    T_sim = 1000.0
    T_eval = 300.0
    N_points = int(20 * T_eval * np.sqrt(mu_spectrum(N)[-1]) / (2 * np.pi))

    print('Pre-compiling Numba physics engine on main thread...')
    _ = system_numba(0.0, default_x0, base_config)
    _ = Jacobian_numba(0.0, default_x0, base_config)
    print('Pre-compilation finished successfully.\n')

    sweeper = LayeredParallelSmartSweeper(base_config_SI, T_sim, default_x0, T_eval, use_3d, use_4d, N_points=N_points)
    
    # Delta sweep configuration (High to Low)
    deltas = np.linspace(-4, 4, 200)[::-1] 
    alphas = np.linspace(0, 2, 200)
    sigmas = np.arange(30, 90, 10)
    
    # Execute the Pipeline
    print(f'Running sweep with Gamma = {base_config_SI['Gammas'][0]:.4f}, tau = {base_config_SI['tau']:.4f}')
    sweep_data = sweeper.run_layered_parallel_sweep(
        p1_name='sigma', p1_vals=sigmas,
        p2_name='alpha', p2_vals=alphas,
        p3_name='delta', p3_vals=deltas 
    )
    
    # =====================================================================
    # 4. DATA EXTRACTION & COMPRESSED NPZ ARCHIVING
    # =====================================================================
    print('\nStructuring and parsing data blocks...')
    
    # Extract flattened parameters arrays
    coords = np.array([res['coords'] for res in sweep_data])
    success = np.array([res['success'] for res in sweep_data], dtype=bool)
    classifications = np.array([res['classification'] for res in sweep_data], dtype='U30')
    final_states = np.array([res['final_state'] for res in sweep_data])
    
    # Handle variable-length peak arrays safely using object arrays
    peaks_freqs = np.array([res['peaks_freqs'] for res in sweep_data], dtype=object)
    peaks_amps = np.array([res['peaks_amps'] for res in sweep_data], dtype=object)
    
    output_filename = f'bruteforce_sweep_results_N={N}.npz'
    print(f'Saving compiled data blocks to {output_filename}...')
    
    np.savez_compressed(
        output_filename,
        coords=coords,
        success=success,
        classifications=classifications,
        final_states=final_states,
        peaks_freqs=peaks_freqs,
        peaks_amps=peaks_amps,
        sigmas_axis=sigmas,
        alphas_axis=alphas,
        deltas_axis=deltas,
        base_config_SI=base_config_SI,
        N=N
    )
    
    print('Execution complete. Configuration safely written to storage.')