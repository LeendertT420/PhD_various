import copy
import multiprocessing as mp
import time as system_time
from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np

from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, hilbert, savgol_filter
from scipy.special import jn_zeros

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from filelock import FileLock

# Local physics module imports
from equations import *


# =====================================================================
# 1. HELPER & ANALYSIS FUNCTIONS
# =====================================================================
def get_mode_locking_order_parameter(xs_array):
    """Computes an AMPLITUDE-WEIGHTED Mode-Locking Order Parameter R_ML(t) in [0, 1]."""
    zs = hilbert(xs_array, axis=1)
    adjacent_cross_terms = zs[1:] * np.conj(zs[:-1])
    
    coherent_sum = np.abs(np.sum(adjacent_cross_terms, axis=0))
    incoherent_sum = np.sum(np.abs(adjacent_cross_terms), axis=0)
    
    return np.divide(
        coherent_sum, 
        incoherent_sum, 
        out=np.zeros_like(coherent_sum), 
        where=incoherent_sum != 0
    )



# =====================================================================
# 2. WORKER EXECUTOR
# =====================================================================
def _parallel_worker(task_args):
    '''Worker executing a sequential delta sweep without hot-starting.'''
    (base_config, T_sim, x0, T_eval_normal, T_eval_HPL, T_eval_lyap, use_3d, use_4d, time_resolution,
     profiles, W3, W4, p1_name, v1, p2_name, v2, p3_name, p3_vals, lasing_threshold, log_filename) = task_args
    
    N_local = base_config['N']
    worker_results = []

    t_eval_normal   = np.linspace(T_sim, T_sim + T_eval_normal, time_resolution * T_eval_normal)
    t_eval_HPL      = np.linspace(T_sim, T_sim + T_eval_HPL,    time_resolution * T_eval_HPL)
    t_eval_lyap     = np.linspace(0,     T_eval_lyap,          time_resolution * T_eval_lyap)

    delta_cusp, alpha_cusp = cusp(N_local)

    for p3_idx, v3 in enumerate(p3_vals):
        current_coords = (v1, v2, v3)
        
        # Define a clean, comprehensive result template for this iteration
        res = {
            'coords': current_coords,
            'success': True,
            'classification': 'NOT CLASSIFIED',
            'T_eval': T_eval_normal,
            'MLE': None,
            'MLE_r_squared': None,
            'peaks_freqs': np.array([]),
            'peaks_amps': np.array([]),
            'ranges_list': None,
            'R_ML_mean': None,
            'sim_time': 0.0,
            'modelocking_time': 0.0,
            'classification_time': 0.0,
            'skipped': False
        }

        run_config = base_config.copy()
        run_config[p1_name] = v1
        run_config[p2_name] = v2
        run_config[p3_name] = v3

        boundary = upper_boundary(N_local, v3)
        
        # Defaults for this iteration
        t_eval = t_eval_normal
        T_eval = T_eval_normal
        check_modelocking = False
        solver = 'RK45'
        rtol = 1e-4
        atol = 1e-6

        # =====================================================================
        # 1. PRE-CLASSIFICATION CHECKS
        # =====================================================================
        if v2 < np.min(lasing_threshold) - 0.01:
            res['classification'] = 'BELOW THRESHOLD'
            res['skipped'] = True
            worker_results.append(res)
            continue

        if v3 < delta_cusp and v2 < boundary - 0.05:
            res['classification'] = 'BELOW THRESHOLD'
            res['skipped'] = True
            worker_results.append(res)
            continue
        
        elif (v3 < delta_cusp and v2 < boundary + 0.15) or (v3 < -1.3 and v2 < alpha_cusp + 0.15 and v2 > alpha_cusp - 0.05):
            t_eval = t_eval_HPL
            T_eval = T_eval_HPL
            check_modelocking = True

        elif v3 < 1 and v2 < lasing_threshold[p3_idx] - 0.02:
            res['classification'] = 'BELOW THRESHOLD'
            res['skipped'] = True
            worker_results.append(res)
            continue
        
        elif v3 > 1 and v2 < lasing_threshold[p3_idx] - 0.1:
            res['classification'] = 'BELOW THRESHOLD'
            res['skipped'] = True
            worker_results.append(res)
            continue
        
        elif v2 > lasing_threshold[p3_idx] + 0.17:
            res['classification'] = 'CHAOTIC'
            res['skipped'] = True
            worker_results.append(res)
            continue

        elif v3 > 0 and v2 < lasing_threshold[p3_idx] + 0.1:
            solver = 'DOP853'
            rtol = 1e-6
            atol = 1e-8

        # =====================================================================
        # 2. INTEGRATION
        # =====================================================================
        sim_start = system_time.time()

        sol = solve_ivp(
            fun=system_numba_hidde,
            t_span=(0, T_sim + T_eval),
            y0=x0,
            args=(run_config, profiles, W3, W4, use_3d, use_4d),
            method=solver,    
            t_eval=t_eval,
            rtol=rtol,
            atol=atol
        )
        sim_stop = system_time.time()
        
        res['success'] = sol.success
        res['T_eval'] = T_eval
        res['sim_time'] = sim_stop - sim_start

        # =====================================================================
        # 3. CLASSIFICATION & LYAPUNOV
        # =====================================================================
        classification_start = system_time.time()

        x_total = np.sum(sol.y[:N_local, :], axis=0)
        x_ac = x_total - np.mean(x_total)

        if sol.success:
            if np.max(x_ac) < 1e-2 or np.std(x_ac) < 1e-12:
                res['classification'] = 'BELOW THRESHOLD'
                continue
            else:
                x0_1 = sol.y[:, -1]
                x0_2 = x0_1 + 1e-9 * np.ones(2 * N_local + 1)

                sol_1 = solve_ivp(
                    fun=system_numba_hidde, t_span=(0, T_eval_lyap), y0=x0_1, 
                    args=(run_config, profiles, W3, W4, use_3d, use_4d),
                    method='DOP853', t_eval=t_eval_lyap, rtol=1e-6, atol=1e-8
                )
                sol_2 = solve_ivp(
                    fun=system_numba_hidde, t_span=(0, T_eval_lyap), y0=x0_2, 
                    args=(run_config, profiles, W3, W4, use_3d, use_4d),
                    method='DOP853', t_eval=t_eval_lyap, rtol=1e-6, atol=1e-8
                )

                distance = np.linalg.norm(sol_2.y - sol_1.y, axis=0)
                sat_indices = np.where(distance >= 1e-1)[0]


                if len(sat_indices) == 0:
                    fft_freqs = np.fft.rfftfreq(len(x_ac), d=1/time_resolution)
                    fft_vals = np.abs(np.fft.rfft(x_ac))
                    peaks, _ = find_peaks(fft_vals, prominence=0.5*np.max(fft_vals), height=0.1*np.max(fft_vals), distance=5)

                    if len(peaks) == 0:
                        res['classification'] = 'NOT CLASSIFIED'
                    elif len(peaks) == 1:
                        res['classification'] = 'SINGLE MODE LASING'
                    else:
                        res['classification'] = 'MULTI MODE LASING'
                    
                    if len(peaks) > 0:
                        res['peaks_freqs'] = fft_freqs[peaks]
                        res['peaks_amps'] = fft_vals[peaks]
                else:
                    log_distance = np.log(distance)
                    sat_index = sat_indices[0]
                    time_fit = sol_1.t[:sat_index]
                    log_dist_fit = log_distance[:sat_index]

                    if len(time_fit) > 2:
                        p, _, _, _, _ = np.polyfit(time_fit, log_dist_fit, 1, full=True)
                        res['MLE'] = p[0]
                        
                        y_fit = np.polyval(p, time_fit)
                        ss_res = np.sum((log_dist_fit - y_fit) ** 2)
                        ss_tot = np.sum((log_dist_fit - np.mean(log_dist_fit)) ** 2)
                        
                        res['MLE_r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                        res['classification'] = 'CHAOTIC' if res['MLE'] > 0 else 'NOT CLASSIFIED'

            
            modelocking_start = system_time.time()
        
            if check_modelocking:
                R_HPL = get_mode_locking_order_parameter(sol.y[:N_local])
                window_size = int(2 * jn_zeros(1, 1)[0] * time_resolution)
                R_HPL_averaged = uniform_filter1d(R_HPL, size=window_size)
                res['R_ML_mean'] = np.mean(R_HPL)

                ranges_list = []
                for threshold in [60, 70, 75, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5, 99]:
                    mask = R_HPL_averaged > (threshold / 100.0)
                    padded_mask = np.pad(mask, (1, 1), mode='constant', constant_values=0).astype(int)
                    diff = np.diff(padded_mask)

                    starts = np.where(diff == 1)[0]
                    stops = np.where(diff == -1)[0]
                    t_eval_padded = np.pad(t_eval.astype(object), (0, 1), constant_values=None)
                    ranges_list.append(list(zip(t_eval_padded[starts], t_eval_padded[stops])))
                res['ranges_list'] = ranges_list

            modelocking_stop = system_time.time()
            res['modelocking_time'] = modelocking_stop - modelocking_start

            
        else:
            res['classification'] = 'NOT CLASSIFIED'
        
        res['classification_time'] = system_time.time() - classification_start
        worker_results.append(res)

        print(res['classification'])


    # BATCH REAL-TIME TIMING LOGGING
    if log_filename:
        lock_path = log_filename + ".lock"
        with FileLock(lock_path):
            with open(log_filename, "a") as f:
                for r in worker_results:
                    c = r['coords']
                    f.write(f"{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},"
                            f"{r['sim_time']:.4f},{r['modelocking_time']:.4f},"
                            f"{r['classification_time']:.4f}\n")

    return worker_results


# =====================================================================
# 3. LAYERED SWEEPER CLASS
# =====================================================================
class LayeredParallelSmartSweeper:
    def __init__(self, base_config_SI, T_sim, x0, T_eval_normal, T_eval_HPL, T_eval_lyap, use_3d, use_4d, 
                 time_resolution, profiles, W3, W4, num_cores=None, log_filename=None):
        self.base_config_SI = copy.deepcopy(base_config_SI)
        self.T_sim = T_sim
        self.x0 = x0
        self.use_3d = use_3d
        self.use_4d = use_4d
        self.T_eval_normal = T_eval_normal
        self.T_eval_HPL = T_eval_HPL
        self.T_eval_lyap = T_eval_lyap
        self.time_resolution = time_resolution
        self.profiles = profiles
        self.W3 = W3
        self.W4 = W4
        self.num_cores = num_cores or max(1, mp.cpu_count() - 2)
        self.log_filename = log_filename

        # Clear/initialize log file header if active
        if self.log_filename:
            with open(self.log_filename, "w") as f:
                f.write("p1_sigma,p2_alpha,p3_delta,sim_time_s,modelocking_time_s,classification_time_s\n")

    def run_layered_parallel_sweep(self, p1_name, p1_vals, p2_name, p2_vals, p3_name, p3_vals, output_prefix='bruteforce_sweep_results'):
        all_results = []
        N = self.base_config_SI['N']
        
        outer_pbar = tqdm(p1_vals, desc=f'Overall {p1_name} layers', position=0)
        
        for v1 in outer_pbar:
            tasks = []
            
            # Update SI dict and compute unitless config once per layer
            config_SI_layer = self.base_config_SI.copy()
            config_SI_layer['d'] = to_SI({'sigma': v1}, verbose=False)['d']
            
            base_config = to_unitless(config_SI_layer, verbose=False)
            base_config['xi'] = np.ones(N)
            base_config['chi_ijk'] = np.load('./chi_ijk.npy')
            base_config['chi_ijkl'] = np.load('./chi_ijkl.npy')

            outer_pbar.set_postfix_str(f'sigma={v1:.1f}, gamma={base_config["gamma"][0]:.4f}, tau={base_config["tau"]:.4f}')

            # --- ADDED: Nested progress bar for lasing threshold calculation ---
            self.lasing_threshold = []
            for delta in tqdm(p3_vals, desc=' └─ Computing Lasing Thresholds', position=1, leave=False):
                self.lasing_threshold.append(exact_numerical_lasing_threshold(base_config, delta, alpha_bracket=[0.001, 0.5]))
            self.lasing_threshold = np.array(self.lasing_threshold)

            # -----------------------------------------------------------------

            for v2 in p2_vals:
                tasks.append((
                    base_config, self.T_sim, self.x0, self.T_eval_normal, self.T_eval_HPL, self.T_eval_lyap,
                    self.use_3d, self.use_4d, self.time_resolution, self.profiles, self.W3, self.W4,
                    p1_name, v1, p2_name, v2, p3_name, p3_vals, self.lasing_threshold, self.log_filename
                ))
            
            # This will cleanly overlay at position=1 right after the lasing threshold bar disappears
            layer_results = process_map(
                _parallel_worker, 
                tasks, 
                max_workers=self.num_cores,
                desc=' └─ Delta Sweeps (Alpha rows)',
                position=1, 
                leave=False
            )
            
            layer_data = []
            for res_list in layer_results:
                layer_data.extend(res_list)

            
            # Structure and save results for this specific sigma layer immediately
            coords = np.array([res['coords'] for res in layer_data])
            success = np.array([res['success'] for res in layer_data], dtype=bool)
            classifications = np.array([res['classification'] for res in layer_data], dtype='U30')
            T_evals = np.array([res['T_eval'] for res in layer_data], dtype=float)
            MLEs = np.array([res['MLE'] for res in layer_data], dtype=float)
            MLE_r_squareds = np.array([res['MLE_r_squared'] for res in layer_data], dtype=float)
            peaks_freqs = np.array([res['peaks_freqs'] for res in layer_data], dtype=object)
            peaks_amps = np.array([res['peaks_amps'] for res in layer_data], dtype=object)
            ranges_list = np.array([res['ranges_list'] for res in layer_data], dtype=object)
            R_ML_mean = np.array([res['R_ML_mean'] for res in layer_data], dtype=object)
            sim_times = np.array([res['sim_time'] for res in layer_data], dtype=float)
            modelocking_times = np.array([res['modelocking_time'] for res in layer_data], dtype=float)
            classification_times = np.array([res['classification_time'] for res in layer_data], dtype=float)
            skippeds = np.array([res['skipped'] for res in layer_data], dtype=bool)



            output_filename = f"{output_prefix}_{p1_name}_{v1:.6g}_N={N}.npz"
            np.savez_compressed(
                output_filename,
                coords=coords,
                success=success,
                classifications=classifications,
                MLE=MLEs,
                MLE_r_squared=MLE_r_squareds,
                peaks_freqs=peaks_freqs,
                peaks_amps=peaks_amps,
                sigma=v1,
                sigmas_axis=p1_vals,
                alphas_axis=p2_vals,
                deltas_axis=p3_vals,
                T_evals=T_evals,
                sim_times=sim_times,
                modelocking_times=modelocking_times,
                classification_times=classification_times, 
                skippeds=skippeds,
                time_resolution=self.time_resolution,
                base_config_SI=self.base_config_SI,
                N=N,
                ranges_list=ranges_list,
                thresholds=np.array([60, 70, 75, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5, 99]),
                R_ML_mean=R_ML_mean
            )
            
            all_results.extend(layer_data)
                
        return all_results


# =====================================================================
# 4. EXECUTION
# =====================================================================
if __name__ == '__main__':
    N = 15
    M = int(1e3)
    use_3d = True
    use_4d = True

    profiles = []
    for i in range(N):
        r, profile = get_bessel_profile(i+1, N_points=M)
        profiles.append(profile)

    profiles = np.array(profiles).T

    W3, W4 = prepare_spatial_vdw_weights(r)

    base_config_SI = {
        'N': N,
        'Gammas': np.ones(N) * 10,
        'tau': 1 / 5000,
        'power': 0,
        'detuning': 0,
        'd': 14e-9
    }
    
    default_x0 = np.zeros(2 * N + 1)
    T_equi = 1500
    T_eval_normal = 1500
    T_eval_HPL = 8000
    T_eval_lyap = 1000

    time_resolution = int(20 * np.sqrt(mu_spectrum(N)[-1])/(2*np.pi))

    print('Pre-compiling Numba physics engine on main thread...')
    base_config = to_unitless(base_config_SI, verbose=False)
    base_config['xi'] = np.ones(N)   
    _ = system_numba_hidde(0.0, default_x0, base_config, profiles, W3, W4)
    print('Pre-compilation finished successfully.\n')

    # Sweep parameters
    deltas = np.linspace(-3.5, 3.5, 200)
    alphas = np.linspace(0, 0.5, 200)
    sigmas = [30, 40, 50, 60, 70]
    log_filename = "realtime_execution_times_sigmasweep.csv"
    num_cores = 16

    # =====================================================================
    # INFORMATIVE STARTUP SUMMARY BANNER
    # =====================================================================
    print("=" * 70)
    print("                 PARAMETER SWEEP CONFIGURATION SUMMARY                ")
    print("=" * 70)
    print(f" • Simulation Start Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" • System Size (N modes)      : {N}")
    print(f" • Spatial Grid Points (M)    : {M}")
    print(f" • Higher order couplings     : 3D: {use_3d}, 4D: {use_4d}")
    print(f" • dissipation (Gamma)        : {np.mean(base_config_SI['Gammas'])} Hz")
    print(f" • thermal lag (tau)          : {np.mean(base_config_SI['tau'])} s")
    print(f"--------------------------------------------------------------------")
    print(f" • Parameter 1 (sigma layers)       : {sigmas} (Total: {len(sigmas)})")
    print(f" • Parameter 2 (alpha grid)         : {alphas[0]} to {alphas[-1]} ({len(alphas)} points)")
    print(f" • Parameter 3 (delta grid)         : {deltas[0]} to {deltas[-1]} ({len(deltas)} points)")
    print(f" • Total Grid Points per Layer      : {len(alphas) * len(deltas):,}")
    print(f" • Grand Total Simulation Pts       : {len(sigmas) * len(alphas) * len(deltas):,}")
    print(f"--------------------------------------------------------------------")
    print(f" • Equilibrating Time (T_equi)      : {T_equi}")
    print(f" • Evaluation Time (T_eval)         : {T_eval_normal}")
    print(f" • HPL Eval Time (T_eval_HPL)       : {T_eval_HPL}")
    print(f" • Lyapunov Eval Time (T_eval_lyap) : {T_eval_lyap}")
    print(f" • timestep                         : {1/time_resolution:.4f}")
    print(f"--------------------------------------------------------------------")
    print(f" • Parallel Workers / Cores   : {num_cores} active CPU cores")
    print(f" • Real-time Log File         : {log_filename}")
    print("=" * 70 + "\n")



    sweeper = LayeredParallelSmartSweeper(
        base_config_SI, T_equi, default_x0, T_eval_normal, T_eval_HPL, T_eval_lyap, use_3d, use_4d,
        time_resolution, profiles=profiles, W3=W3, W4=W4,
        num_cores=num_cores, log_filename=log_filename
    )
    
    # Executes sweep and saves per sigma layer directly
    sweep_data = sweeper.run_layered_parallel_sweep(
        p1_name='sigma', p1_vals=sigmas,
        p2_name='alpha', p2_vals=alphas,
        p3_name='delta', p3_vals=deltas,
        output_prefix='bruteforce_sweep_results'
    )

    print('\nExecution complete. All per-sigma files saved successfully.')