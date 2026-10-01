import copy
import multiprocessing as mp
import sys
import time as system_time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, hilbert, savgol_filter
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from filelock import FileLock

# Local physics module imports
from equations import *


# =====================================================================
# 1. HELPER & ANALYSIS FUNCTIONS
# =====================================================================
def remove_slow_offset_auto(xs, dt, freqs, poly=3, period_factor=8):
    """Automatically choose Savitzky–Golay window per mode based on estimated frequency."""
    N_modes, N_t = xs.shape
    xs_detrended = np.zeros_like(xs)
    trends = np.zeros_like(xs)
    windows = []

    for i in range(N_modes):
        f = freqs[i]
        if f == 0:
            window = 101
        else:
            period_samples = int(1 / (f * dt))
            window = period_factor * period_samples

        window = max(window, poly + 2)
        if window % 2 == 0:
            window += 1
        window = min(window, N_t - (1 - N_t % 2))

        trend = savgol_filter(xs[i], window_length=window, polyorder=poly)
        xs_detrended[i] = xs[i] - trend
        trends[i] = trend
        windows.append(window)

    return xs_detrended, trends, np.array(windows), freqs


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
    '''Worker executing a sequential delta sweep with hot-starting.'''
    (base_config, T_sim, x0_init, T_eval, T_lyap, use_3d, use_4d, N_points_eval, N_points_lyap,
     profiles, W3, W4, p1_name, v1, p2_name, v2, p3_name, p3_vals, log_filename) = task_args
    
    N_local = base_config['N']
    current_x0 = x0_init
    worker_results = []

    for p3_idx, v3 in enumerate(p3_vals):
        current_coords = (v1, v2, v3)
        run_config = base_config.copy()
        run_config[p1_name] = v1
        run_config[p2_name] = v2
        run_config[p3_name] = v3

        boundary = upper_boundary(N_local, v3)
        if np.isnan(boundary) or v2 < boundary-0.05 or v2 > boundary + 0.15:
            worker_results.append({
                'coords': current_coords, 'success': False, 'classification': 'BELOW THRESHOLD',
                'roots': None, 'eigvals': None, 'eigvecs': None, 'peaks_freqs': np.array([]),
                'peaks_amps': np.array([]), 'final_state': np.zeros(2*N_local+1), 'ranges_list': None,
                'R_ML_mean': None, 'sim_time': 0.0, 'modelocking_time': 0.0, 'classification_time': 0.0,
                'boundary': boundary
            })
            continue

        t_eval = np.linspace(T_sim, T_sim + T_eval, N_points_eval)

        # 1. INTEGRATION
        sim_start = system_time.time()
        sol = solve_ivp(
            fun=system_numba_hidde,
            t_span=(0, T_sim + T_eval),
            y0=current_x0,
            args=(run_config, profiles, W3, W4, use_3d, use_4d),
            method='RK45',    
            t_eval=t_eval
        )
        sim_stop = system_time.time()
        
        dt = t_eval[1] - t_eval[0]
        
        # 2. MODE-LOCKING ANALYSIS
        modelocking_start = system_time.time()
        xs_detrended, _, _, _ = remove_slow_offset_auto(sol.y[:N_local], dt, np.sqrt(run_config['mu'])/(2*np.pi))
        
        R_ML = get_mode_locking_order_parameter(xs_detrended)
        window_size = max(len(t_eval) // 100, 10) 
        R_ML_running_avg = uniform_filter1d(R_ML, size=window_size)
        R_ML_mean = float(np.mean(R_ML))

        ranges_list = []
        for threshold in [60, 70, 75, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5, 99]:
            mask = R_ML_running_avg > (threshold / 100.0)
            padded_mask = np.pad(mask, (1, 1), mode='constant', constant_values=0).astype(int)
            diff = np.diff(padded_mask)

            starts = np.where(diff == 1)[0]
            stops = np.where(diff == -1)[0]
            t_eval_padded = np.pad(t_eval.astype(object), (0, 1), constant_values=None)
            ranges_list.append(list(zip(t_eval_padded[starts], t_eval_padded[stops])))

        modelocking_stop = system_time.time()

        # 3. CLASSIFICATION & LYAPUNOV
        classification_start = system_time.time()
        classification = 'NOT CLASSIFIED'
        roots = None
        peaks, fft_freqs, fft_vals = np.array([]), np.array([]), np.array([])

        if sol.success:
            current_x0 = sol.y[:, -1] # Update hot-start vector
            
            x_total = np.sum(sol.y[:N_local, :], axis=0)
            x_ac = x_total - np.mean(x_total)
            plt.plot(t_eval, x_total)
            plt.show()

            if np.max(x_ac) < 1e-3:
                classification = 'BELOW THRESHOLD'
            else:
                roots = fixed_points_num(system_numba_hidde, args=(run_config, profiles, W3, W4, use_3d, use_4d))
                
                fft_freqs = np.fft.rfftfreq(N_points_eval, d=T_eval / N_points_eval)
                sigma_x = np.std(x_ac)
                
                if sigma_x < 1e-12:
                    classification = 'NO VARIANCE'
                else:
                    x_normalized = x_ac / sigma_x
                    fft_vals = (2.0 / N_points_eval) * np.abs(np.fft.rfft(x_normalized))
                    peaks, _ = find_peaks(fft_vals, prominence=0.1, height=0.05)
                    
                    if 0 < T_lyap < T_eval:
                        x0_2 = sol.y[:, -N_points_lyap] + 1e-9 * np.ones(2 * N_local + 1)

                        t_lyap = np.linspace(T_sim + T_eval - T_lyap, T_sim + T_eval, N_points_lyap)

                        sol_2 = solve_ivp(
                            fun=system_numba_hidde,
                            t_span=(T_sim + T_eval - T_lyap, T_sim + T_eval),
                            y0=x0_2, 
                            args=(run_config, profiles, W3, W4, use_3d, use_4d),
                            method='RK45',    
                            t_eval=t_lyap
                        )

                        distance = np.linalg.norm(sol_2.y - sol.y[:, -N_points_lyap:], axis=0)
                        plt.plot(t_lyap, distance)
                        plt.yscale('log')
                        plt.show()
                    
                        if len(np.where(distance >= 1)[0]) == 0:
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
                                classification = 'MODE LOCKED' if is_mode_locked else 'MULTI MODE LASING'
                        else:
                            log_distance = np.log(distance)
                            time_pts = sol.t[-N_points_lyap:]
                            sat_indices = np.where(distance >= 1)[0]
                            time_fit = time_pts[time_pts <= time_pts[sat_indices[0]]]
                            log_dist_fit = log_distance[:len(time_fit)]

                            if len(time_fit) > 1:
                                mle_estimate, _ = np.polyfit(time_fit, log_dist_fit, 1)
                                classification = 'CHAOTIC' if mle_estimate > 0 else 'NOT CLASSIFIED'
        else:
            current_x0 = np.zeros(2*N_local+1)
        
        classification_stop = system_time.time()

        worker_results.append({
            'coords': current_coords,
            'success': sol.success,
            'classification': classification,
            'roots': roots,
            'peaks_freqs': fft_freqs[peaks] if len(peaks) > 0 else np.array([]),
            'peaks_amps': fft_vals[peaks] if len(peaks) > 0 else np.array([]),
            'final_state': sol.y[:, -1] if len(sol.t) > 0 else np.zeros(2 * N_local + 1),
            'ranges_list': ranges_list,
            'R_ML_mean': R_ML_mean,
            'sim_time': sim_stop - sim_start,
            'modelocking_time': modelocking_stop - modelocking_start,
            'classification_time': classification_stop - classification_start,
            'boundary': boundary
        })

    # BATCH REAL-TIME TIMING LOGGING
    if log_filename:
        lock_path = log_filename + ".lock"
        with FileLock(lock_path):
            with open(log_filename, "a") as f:
                for r in worker_results:
                    c = r['coords']
                    f.write(f"{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},"
                            f"{r['sim_time']:.4f},{r['modelocking_time']:.4f},"
                            f"{r['classification_time']:.4f},"
                            f"{r['boundary']:.4f}\n")

    return worker_results


# =====================================================================
# 3. LAYERED SWEEPER CLASS
# =====================================================================
class LayeredParallelSmartSweeper:
    def __init__(self, base_config_SI, T_sim, x0, T_eval, T_lyap, use_3d, use_4d, 
                 N_points_eval, N_points_lyap, profiles, W3, W4, num_cores=None, log_filename=None):
        self.base_config_SI = copy.deepcopy(base_config_SI)
        self.T_sim = T_sim
        self.x0 = x0
        self.use_3d = use_3d
        self.use_4d = use_4d
        self.T_eval = T_eval
        self.T_lyap = T_lyap
        self.N_points_eval = N_points_eval
        self.N_points_lyap = N_points_lyap
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
            config_SI_layer['d'] = to_SI({'sigma': v1})['d']
            
            base_config = to_unitless(config_SI_layer)
            base_config['xi'] = np.ones(N)

            outer_pbar.set_postfix_str(f'sigma={v1:.4e}, gamma={base_config["gamma"][0]:.4f}, tau={base_config["tau"]:.4f}')

            for v2 in p2_vals:
                tasks.append((
                    base_config, self.T_sim, self.x0, self.T_eval, self.T_lyap,
                    self.use_3d, self.use_4d, self.N_points_eval, self.N_points_lyap,
                    self.profiles, self.W3, self.W4,
                    p1_name, v1, p2_name, v2, p3_name, p3_vals, self.log_filename
                ))
            
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
            final_states = np.array([res['final_state'] for res in layer_data])
            peaks_freqs = np.array([res['peaks_freqs'] for res in layer_data], dtype=object)
            peaks_amps = np.array([res['peaks_amps'] for res in layer_data], dtype=object)
            ranges_list = np.array([res['ranges_list'] for res in layer_data], dtype=object)
            R_ML_mean = np.array([res['R_ML_mean'] for res in layer_data], dtype=object)

            output_filename = f"{output_prefix}_{p1_name}_{v1:.6g}_N={N}.npz"
            np.savez_compressed(
                output_filename,
                coords=coords,
                success=success,
                classifications=classifications,
                final_states=final_states,
                peaks_freqs=peaks_freqs,
                peaks_amps=peaks_amps,
                sigma=v1,
                sigmas_axis=p1_vals,
                alphas_axis=p2_vals,
                deltas_axis=p3_vals,
                T_sim=self.T_sim,
                T_eval=self.T_eval,
                T_lyap=self.T_lyap,
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
    T_sim = 1000
    T_eval = 1000
    T_lyap = 500

    time_resolution = int(10 * np.sqrt(mu_spectrum(N)[-1])/(2*np.pi))
    N_points_eval = T_eval * time_resolution
    N_points_lyap = T_lyap * time_resolution

    print('Pre-compiling Numba physics engine on main thread...')
    base_config = to_unitless(base_config_SI)
    base_config['xi'] = np.ones(N)

    print(f"TOTAL SIMULATION TIME: {(T_sim+T_eval)/base_config['Omega1']} seconds")
    
    _ = system_numba_hidde(0.0, default_x0, base_config, profiles, W3, W4)
    print('Pre-compilation finished successfully.\n')

    # Sweep parameters
    deltas = np.linspace(-4, -4, 10)[::-1] 
    alphas = np.linspace(0, 1, 10)
    ds = np.linspace(9.385, 18.31, 13)
    sigmas = [30, 50, 70]
    log_filename = "realtime_execution_times_sigmasweep.csv"

    sweeper = LayeredParallelSmartSweeper(
        base_config_SI, T_sim, default_x0, T_eval, T_lyap, use_3d, use_4d,
        N_points_eval=N_points_eval, N_points_lyap=N_points_lyap,
        profiles=profiles, W3=W3, W4=W4,
        num_cores=1, log_filename=log_filename
    )
    
    print(f'Running sweep with Gamma = {base_config_SI["Gammas"][0]:.4f}, tau = {base_config_SI["tau"]:.4f}')
    
    # Executes sweep and saves per sigma layer directly
    sweep_data = sweeper.run_layered_parallel_sweep(
        p1_name='sigma', p1_vals=sigmas,
        p2_name='alpha', p2_vals=alphas,
        p3_name='delta', p3_vals=deltas,
        output_prefix='bruteforce_sweep_results'
    )

    print('\nExecution complete. All per-sigma files saved successfully.')