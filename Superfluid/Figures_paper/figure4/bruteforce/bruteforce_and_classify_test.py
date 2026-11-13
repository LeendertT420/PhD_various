import time as system_time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, hilbert
from scipy.special import jn_zeros

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
# 2. SINGLE-POINT TROUBLESHOOTING RUNNER
# =====================================================================
def run_single_simulation_diagnostic(sigma_val, alpha_val, delta_val, base_config_SI, N, M, use_3d, use_4d):
    print("=" * 70)
    print("           SINGLE-POINT SIMULATION & DIAGNOSTIC RUNNER           ")
    print("=" * 70)
    print(f" • Parameters -> sigma (v1): {sigma_val}")
    print(f" • Parameters -> alpha (v2): {alpha_val}")
    print(f" • Parameters -> delta (v3): {delta_val}")
    print(f" • System Size (N)         : {N}")
    print("-" * 70)

    # 1. Setup Profiles and Weights
    profiles = []
    for i in range(N):
        r, profile = get_bessel_profile(i + 1, N_points=M)
        profiles.append(profile)
    profiles = np.array(profiles).T
    W3, W4 = prepare_spatial_vdw_weights(r)

    # 2. Configure Unitless System
    config_SI_layer = base_config_SI.copy()
    config_SI_layer['d'] = to_SI({'sigma': sigma_val}, verbose=False)['d']
    
    base_config = to_unitless(config_SI_layer, verbose=False)
    base_config['xi'] = np.ones(N)
    base_config['chi_ijk'] = np.load('./chi_ijk.npy')
    base_config['chi_ijkl'] = np.load('./chi_ijkl.npy')

    # Update run config with specific target values
    run_config = base_config.copy()
    run_config['sigma'] = sigma_val
    run_config['alpha'] = alpha_val
    run_config['delta'] = delta_val

    # Time configuration
    T_equi = 1500
    T_eval_normal = 1500
    T_eval_HPL = 8000
    T_eval_lyap = 500

    time_resolution = int(20 * np.sqrt(mu_spectrum(N)[-1]) / (2 * np.pi))

    t_eval_normal = np.linspace(T_equi, T_equi + T_eval_normal, time_resolution * T_eval_normal)
    t_eval_HPL = np.linspace(T_equi, T_equi + T_eval_HPL, time_resolution * T_eval_HPL)
    t_eval_lyap = np.linspace(0, T_eval_lyap, time_resolution * T_eval_lyap)

    # Pre-compilation check
    default_x0 = np.zeros(2 * N + 1)
    print("Pre-compiling Numba physics engine...")
    _ = system_numba_hidde(0.0, default_x0, base_config, profiles, W3, W4)
    print("Pre-compilation complete.\n")

    # 3. Lasing Threshold & Pre-classification checks
    lasing_threshold = exact_numerical_lasing_threshold(base_config, delta_val, alpha_bracket=[0.001, 0.5])
    delta_cusp, alpha_cusp = cusp(N)
    boundary = upper_boundary(N, delta_val)

    print(f"[Check] Computed Lasing Threshold for delta={delta_val:.3f}: {lasing_threshold:.4f}")
    print(f"[Check] Cusp boundary: delta={delta_cusp:.3f}, alpha={alpha_cusp:.3f}")
    print(f"[Check] Upper boundary: {boundary:.4f}")

    t_eval = t_eval_normal
    T_eval = T_eval_normal
    check_modelocking = False
    solver = 'RK45'
    rtol = 1e-4
    atol = 1e-6
    classification = 'NOT CLASSIFIED'

    # Decision logic matching the sweep script
    if alpha_val < np.min(lasing_threshold) - 0.01:
        classification = 'BELOW THRESHOLD (Alpha below min threshold)'
        print(f"-> Early Exit: {classification}")
        return
    elif delta_val < delta_cusp and alpha_val < boundary - 0.05:
        classification = 'BELOW THRESHOLD (Cusp region lower bound)'
        print(f"-> Early Exit: {classification}")
        return
    elif (delta_val < delta_cusp and alpha_val < boundary + 0.15) or (delta_val < -1.3 and alpha_val < alpha_cusp + 0.15 and alpha_val > alpha_cusp - 0.05):
        t_eval = t_eval_HPL
        T_eval = T_eval_HPL
        check_modelocking = True
        print("-> Condition met: Enabling High-Power Laser (HPL) & Mode-Locking Evaluation mode.")
    elif delta_val < 1 and alpha_val < lasing_threshold - 0.02:
        classification = 'BELOW THRESHOLD'
        print(f"-> Early Exit: {classification}")
        return
    elif delta_val > 1 and alpha_val < lasing_threshold - 0.1:
        classification = 'BELOW THRESHOLD'
        print(f"-> Early Exit: {classification}")
        return
    elif alpha_val > lasing_threshold + 0.17:
        classification = 'CHAOTIC (High alpha override)'
        print(f"-> Early Exit: {classification}")
        return
    elif delta_val > 0 and alpha_val < lasing_threshold + 0.1:
        solver = 'DOP853'
        rtol = 1e-6
        atol = 1e-8
        print("-> High precision solver (DOP853) selected.")

    # 4. Main Integration
    print(f"\n[Integration] Running solve_ivp with method={solver} for t_span=(0, {T_equi + T_eval})...")
    sim_start = system_time.time()
    sol = solve_ivp(
        fun=system_numba_hidde,
        t_span=(0, T_equi + T_eval),
        y0=default_x0,
        args=(run_config, profiles, W3, W4, use_3d, use_4d),
        method=solver,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )
    sim_duration = system_time.time() - sim_start
    print(f"[Integration] Completed in {sim_duration:.2f}s. Success: {sol.success}")

    if not sol.success:
        print("-> Integration failed!")
        return

    # 5. Analysis & Post-Processing
    x_total = np.sum(sol.y[:N, :], axis=0)
    x_ac = x_total - np.mean(x_total)
    max_ac = np.max(x_ac)
    std_ac = np.std(x_ac)
    print(f"[Analysis] AC signal max amplitude: {max_ac:.6f}, std deviation: {std_ac:.6f}")

    R_HPL = None
    R_HPL_averaged = None
    if check_modelocking:
        R_HPL = get_mode_locking_order_parameter(sol.y[:N])
        window_size = int(2 * jn_zeros(1, 1)[0] * time_resolution)
        R_HPL_averaged = uniform_filter1d(R_HPL, size=window_size)
        print(f"[Analysis] Mean Mode-Locking Order Parameter R_ML: {np.mean(R_HPL):.4f}")

    if max_ac < 1e-2 or std_ac < 1e-12:
        classification = 'BELOW THRESHOLD (Low AC activity)'
    else:
        print("[Lyapunov] Running perturbed trajectory integration for Lyapunov Exponent...")
        x0_1 = sol.y[:, -1]
        x0_2 = x0_1 + 1e-9 * np.ones(2 * N + 1)

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
        sat_indices = np.where(distance >= 1)[0]

        mle = None
        r_squared = None
        peaks_freqs, peaks_amps = np.array([]), np.array([])

        if len(sat_indices) == 0:
            print("[Spectrum] No distance saturation found. Performing FFT peak analysis...")
            fft_freqs = np.fft.rfftfreq(len(x_ac), d=1/time_resolution)
            fft_vals = np.abs(np.fft.rfft(x_ac))
            peaks, _ = find_peaks(fft_vals, prominence=0.5*np.max(fft_vals), height=0.1*np.max(fft_vals), distance=5)

            if len(peaks) == 0:
                classification = 'NOT CLASSIFIED'
            elif len(peaks) == 1:
                classification = 'SINGLE MODE LASING'
            else:
                classification = 'MULTI MODE LASING'
            
            if len(peaks) > 0:
                peaks_freqs = fft_freqs[peaks]
                peaks_amps = fft_vals[peaks]
                print(f"[Spectrum] Detected {len(peaks)} peak(s) at frequencies: {peaks_freqs}")
        else:
            log_distance = np.log(distance)
            sat_index = sat_indices[0]
            time_fit = sol_1.t[:sat_index]
            log_dist_fit = log_distance[:sat_index]

            if len(time_fit) > 2:
                p, _, _, _, _ = np.polyfit(time_fit, log_dist_fit, 1, full=True)
                mle = p[0]
                y_fit = np.polyval(p, time_fit)
                ss_res = np.sum((log_dist_fit - y_fit) ** 2)
                ss_tot = np.sum((log_dist_fit - np.mean(log_dist_fit)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                classification = 'CHAOTIC' if mle > 0 else 'NOT CLASSIFIED'
                print(f"[Lyapunov] Max Lyapunov Exponent (MLE): {mle:.4f} (R² = {r_squared:.4f})")

    print(f"\n=====================================================================")
    print(f" FINAL CLASSIFICATION RESULT: {classification}")
    print(f"=====================================================================\n")

    # 6. Comprehensive Diagnostic Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Diagnostics: $\\sigma$={sigma_val}, $\\alpha$={alpha_val}, $\\delta$={delta_val} | Class: {classification}", fontsize=12, fontweight='bold')

    # Panel 1: Full Displacement Trace
    axes[0, 0].plot(sol.t, x_ac, color='b', lw=0.8)
    axes[0, 0].set_title("AC Total Displacement Trace $x_{ac}(t)$")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: FFT Spectrum & Peaks
    fft_freqs = np.fft.rfftfreq(len(x_ac), d=1/time_resolution)
    fft_vals = np.abs(np.fft.rfft(x_ac))
    axes[0, 1].plot(fft_freqs, fft_vals, color='k', lw=1)
    if len(peaks_freqs) > 0:
        axes[0, 1].scatter(peaks_freqs, peaks_amps, color='r', zorder=5, label='Detected Peaks')
        axes[0, 1].legend()
    axes[0, 1].set_title("FFT Spectrum & Peaks")
    axes[0, 1].set_xlabel("Frequency")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Lyapunov Divergence / Distance
    if 'sol_1' in locals() and 'sol_2' in locals():
        axes[1, 0].semilogy(sol_1.t, distance, color='purple', lw=1.2, label='Trajectory Distance')
        axes[1, 0].axhline(1.0, color='gray', linestyle='--', label='Saturation Threshold')
        axes[1, 0].set_title("Lyapunov Trajectory Divergence")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Euclidean Distance (log scale)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, "Lyapunov not computed\n(Skipped/Below Threshold)", ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Lyapunov Trajectory Divergence")

    # Panel 4: Mode-Locking Order Parameter R_ML(t)
    if check_modelocking and R_HPL is not None:
        axes[1, 1].plot(t_eval, R_HPL, color='teal', alpha=0.4, label='$R_{ML}(t)$ (raw)')
        axes[1, 1].plot(t_eval, R_HPL_averaged, color='darkcyan', lw=1.5, label='Smoothed $R_{ML}$')
        axes[1, 1].set_title("Mode-Locking Order Parameter")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("$R_{ML}$")
        axes[1, 1].set_ylim(-0.05, 1.05)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "Mode-locking check not triggered", ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Mode-Locking Order Parameter")

    plt.tight_layout()
    plt.show()


# =====================================================================
# 3. EXECUTION ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    N = 15
    M = int(1e3)
    use_3d = True
    use_4d = True

    base_config_SI = {
        'N': N,
        'Gammas': np.ones(N) * 10,
        'tau': 1 / 5000,
        'power': 0,
        'detuning': 0,
        'd': 14e-9
    }

    # Specify single test point values here
    TEST_SIGMA = 50
    TEST_ALPHA = 0.341
    TEST_DELTA = -2.96

    run_single_simulation_diagnostic(
        sigma_val=TEST_SIGMA,
        alpha_val=TEST_ALPHA,
        delta_val=TEST_DELTA,
        base_config_SI=base_config_SI,
        N=N,
        M=M,
        use_3d=use_3d,
        use_4d=use_4d
    )