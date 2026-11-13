import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
from scipy.special import j0, jn_zeros
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import uniform_filter1d

import time

from equations import *
from finding_modelocking_helpers import *


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


config_SI = {
        'N': N,
        'Gammas': np.ones(N) * 10,
        'tau': 1 / 5000,
        'power': 0,
        'detuning': 0,
        'd': to_SI({'sigma': 33})['d']
    }

config = to_unitless(config_SI)
config['mu'] = mu_spectrum(N)
config['xi'] = np.ones(N)

delta = -2
alpha_bif = upper_boundary(N, delta)
alphas = [0.17]#np.linspace(0.145, 0.25, 20)
print(f'alpha bifurcation:{alpha_bif}')

u0 = np.zeros(2*N+1)
u0[:N] = 10/N
u0[-1] = 1/N

dt = 1/(20*np.sqrt(mu_spectrum(N)[-1]))
time_resolution = int(20 * np.sqrt(mu_spectrum(N)[-1])/(2*np.pi))

for i, alpha in enumerate(alphas):

    print(f'\n_____Running alpha sweep {i+1}/{len(alphas)} with alpha = {alpha}_____\n')

    config['alpha'] = alpha
    config['delta'] = delta

    fixed_points = np.array(fixed_points_num(system_numba_hidde, args=(config, profiles, W3, W4, use_3d, use_4d)))
    lower_fixed_points = fixed_points[1:]
    saddle_remnant = np.mean(lower_fixed_points, axis=0)
    main_fixed_point = fixed_points[0]

    if i == 0:
        T_equi = 0
        T_eval = 2000
    else:
        T_equi = 0
        T_eval = 5000

    N_points = T_eval * time_resolution

    t_eval = np.linspace(T_equi, T_equi + T_eval, N_points)
    offset = np.full_like(t_eval, np.sum(fixed_points[0][:N]))

    start = time.time()
    sol = solve_ivp(
        fun=system_numba_hidde,
        t_span=(0, T_equi + T_eval),
        y0=u0,
        args=(config, profiles, W3, W4, use_3d, use_4d),# delta_sweep, delta_start, delta_stop, sweepspeed),
        method='RK45',
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11
    )
    stop = time.time()
    print(f'simulated time {T_equi + T_eval} = {(T_equi + T_eval)/config['Omega1']} seconds')
    print(f'computation time: {stop-start} seconds')

    u0 = sol.y[:, -1]


    v_dot = np.array([system_numba_hidde(t, y, config, profiles, W3, W4, use_3d, use_4d) 
                    for t, y in zip(sol.t, sol.y.T)])

    speed = np.linalg.norm(v_dot, axis=1)


    x = sol.y[:N, :]
    y = sol.y[N:2*N, :]
    film_height = np.sum(x, axis=0)
    dt = t_eval[1] - t_eval[0]

    saddle_remnant = np.zeros(2*N+1)
    x_bar = -2*delta/3 + np.sqrt(delta**2-3)/3
    saddle_remnant[:N] = x_bar
    saddle_remnant[-1] = x_bar

    distance_to_fixed_point = np.sqrt(np.sum((sol.y - saddle_remnant.reshape(-1, 1))**2, axis=0))


    Omegas = np.sqrt(config['mu'])
    modenumbers = np.arange(1, N+1)

    # Compute Hilbert Transform
    analytic_signal = hilbert(x, axis=-1)
    phase_hilbert = np.angle(analytic_signal)  # Instantaneous wrapped phase [-pi, pi]
    phase_hilbert_unwrapped = np.unwrap(phase_hilbert, axis=-1)

    # Hilbert-based detrended Kuramoto Order Parameter
    detrended_phase_h = phase_hilbert_unwrapped - modenumbers.reshape(-1, 1) * phase_hilbert_unwrapped[0, :]
    detrended_kuramoto_h = 1/N * np.abs(np.sum(np.exp(1j * phase_hilbert), axis=0))


    dt = t_eval[1] - t_eval[0]

    all_peak_indices, _ = find_peaks(film_height,
                                    height=0.5*np.max(film_height),
                                    distance=int(1/dt),
                                    prominence=3.5)

    diffs = np.diff(t_eval[all_peak_indices])
    diffs = diffs[(diffs > 1) & (diffs < 20)]


    #xs_detrended, _, _, _ = remove_slow_offset_auto(x, dt, Omegas/(2*np.pi), poly=3, period_factor=8)
    xs_detrended = x

    R_ML = get_mode_locking_order_parameter(xs_detrended)
    #R_ML2 = get_mode_locking_order_parameter2(xs_detrended)
    # Compute a running average of R_ML over 5% of the data window
    window_size = max(len(t_eval) // 100, 10) 
    R_ML_running_avg = uniform_filter1d(R_ML, size=int(2*np.pi/dt))

    mask = R_ML_running_avg > .80


    padded_mask = np.pad(mask, (1, 1), mode='constant', constant_values=0).astype(int)

    diff = np.diff(padded_mask)

    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0]


    # Pad `a` with None to safely handle sequences extending to the end of the array
    t_eval_padded = np.pad(t_eval.astype(object), (0, 1), constant_values=None)

    # Pair start and stop values
    ranges = list(zip(t_eval_padded[starts], t_eval_padded[stops]))


    fig, axs = plt.subplots(7, 1, figsize=(12, 12))

    # --- Left Panel: Time Series & Peaks ---
    axs[0].plot(t_eval, film_height, c="k", zorder=100, label="Film height", alpha=1)
    ax0_right = axs[0].twinx()
    axs[0].set_ylabel('Film Displacement')
    axs[0].hlines(np.sum(fixed_points[0][:N]), t_eval[0], t_eval[-1], color='r', linestyle='--', label=r'$X^*$')
    axs[0].legend()

    for i in range(N):
        axs[1].plot(t_eval, sol.y[i, :], c="k", zorder=100, alpha=0.3)
    axs[1].set_ylabel('Mode Displacement')

    axs[2].plot(t_eval, speed)
    axs[2].set_ylabel('Speed')

    axs[3].plot(t_eval, sol.y[-1,:])
    axs[3].hlines(fixed_points[0][-1], t_eval[0], t_eval[-1], color='r', linestyle='--', label=r'$z^*$')
    axs[3].set_ylabel('Optical Field')
    axs[3].legend()

    #axs[3].plot(t_eval, distance_to_fixed_point, c='orange', zorder=100)

    axs[4].plot(t_eval, R_ML, c="g", zorder=101, alpha=.5)
    axs[4].plot(t_eval, R_ML_running_avg, c="g", zorder=101)
    axs[4].set_ylabel(r'$R_{ML}$')
    axs[4].set_xlabel("Time")


    axs[5].set_xlim(T_equi, T_equi+T_eval)
    axs[5].plot(t_eval, distance_to_fixed_point)
    axs[5].set_ylabel('Distance to saddle remnant')

    fft_freqs = np.fft.rfftfreq(N_points, d=dt)
    fft_vals = (2.0 / N_points) * np.abs(np.fft.rfft(film_height-np.mean(film_height)))

    spectrum = np.sqrt(config['mu'])/2/np.pi
    f_max = spectrum[-1]+np.mean(np.diff(spectrum))

    axs[6].vlines(spectrum, np.min(fft_vals), np.max(fft_vals), color='k', linestyle='--', alpha=.7, label='unperturbed spectrum')
    axs[6].vlines(np.arange(1, N+3, 1)*fft_freqs[np.argmax(fft_vals)], np.min(fft_vals), np.max(fft_vals), color='r', linestyle='--', alpha=.7, label='frequency multiples')
    axs[6].plot(fft_freqs, fft_vals)
    axs[6].set_xlim(0, spectrum[-1]+2*np.mean(np.diff(spectrum)))
    axs[6].legend()
    axs[6].set_xlabel('Frequency')
    axs[6].set_ylabel('FFT')

    for i in np.arange(1,6):
        axs[i].sharex(axs[0])

    plt.tight_layout()
    plt.savefig(f"./figures/alpha_sweep/alpha={alpha}.png")
    plt.show()



plt_3D_modes = True

if plt_3D_modes:
    pass
    # ==============================================================================
# 3D Phase Space Plot: (x_i, y_i, z) per Mode with All Numerical Fixed Points
# ==============================================================================

# Extract all numerically calculated fixed points
num_fp = len(fixed_points)

# Grid setup for N subplots (5 columns for N=15)
ncols = 5
nrows = int(np.ceil(N / ncols))

fig_3d = plt.figure(figsize=(16, 3.2 * nrows))
fig_3d.suptitle("3D Phase Space Trajectories $(x_i, y_i, z)$ & Numerical Fixed Points", fontsize=14, y=0.98)

z_vals = sol.y[-1, :]  # Shared optical field coordinate z

# Color palette & markers for distinct fixed points
fp_colors = ['crimson', 'darkorange', 'purple', 'green', 'cyan']
fp_markers = ['X', 'D', '^', 's', 'p']

for i in range(N):
    ax3d = fig_3d.add_subplot(nrows, ncols, i + 1, projection='3d')
    
    x_mode = sol.y[i, :]
    y_mode = sol.y[N + i, :]
    
    # Trajectory plot
    ax3d.plot(x_mode, y_mode, z_vals, color='tab:blue', alpha=0.5, lw=0.8)
    
    # Plot every numerical fixed point found by fixed_points_num
    for fp_idx, fp in enumerate(fixed_points):
        x_fp = fp[i]
        y_fp = fp[N + i]
        z_fp = fp[-1]
        
        color = fp_colors[fp_idx % len(fp_colors)]
        marker = fp_markers[fp_idx % len(fp_markers)]
        
        # Label each fixed point (only for the first subplot to prevent legend duplication)
        label = f'FP {fp_idx + 1}' if i == 0 else None
        
        ax3d.scatter(
            x_fp, y_fp, z_fp, 
            color=color, 
            marker=marker, 
            s=65, 
            zorder=10, 
            edgecolor='black', 
            linewidth=0.5,
            label=label
        )
    
    # Subplot details
    ax3d.set_xlabel(f'$x_{{{i+1}}}$', labelpad=-2)
    ax3d.set_ylabel(f'$y_{{{i+1}}}$', labelpad=-2)
    ax3d.set_zlabel('$z$', labelpad=-2)
    ax3d.set_title(f'Mode {i+1}', fontsize=10, pad=0)
    ax3d.tick_params(axis='both', which='major', labelsize=8)

# Add single legend for fixed points at the figure level
fig_3d.legend(loc='upper right', bbox_to_anchor=(0.98, 0.96), framealpha=0.9)

plt.tight_layout()
plt.show()






Lyapunov = True
tspan = (0.0, 1000)
epsilon = 1e-12
init1 = sol.y[:,-1]
init2 = init1 + epsilon*np.ones(2*N+1)

if Lyapunov:
    teval = np.linspace(tspan[0], tspan[1], int((tspan[1]-tspan[0])*time_resolution))

    sol1 = solve_ivp(
            fun=system_numba_hidde,
            t_span=tspan,
            y0=init1,  
            args=(config, profiles, W3, W4, use_3d, use_4d),
            method='DOP853',
            rtol=1e-10,
            atol=1e-12,
            t_eval=teval)

    sol2 = solve_ivp(
            fun=system_numba_hidde,
            t_span=tspan,
            y0=init2, 
            args=(config, profiles, W3, W4, use_3d, use_4d),
            method='DOP853',
            rtol=1e-10,
            atol=1e-12,
            t_eval=teval)
    
    print('simulations are the same:', np.allclose(sol1.y, sol2.y))
    print(np.max(np.abs(sol1.y - sol2.y)))


    # ==========================================
    # 1. Distance Calculation & Data Masking
    # ==========================================
    distance = np.linalg.norm(sol2.y - sol1.y, axis=0)
    log_distance = np.log(distance)
    time = sol1.t

    saturation_idx = np.where(distance >= 0.05)[0]
    print(saturation_idx)
    if len(saturation_idx)==0:
        saturation_idx = -1
        fit = False
    else:
        saturation_idx = saturation_idx[0]
        fit = True



    # Fallback: Ensure we have at least a few points to fit
    if saturation_idx < 5:
        saturation_idx = min(20, len(time) - 1)

    t_fit_max = time[saturation_idx]
    print(f"Automatically detected t_fit_max: {t_fit_max:.2f} (Index: {saturation_idx})")

    # Filter data points where t < t_fit_max for the linear regression
    fit_mask = time <= t_fit_max
    time_fit = time[fit_mask]
    log_dist_fit = log_distance[fit_mask]

    # ==========================================
    # 2. Linear Regression (y = mx + c)
    # ==========================================
    # slope (m) is the MLE, intercept (c) is the estimated log(d0)
    slope, intercept = np.polyfit(time_fit, log_dist_fit, 1)
    mle_estimate = slope

    print(f"Calculated MLE (Slope): {mle_estimate:.4f}")

    # Generate the fitted line values over the fitted time range
    fitted_log_distance = slope * time_fit + intercept
    fitted_distance = np.exp(fitted_log_distance)

    # ==========================================
    # 3. Plotting Trajectories and Fit Line
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    X1 = np.sum(sol1.y[:N, :], axis=0)
    X2 = np.sum(sol2.y[:N, :], axis=0)

    # Left Subplot: The Trajectories (showing the divergence split)
    ax1.plot(time, X1, label='Trajectory 1', color='tab:blue', alpha=0.8)
    ax1.plot(time, X2, label='Trajectory 2', color='tab:red', alpha=0.8)
    if fit:
        ax1.axvline(x=t_fit_max, color='gray', linestyle=':', label=f't_fit boundary ({t_fit_max:.2f})')
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('Total film thickness')
    ax1.set_title(f'Trajectories')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(0, tspan[1])
    ax1.legend()

    # Right Subplot: Log Distance & Linear Fit
    ax2.semilogy(time, distance, color='purple', alpha=0.4, label='Actual Distance')
    ax2.semilogy(time_fit, distance[fit_mask], color='purple', linewidth=2, label='Data used for Fit')
    if fit:
        ax2.semilogy(time_fit, fitted_distance, color='r', linestyle='--', linewidth=2,
                label=f'Linear Fit (Slope/MLE = {mle_estimate:.4f})')

        ax2.axvline(x=t_fit_max, color='gray', linestyle=':', label='t_fit boundary')
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('Euclidean Phase-space-distance')
    ax2.set_title('Log Distance vs. Time')
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    ax2.set_xlim(0, tspan[1])
    ax2.legend()

    plt.tight_layout()
    plt.show()