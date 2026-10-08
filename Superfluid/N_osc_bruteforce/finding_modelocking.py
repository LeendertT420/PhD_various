import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
from scipy.special import j0, jn_zeros
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import savgol_filter, hilbert
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

import time

from equations import *

def remove_slow_offset_auto(xs, dt, freqs, poly=3, period_factor=8):
    """
    Automatically choose Savitzky–Golay window per mode
    based on estimated oscillation frequency.
    """
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


def get_phase(t, x):
    analytic = hilbert(x)
    return np.unwrap(np.angle(analytic))


def get_mode_locking_order_parameter(xs_array):
    """
    Computes an AMPLITUDE-WEIGHTED Mode-Locking Order Parameter R_ML(t) in [0, 1].
    This ignores phase noise from weak/zero-amplitude modes.
    xs_array: shape (N_modes, N_time_steps)
    """
    # Get complex analytic signals
    zs = hilbert(xs_array, axis=1)
    
    # Calculate z_{n+1} * conjugate(z_n)
    # The angle of this product is exactly (phi_{n+1} - phi_n)
    # The magnitude is A_{n+1} * A_n (this provides the natural amplitude weighting)
    adjacent_cross_terms = zs[1:] * np.conj(zs[:-1])
    
    # Coherent sum (magnitudes of the summed vectors)
    coherent_sum = np.abs(np.sum(adjacent_cross_terms, axis=0))
    
    # Incoherent sum (sum of the magnitudes)
    incoherent_sum = np.sum(np.abs(adjacent_cross_terms), axis=0)
    
    # Safe division
    R_ML_weighted = np.divide(
        coherent_sum, 
        incoherent_sum, 
        out=np.zeros_like(coherent_sum), 
        where=incoherent_sum!=0
    )
    
    return R_ML_weighted


def get_mode_locking_order_parameter2(xs_array):
    """
    Computes an AMPLITUDE-WEIGHTED Mode-Locking Order Parameter R_ML(t) in [0, 1].
    This ignores phase noise from weak/zero-amplitude modes.
    xs_array: shape (N_modes, N_time_steps)
    """
    # Get complex analytic signals
    zs = hilbert(xs_array, axis=1)

    coherent_sum = np.zeros(np.shape(xs_array)[1]).astype('complex128')
    incoherent_sum = np.zeros(np.shape(xs_array)[1]).astype('complex128')

    for i in range(np.shape(xs_array)[0]):
        for j in range(np.shape(xs_array)[0]):
            if i != j:
                crossterm = zs[i] * np.conj(zs[j])

                coherent_sum += crossterm

                incoherent_sum += np.abs(crossterm)
    
    # Calculate z_{n+1} * conjugate(z_n)
    # The angle of this product is exactly (phi_{n+1} - phi_n)
    # The magnitude is A_{n+1} * A_n (this provides the natural amplitude weighting)

    
    # Safe division
    R_ML_weighted = np.divide(
        np.abs(coherent_sum), 
        incoherent_sum, 
        out=np.zeros_like(coherent_sum), 
        where=incoherent_sum!=0
    )
    
    return R_ML_weighted



N = 15
M = int(1e3)
use_3d = True
use_4d = True


config_SI = {
        'N': N,
        'Gammas': np.ones(N) * 10,
        'tau': 1 / 5000,
        'power': 0,
        'detuning': 0,
        'd': to_SI({'sigma': 33})['d']
    }

config = to_unitless(config_SI)

delta = -2
alpha_bif = upper_boundary(N, delta)
print(f'cusp: {cusp(N)}')
print(f'bifurcation: {alpha_bif}')

config_final = {'N': N,
          'gamma': config['gamma'],
          'mu': mu_spectrum(N),
          'tau': config['tau'],
          'alpha': 0.3,
          'delta': delta,
          'sigma': config['sigma'],
          'xi': np.ones(N),
          'nu': config['nu']}

final_config_SI = to_SI(config_final)
print(final_config_SI)

config_final['alpha'] = 0.145
config_final['delta'] = -2 

u0 = .3*np.random.random(2*N+1)



profiles = []
for i in range(N):
    r, profile = get_bessel_profile(i+1, N_points=M)
    profiles.append(profile)

profiles = np.array(profiles).T
#print(np.shape(profiles))

W3, W4 = prepare_spatial_vdw_weights(r)

fixed_points = np.array(fixed_points_num(system_numba_hidde, args=(config_final, profiles, W3, W4, use_3d, use_4d)))
lower_fixed_points = fixed_points[1:]
saddle_remnant = np.mean(lower_fixed_points, axis=0)
main_fixed_point = fixed_points[0]







Dt_sim = 1200
t_span = (0.0, Dt_sim)
dt = 1/(20*np.sqrt(mu_spectrum(N)[-1]))
time_resolution = int(1000 * np.sqrt(mu_spectrum(N)[-1])/(2*np.pi))

N_points = Dt_sim * time_resolution

t_eval = np.linspace(t_span[0], t_span[1], N_points)
offset = np.full_like(t_eval, np.sum(fixed_points[0][:N]))

delta_sweep = False
if delta_sweep:
    delta_start = -2.5
    delta_stop = -2.6
    sweepspeed = abs(delta_start-delta_stop)/1000
    deltas = delta_start - t_eval*sweepspeed
    deltas[deltas < delta_stop] = delta_stop




sol = solve_ivp(
    fun=system_numba_hidde,
    t_span=t_span,
    y0=u0,
    args=(config_final, profiles, W3, W4, use_3d, use_4d),# delta_sweep, delta_start, delta_stop, sweepspeed),
    method='DOP853',
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-8
)

v_dot = np.array([system_numba_hidde(t, y, config_final, profiles, W3, W4, use_3d, use_4d) 
                for t, y in zip(sol.t, sol.y.T)])

speed = np.linalg.norm(v_dot, axis=1)





x = sol.y[:N, :]
y = sol.y[N:2*N, :]
film_height = np.sum(x, axis=0)
dt = t_eval[1] - t_eval[0]

distance_to_fixed_point = np.sqrt(np.mean((sol.y - saddle_remnant.reshape(-1, 1))**2, axis=0))


Omegas = np.sqrt(config_final['mu'])
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


fig, axs = plt.subplots(6, 1, figsize=(12, 12), sharex=True)

# --- Left Panel: Time Series & Peaks ---
axs[0].plot(t_eval, film_height, c="k", zorder=100, label="Film height", alpha=1)
ax0_right = axs[0].twinx()
for i in range(N):
    ax0_right.plot(t_eval, sol.y[i, :], c="k", zorder=100, alpha=0.3)
axs[0].set_ylabel('Film Displacement')

axs[1].plot(t_eval, speed)
axs[1].set_ylabel('Speed (local vector field strength)')

axs[2].plot(t_eval, sol.y[-1,:])
axs[2].set_ylabel('Optical Field')

axs[3].plot(t_eval, distance_to_fixed_point, c='orange', zorder=100)

axs[4].plot(t_eval, R_ML, c="g", zorder=101, alpha=.5)
axs[4].plot(t_eval, R_ML_running_avg, c="g", zorder=101)
for range in ranges:
    axs[4].vlines(range[0], 0, 1)
    axs[4].vlines(range[1], 0, 1)
axs[4].set_xlabel("Time")

fft_freqs = np.fft.rfftfreq(N_points, d=dt)
fft_vals = (2.0 / N_points) * np.abs(np.fft.rfft(film_height-np.mean(film_height)))

spectrum = np.sqrt(config_final['mu'])/2/np.pi
f_max = spectrum[-1]+np.mean(np.diff(spectrum))

axs[5].vlines(spectrum, np.min(fft_vals), np.max(fft_vals), color='k', linestyle='--', alpha=.7, label='unperturbed spectrum')
axs[5].plot(fft_freqs, fft_vals)

plt.tight_layout()
plt.show()


spectrum = False

if spectrum:
    analytic_signal = hilbert(x, axis=-1)

    # 2. Extract instantaneous unwrapped phase for each oscillator
    instant_phase = np.unwrap(np.angle(analytic_signal), axis=-1)

    # 3. Take numerical gradient with respect to time (radians / time_unit)
    instant_freq = np.gradient(instant_phase, dt, axis=-1)/ (2.0 * np.pi)
    plt.figure(figsize=(10, 6))
    N=15
    for i in np.arange(N):
        plt.plot(sol.t, instant_freq[i, :], label=f'Oscillator {i+1}', alpha=0.8, linewidth=1.5)

    plt.xlabel('Time $t$')
    plt.ylabel('Instantaneous Phase $\\theta_i(t)$ (rad)')
    plt.title('Instantaneous Phase Over Time Across All Oscillators')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Show legend if N is reasonable; otherwise skip or adjust
    if N <= 15:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()

    instant_freq_avg = np.mean(instant_freq, axis=-1)

    fft_freqs = np.fft.rfftfreq(N_points, d=dt)
    fft_vals = (2.0 / N_points) * np.abs(np.fft.rfft(film_height-np.mean(film_height)))

    spectrum = np.sqrt(config_final['mu'])/2/np.pi
    f_max = spectrum[-1]+np.mean(np.diff(spectrum))

    plt.vlines(spectrum, np.min(fft_vals), np.max(fft_vals), color='k', linestyle='--', alpha=.7, label='unperturbed spectrum')
    #plt.vlines(instant_freq_avg, np.min(fft_vals), np.max(fft_vals), color='r', linestyle='--', alpha=.7, label='')
    plt.plot(fft_freqs, fft_vals)
    #plt.yscale('log')
    #plt.xlim(0, spectrum[-1]+np.mean(np.diff(spectrum)))
    plt.show()


run_peak_finder = False

if run_peak_finder:
    # Set transient cutoff time
    t_start_peak = 200.0  
    start_idx_peak = np.searchsorted(t_eval, t_start_peak)

    t_signal = t_eval[start_idx_peak:]
    signal = film_height[start_idx_peak:]
    
    # Calculate sampling resolution
    dt_sample = t_signal[1] - t_signal[0]

    # Pre-extract ALL local maxima with lenient conditions
    all_peak_indices, _ = find_peaks(signal, distance=1)
    all_heights = signal[all_peak_indices]
    all_prominences, _, _ = peak_prominences(signal, all_peak_indices)

    # Matplotlib Interactive Window Setup
    fig_pf, ax_pf = plt.subplots(figsize=(11, 7))
    plt.subplots_adjust(bottom=0.35)

    line_sig, = ax_pf.plot(t_signal, signal, color='navy', alpha=0.6, lw=1, label='Signal')
    scatter_peaks, = ax_pf.plot([], [], 'ro', ms=5, zorder=10, label='Filtered Peaks')

    ax_pf.set_xlabel('Time')
    ax_pf.set_ylabel('Film Displacement')
    ax_pf.grid(True, alpha=0.3)
    ax_pf.legend(loc='upper right')

    # Slider Axes Placement
    ax_height = plt.axes([0.20, 0.22, 0.65, 0.03])
    ax_dist   = plt.axes([0.20, 0.16, 0.65, 0.03])
    ax_prom   = plt.axes([0.20, 0.10, 0.65, 0.03])
    ax_button = plt.axes([0.80, 0.02, 0.12, 0.04])

    sig_min, sig_max = np.min(signal), np.max(signal)
    sig_ptp = sig_max - sig_min

    s_height = Slider(ax_height, 'Min Height', sig_min, sig_max, valinit=sig_min)
    s_dist   = Slider(ax_dist, 'Min Distance (pts)', 1, 1000, valinit=1, valstep=1)
    s_prom   = Slider(ax_prom, 'Min Prominence', 0.0, sig_ptp * 0.5, valinit=0.0)
    btn_print = Button(ax_button, 'Print Params')

    def filter_peaks(min_h, min_d, min_p):
        mask = (all_heights >= min_h) & (all_prominences >= min_p)
        candidate_indices = all_peak_indices[mask]
        
        if len(candidate_indices) == 0:
            return np.array([], dtype=int)
        
        if min_d > 1:
            keep = [candidate_indices[0]]
            for idx in candidate_indices[1:]:
                if idx - keep[-1] >= min_d:
                    keep.append(idx)
            return np.array(keep, dtype=int)
        
        return candidate_indices

    def update_peaks(val):
        min_h = s_height.val
        min_d_pts = int(s_dist.val)
        min_p = s_prom.val
        
        min_d_time = min_d_pts * dt_sample
        filtered_indices = filter_peaks(min_h, min_d_pts, min_p)
        
        scatter_peaks.set_data(t_signal[filtered_indices], signal[filtered_indices])
        
        ax_pf.set_title(
            f"Peaks: {len(filtered_indices)} | "
            f"Height ≥ {min_h:.3e} | "
            f"Min Dist: {min_d_time:.3f} t-units ({min_d_pts} pts) | "
            f"Prom ≥ {min_p:.3e}",
            fontsize=10
        )
        fig_pf.canvas.draw_idle()

    def print_params(event):
        min_d_pts = int(s_dist.val)
        min_d_time = min_d_pts * dt_sample
        
        print("\n=======================================")
        print("      OPTIMAL PEAK PARAMETERS          ")
        print("=======================================")
        print(f"  Min Height     : {s_height.val:.6e}")
        print(f"  Min Distance   : {min_d_time:.4f} time units  ({min_d_pts} points)")
        print(f"  Min Prominence : {s_prom.val:.6e}")
        print(f"  dt resolution  : {dt_sample:.6f} time units/point")
        print("=======================================\n")

    s_height.on_changed(update_peaks)
    s_dist.on_changed(update_peaks)
    s_prom.on_changed(update_peaks)
    btn_print.on_clicked(print_params)

    update_peaks(None)
    plt.show()


animate = True

if animate:
        # =====================================================================
        # RADIAL PROFILE VIDEO (Slower Playback + Edge Film Thickness)
        # =====================================================================
        # Zeros of J1(u) for the first N Bessel modes
        zetas = jn_zeros(1, N)

        # Grid across radial cross-section r/R in [-1, 1]
        r_points = 250
        u_grid = np.linspace(-1, 1, r_points)

        # Construct Spatial Basis Matrix S: shape (r_points, N)
        # S_ij = J0(zeta_j * |u_i|) / J0(zeta_j)
        S = np.zeros((r_points, N))
        for i in np.arange(0, N, 1):
                S[:, i] = j0(zetas[i] * np.abs(u_grid)) / j0(zetas[i])

        # Full spatial-temporal field eta(u, t): shape (r_points, N_points)
        x = x
        eta_rt = S @ x

        ## =====================================================================
        # RADIAL PROFILE VIDEO (Filtered Start Time)
        # =====================================================================
        # Set how many initial time units to skip (e.g., x = 200)
        t_start_anim = 200.0  

        # Filter indices where time >= t_start_anim
        start_idx = np.searchsorted(t_eval, t_start_anim)

        t_eval_filt = t_eval[start_idx:]
        eta_rt_filt = eta_rt[:, start_idx:]
        edge_height_filt = film_height[start_idx:]

        # -----------------------------------------------------------------
        # SLOW MOTION & DOWNSAMPLING SETTINGS
        # -----------------------------------------------------------------
        target_frames = 3000 
        stride = max(1, len(t_eval_filt) // target_frames)

        t_anim = t_eval_filt[::stride]
        eta_anim = eta_rt_filt[:, ::stride]
        edge_anim = edge_height_filt[::stride]

        # Setup Animation Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={'height_ratios': [2, 1]})

        # --- Top Subplot: Radial Profile eta(r, t) ---
        line_film, = ax1.plot(u_grid, eta_anim[:, 0], lw=2, color='navy', label='Film Profile')
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(np.min(eta_rt) * 1.2, np.max(eta_rt) * 1.2)
        ax1.set_xlabel('Normalized Radius ($r/R$)')
        ax1.set_ylabel('Film Profile $\eta(r, t)$')
        ax1.grid(True, alpha=0.3)
        title_text = ax1.set_title('')

        # --- Bottom Subplot: Edge Film Thickness Over Time ---
        line_edge, = ax2.plot(t_anim, edge_anim, color='crimson', alpha=0.6, label='Edge Displacement $\eta(R, t)$')
        time_marker = ax2.axvline(t_anim[0], color='black', linestyle='--', label='Current Time')

        # Set x-axis limits to start from t_start_anim
        ax2.set_xlim(t_start_anim, t_span[1])
        ax2.set_ylim(np.min(edge_height_filt) * 1.2, np.max(edge_height_filt) * 1.2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Edge Height $\eta(R, t)$')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        def update(frame):
                current_t = t_anim[frame]
                
                # Update spatial profile
                line_film.set_ydata(eta_anim[:, frame])
                
                # Update time tracker line
                time_marker.set_xdata([current_t, current_t])
                
                title_text.set_text(
                        f'Superfluid Film Cross-Section | Time t = {current_t:.2f} | Edge Displacement = {edge_anim[frame]:.3f}'
                )
                return line_film, time_marker, title_text

        ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t_anim),
        interval=40,  # 10 fps live playback
        blit=True
        )

        plt.tight_layout()

        # Save options (uncomment to export):
        ani.save(f'superfluid_edge_from_t{int(t_start_anim)}.gif', writer='pillow', fps=10)
        ani.save(f'superfluid_edge_from_t{int(t_start_anim)}.mp4', writer='ffmpeg', fps=10, dpi=150)

        plt.show()

Lyapunov = True
tspan = (0.0, 2000)
epsilon = 1e-10
init1 = sol.y[:,-1]
init2 = init1 + epsilon*np.ones(2*N+1)

if Lyapunov:
    teval = np.linspace(tspan[0], tspan[1], int(10*tspan[1]*np.sqrt(config_final['mu'])[-1]/(2*np.pi)))

    sol1 = solve_ivp(
            fun=system_numba_hidde,
            t_span=tspan,
            y0=init1,  
            args=(config_final, profiles, W3, W4, use_3d, use_4d),
            method='RK45',            
            t_eval=teval)

    sol2 = solve_ivp(
            fun=system_numba_hidde,
            t_span=tspan,
            y0=init2, 
            args=(config_final, profiles, W3, W4, use_3d, use_4d),
            method='RK45',           
            t_eval=teval)
    
    print('simulations are the same:', np.allclose(sol1.y, sol2.y))
    print(np.max(np.abs(sol1.y - sol2.y)))


    # ==========================================
    # 1. Distance Calculation & Data Masking
    # ==========================================
    distance = np.linalg.norm(sol2.y - sol1.y, axis=0)
    log_distance = np.log(distance)
    time = sol1.t

    saturation_idx = np.where(distance >= 1)[0]
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