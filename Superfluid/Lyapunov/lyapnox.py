import nolds
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import get_window
import os

plt.rcParams.update({'mathtext.fontset': 'cm'})
plt.rcParams.update({'font.family': 'STIXGeneral'})
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.xmargin': 0})

N = 15
sigma = 60
alpha_target = 1.608#2#0.693
delta_target = 2.04#1#-2.73
i_alpha = int(alpha_target/2*200-1)
i_delta = int((delta_target+4)/7*250 -1)

# File paths
raw_filename = f'C:\\Users\\lion_remote\\Documents\\Geert\\results\\sweep_results_N={N}_sigma={sigma}.npz'
cache_filename = f'C:\\Users\\lion_remote\\Documents\\Geert\\results\\sliced_amp_N={N}_sigma={sigma}_a{i_alpha}_d{i_delta}.npy'
time_cache_filename = f'C:\\Users\\lion_remote\\Documents\\Geert\\results\\time_N={N}_sigma={sigma}.npy'

# =====================================================================
# DATA LOADING (With Caching)
# =====================================================================
if os.path.exists(cache_filename) and os.path.exists(time_cache_filename):
    print("Loading pre-sliced data from cache (fast)...")
    amp = np.load(cache_filename)
    time = np.load(time_cache_filename)
    print("Data successfully loaded from cache.")
else:
    print(f'Cache not found. Loading raw big data from {raw_filename} (slow)...')
    raw_data = np.load(raw_filename)

    alpha = raw_data['alphas'][i_alpha]
    delta = raw_data['deltas'][i_delta]
    print(f'Indexed at alpha={alpha}, delta={delta}')

    # Slicing the data
    amp = np.sum(raw_data['states'][i_alpha, i_delta, :N, :], axis=0)
    time = raw_data['time']
    
    print('Saving sliced data to separate files for future fast loading...')
    np.save(cache_filename, amp)
    np.save(time_cache_filename, time)
    print('Data successfully saved.')

print(f'{len(amp)} timesteps ready.')


def get_fft(x, fs, window='hann', nfft=None, detrend=True):
    x = np.asarray(x)

    # --- remove DC / trend ---
    if detrend:
        x = x - np.mean(x)

    N = len(x)

    # --- windowing ---
    if window is not None:
        w = get_window(window, N)
        xw = x * w
        scale = np.sum(w) / N
    else:
        xw = x
        scale = 1.0

    # --- zero padding (optional) ---
    if nfft is None:
        nfft = N
    elif nfft < N:
        raise ValueError("nfft must be >= len(x)")

    # --- FFT (real → positive freqs only) ---
    X = np.fft.rfft(xw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1/fs)

    # --- amplitude spectrum ---
    amplitude = np.abs(X) / (N * scale)

    # correct for single-sided spectrum
    amplitude[1:-1] *= 2

    return freqs, amplitude


# =====================================================================
# PLOTTING
# =====================================================================


def freq_spectrum(N : int):
    from scipy.special import jn_zeros
    return jn_zeros(1, N) / jn_zeros(1, 1) / (2*np.pi)
#amp = np.cos(2*np.pi*time)
freq_spectrum = freq_spectrum(N)
print(freq_spectrum)
# =====================================================================
# LYAP
# =====================================================================
dt = np.mean(np.diff(time))
print(dt)
lag = int(1/(4*dt))
print(lag)
theiler_window = int(2/dt)
print(theiler_window)

# 1. Tell nolds to track a much longer trajectory to see the saturation plateau
full_trajectory_len = 100
fit_cutoff = 20        # Only fit the linear behavior up to this step

lle, debug_data = nolds.lyap_r(
    amp, 
    emb_dim=10, 
    lag=None, 
    debug_data=True, 
    trajectory_len=full_trajectory_len, # Track the full curve
    min_tsep=50
)
print(debug_data)

# Extract full data arrays from nolds
steps = debug_data[0]             # nolds returns the actual x-axis steps here
avg_log_distances = debug_data[1] # nolds returns the actual y-axis log-distances here

# 2. Slice the arrays to isolate the initial linear region for fitting
linear_steps = steps[:fit_cutoff]
linear_distances = avg_log_distances[:fit_cutoff]

# 3. Manually recalculate the linear fit on just the first part
slope, intercept = np.polyfit(linear_steps, linear_distances, 1)
custom_fit_line = slope * steps + intercept  # Extrapolate the line across the whole plot

print(f"Original nolds LLE (fitted to {full_trajectory_len} steps): {lle:.4f}")
print(f"Corrected LLE (fitted only to first {fit_cutoff} steps): {slope:.4f}")

# =====================================================================
# PLOTTING AS A SCATTER PLOT
# =====================================================================
fig, axs = plt.subplots(1, 3, figsize=(14, 5))


axs[0].plot(time, amp)
axs[0].set_xlim(1000, 1030)

freqs, fourier_amps = get_fft(x = amp, fs = 1/(np.average(np.diff(time))))
axs[1].plot(freqs, fourier_amps)
axs[1].set_xlim(0, freq_spectrum[-1]+np.mean(np.diff(freq_spectrum)))


# Plot the full data as a scatter plot to emphasize the flattening out
axs[2].scatter(steps, avg_log_distances, color='black', s=15, alpha=0.7, label=r'Calculated $\langle \ln(d_k) \rangle$')

# Highlight the region used for the fit
axs[2].scatter(linear_steps, linear_distances, color='darkorange', s=25, zorder=3, label=r'Region used for fit')

# Plot the custom linear fit line
axs[2].plot(steps, custom_fit_line, color='red', linestyle='--', linewidth=2, 
         label=rf'Linear Fit ($\lambda_{{true}} \approx {slope:.4f}$)')

# Visual indicator for where saturation/flattening dominates
axs[2].axvline(x=fit_cutoff, color='gray', linestyle=':', alpha=0.7, label='Fit Cutoff')

# Styling
axs[2].set_xlabel(r'Time Steps ($k$)')
axs[2].set_ylabel(r'Avg. Log-Separation $\langle \ln(d_k) \rangle$')
axs[2].set_title('Phase Space Divergence & Saturation Plateau')
axs[2].set_xlim(0, full_trajectory_len)
axs[2].legend()
axs[2].grid(True, linestyle=':', alpha=0.5)

plt.show()