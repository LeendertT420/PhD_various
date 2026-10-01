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

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.figsize': (6.8, 8.5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

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

profiles = []
for i in range(N):
    r, profile = get_bessel_profile(i+1, N_points=M)
    profiles.append(profile)

profiles = np.array(profiles).T
#print(np.shape(profiles))

W3, W4 = prepare_spatial_vdw_weights(r)

config_SI = {'N': N,
             'Gammas': np.ones(N)*10,
             'tau': 1/5000,
             'power': 200e-6,
             'detuning': -3e6,
             'd': 12e-9}

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

fixed_points = np.array(fixed_points_num(system_numba_hidde, args=(config_final, profiles, W3, W4, use_3d, use_4d)))
lower_fixed_points = fixed_points[1:]
saddle_remnant = np.mean(lower_fixed_points, axis=0)
main_fixed_point = fixed_points[0]

print(fixed_points)

config_final['alpha'] = 0.34
config_final['delta'] = -3
config_final['sigma'] = 43#config_final['sigma']
u0 = .3*np.random.random(2*N+1)

Dt_sim = 2000
t_span = (0.0, Dt_sim)
dt = 1/(20*np.sqrt(mu_spectrum(N)[-1]))
N_points = int(Dt_sim/dt)
print(N_points, dt, mu_spectrum(N)[-1], np.sqrt(mu_spectrum(N)[-1]) / (2 * np.pi), 1/(np.sqrt(mu_spectrum(N)[-1]) / (2 * np.pi)))
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
    method='RK45',
    t_eval=t_eval
)



x = sol.y[:N, :]
y = sol.y[N:2*N, :]
film_height = np.sum(x, axis=0)
dt = t_eval[1] - t_eval[0]

distance_to_fixed_point = np.mean(sol.y - main_fixed_point.reshape(-1, 1), axis=0)


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

R_ML = get_mode_locking_order_parameter(x)

# Compute a running average of R_ML over 5% of the data window
window_size = max(len(t_eval) // 100, 10) 
R_ML_running_avg = uniform_filter1d(R_ML, size=int(2*3.8317/dt))

mask = R_ML_running_avg > .80


padded_mask = np.pad(mask, (1, 1), mode='constant', constant_values=0).astype(int)

diff = np.diff(padded_mask)

starts = np.where(diff == 1)[0]
stops = np.where(diff == -1)[0]


# Pad `a` with None to safely handle sequences extending to the end of the array
t_eval_padded = np.pad(t_eval.astype(object), (0, 1), constant_values=None)

# Pair start and stop values
ranges = list(zip(t_eval_padded[starts], t_eval_padded[stops]))


# ============================================================
# PLOTTING
# Replace everything below line 419 with this
# ============================================================

# ============================================================
# Detect mode-locked intervals
# ============================================================

threshold = 0.80

mask = R_ML_running_avg > threshold

padded_mask = np.pad(mask.astype(int), (1, 1))
diff = np.diff(padded_mask)

starts = np.where(diff == 1)[0]
stops  = np.where(diff == -1)[0]

ranges = []

for s, e in zip(starts, stops):

    t_start = t_eval[s]

    # Handle interval extending to final sample
    if e >= len(t_eval):
        t_stop = t_eval[-1]
    else:
        t_stop = t_eval[e]

    ranges.append((t_start, t_stop))

print("Detected mode-locked intervals:")
for r in ranges:
    print(r)

# ============================================================
# Plotting
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8, 4),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1]}
)

# ------------------------------------------------------------
# Top panel: Film displacement
# ------------------------------------------------------------

ax1.plot(
    t_eval,
    film_height,
    lw=1.5,
    color="cornflowerblue",
    label="Film displacement"
)

ax1.text(
    0.02, 0.95,
    "(a)",
    transform=ax1.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="left"
)

first = True

for t_start, t_stop in ranges:

    ax1.axvspan(
        t_start,
        t_stop,
        alpha=0.25,
        color="gold",
        label="Mode-locked interval" if first else None
    )

    first = False

ax1.set_ylabel(r"Film displacement $\eta(r,\,t)$ [$\kappa/G$]")
ax1.set_title("Intermittent Harmonic Phase Locking")
ax1.legend()#loc="upper right")

# ------------------------------------------------------------
# Bottom panel: Order parameter
# ------------------------------------------------------------

ax2.plot(
    t_eval,
    R_ML,
    color="green",
    alpha=0.25,
    lw=1,
    label=r"$R_{\rm HPL}(t)$"
)

ax2.text(
    0.02, 0.95,
    "(b)",
    transform=ax2.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="left"
)

ax2.plot(
    t_eval,
    R_ML_running_avg,
    color="green",
    lw=2,
    label=r"Running average"
)

ax2.axhline(
    threshold,
    color="red",
    linestyle="--",
    lw=1.5,
    label=rf"Threshold = {threshold:.2f}"
)

for t_start, t_stop in ranges:

    ax2.axvspan(
        t_start,
        t_stop,
        alpha=0.25,
        color="gold"
    )

ax2.set_ylim(0, 1.05)
ax2.set_ylabel(r"$R_{\rm HPL}(t)$")
ax2.set_xlabel(r"Time [$1/\Omega_1$]")
ax2.legend(loc="lower right")
ax2.set_xlim(1000, t_span[1])

plt.tight_layout()
plt.savefig('fig3.pdf', dpi=300)
plt.show()