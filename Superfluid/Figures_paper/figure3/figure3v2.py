import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from matplotlib.widgets import Button, Slider
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, hilbert, peak_prominences, savgol_filter
from scipy.special import j0, jn_zeros
from tqdm import tqdm

from equations import *

plt.rcParams.update(
    {
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
        'savefig.bbox': 'tight',
    }
)


def remove_slow_offset_auto(xs, dt, freqs, poly=3, period_factor=8):
  """Automatically choose Savitzky–Golay window per mode

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
  """Computes an AMPLITUDE-WEIGHTED Mode-Locking Order Parameter R_ML(t) in [0, 1].

  This ignores phase noise from weak/zero-amplitude modes.
  xs_array: shape (N_modes, N_time_steps)
  """
  zs = hilbert(xs_array, axis=1)
  adjacent_cross_terms = zs[1:] * np.conj(zs[:-1])
  coherent_sum = np.abs(np.sum(adjacent_cross_terms, axis=0))
  incoherent_sum = np.sum(np.abs(adjacent_cross_terms), axis=0)

  R_ML_weighted = np.divide(
      coherent_sum,
      incoherent_sum,
      out=np.zeros_like(coherent_sum),
      where=incoherent_sum != 0,
  )

  return R_ML_weighted


N = 15
M = int(1e3)
use_3d = True
use_4d = True

profiles = []
for i in range(N):
  r, profile = get_bessel_profile(i + 1, N_points=M)
  profiles.append(profile)

profiles = np.array(profiles).T
W3, W4 = prepare_spatial_vdw_weights(r)

config_SI = {
    'N': N,
    'Gammas': np.ones(N) * 10,
    'tau': 1 / 5000,
    'power': 200e-6,
    'detuning': -3e6,
    'd': 12e-9,
}

config = to_unitless(config_SI)

delta = -2
alpha_bif = upper_boundary(N, delta)

config_final = {
    'N': N,
    'gamma': config['gamma'],
    'mu': mu_spectrum(N),
    'tau': config['tau'],
    'alpha': 0.34,
    'delta': -3,
    'sigma': 43,
    'xi': np.ones(N),
    'nu': config['nu'],
}

u0 = 0.3 * np.random.random(2 * N + 1)

Dt_sim = 2000
t_span = (0.0, Dt_sim)
dt = 1 / (20 * np.sqrt(mu_spectrum(N)[-1]))
N_points = int(Dt_sim / dt)
t_eval = np.linspace(t_span[0], t_span[1], N_points)

# ------------------------------------------------------------
# 1. Base Simulation (t = 0 to 2000)
# ------------------------------------------------------------
sol = solve_ivp(
    fun=system_numba_hidde,
    t_span=t_span,
    y0=u0,
    args=(config_final, profiles, W3, W4, use_3d, use_4d),
    method='RK45',
    t_eval=t_eval,
)

x = sol.y[:N, :]
y = sol.y[N : 2 * N, :]
film_height = np.sum(x, axis=0)

# ------------------------------------------------------------
# 2. Phase Space Divergence Calculation (t >= 1000)
# ------------------------------------------------------------
idx_1000 = np.searchsorted(t_eval, 1000.0)
t_eval_sub = t_eval[idx_1000:]
state_at_1000 = sol.y[:, idx_1000]

epsilon = 1e-8
dim = 2 * N + 1
perturbed_distances = np.zeros((dim, len(t_eval_sub)))

# Unperturbed trajectory segment from t = 1000
unperturbed_traj = sol.y[:, idx_1000:]

for d in range(dim):
  u0_pert = state_at_1000.copy()
  u0_pert[d] += epsilon

  sol_pert = solve_ivp(
      fun=system_numba_hidde,
      t_span=(1000.0, Dt_sim),
      y0=u0_pert,
      args=(config_final, profiles, W3, W4, use_3d, use_4d),
      method='RK45',
      t_eval=t_eval_sub,
  )

  # Compute Euclidean distance in full phase space
  diff = sol_pert.y - unperturbed_traj
  perturbed_distances[d, :] = np.linalg.norm(diff, axis=0)

# Mean Euclidean divergence across all perturbed directions
mean_divergence = np.mean(perturbed_distances, axis=0)

# ------------------------------------------------------------
# Order Parameter & Intervals
# ------------------------------------------------------------
R_ML = get_mode_locking_order_parameter(x)
window_size = max(len(t_eval) // 100, 10)
R_ML_running_avg = uniform_filter1d(R_ML, size=int(2 * 3.8317 / dt))

threshold = 0.80
mask = R_ML_running_avg > threshold

padded_mask = np.pad(mask.astype(int), (1, 1))
diff = np.diff(padded_mask)

starts = np.where(diff == 1)[0]
stops = np.where(diff == -1)[0]

ranges = []
for s, e in zip(starts, stops):
  t_start = t_eval[s]
  t_stop = t_eval[-1] if e >= len(t_eval) else t_eval[e]
  ranges.append((t_start, t_stop))

# Filter ranges for t > 1000 display
ranges_sub = [(max(1000.0, s), e) for s, e in ranges if e > 1000.0]

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(
    3,
    1,
    figsize=(6.8, 7.5),
    sharex=True,
    gridspec_kw={'height_ratios': [1, 1, 1]},
)

# --- Top panel: Film displacement ---
ax1.plot(
    t_eval,
    film_height,
    lw=1.0,
    color='cornflowerblue',
    label='Film displacement',
)
ax1.text(
    0.02,
    0.95,
    '(a)',
    transform=ax1.transAxes,
    fontsize=12,
    fontweight='bold',
    va='top',
    ha='left',
)

first = True
for t_start, t_stop in ranges_sub:
  ax1.axvspan(
      t_start,
      t_stop,
      alpha=0.25,
      color='gold',
      label='Mode-locked interval' if first else None,
  )
  first = False

ax1.set_ylabel(r'Film displacement $\eta(r,\,t)$ [$\kappa/G$]')
ax1.set_title('Intermittent Harmonic Phase Locking')
ax1.legend(loc='upper right')

# --- Middle panel: Order parameter ---
ax2.plot(
    t_eval,
    R_ML,
    color='green',
    alpha=0.25,
    lw=0.8,
    label=r'$R_{\rm HPL}(t)$',
)
ax2.text(
    0.02,
    0.95,
    '(b)',
    transform=ax2.transAxes,
    fontsize=12,
    fontweight='bold',
    va='top',
    ha='left',
)
ax2.plot(
    t_eval, R_ML_running_avg, color='green', lw=1.5, label='Running average'
)
ax2.axhline(
    threshold,
    color='red',
    linestyle='--',
    lw=1.2,
    label=rf'Threshold = {threshold:.2f}',
)

for t_start, t_stop in ranges_sub:
  ax2.axvspan(t_start, t_stop, alpha=0.25, color='gold')

ax2.set_ylim(0, 1.05)
ax2.set_ylabel(r'$R_{\rm HPL}(t)$')
ax2.legend(loc='lower right')

# --- Bottom panel: Phase space divergence ---
ax3.plot(
    t_eval_sub,
    mean_divergence,
    lw=1.2,
    color='crimson',
    label=r'Mean divergence $\langle \|\delta \mathbf{v}(t)\| \rangle$',
)
ax3.text(
    0.02,
    0.95,
    '(c)',
    transform=ax3.transAxes,
    fontsize=12,
    fontweight='bold',
    va='top',
    ha='left',
)

for t_start, t_stop in ranges_sub:
  ax3.axvspan(t_start, t_stop, alpha=0.25, color='gold')

ax3.set_yscale('log')
ax3.set_ylabel(r'Phase space distance')
ax3.set_xlabel(r'Time [$1/\Omega_1$]')
ax3.set_xlim(1000, Dt_sim)
ax3.legend(loc='lower right')

plt.tight_layout()
plt.savefig('fig3.pdf', dpi=300)
plt.show()