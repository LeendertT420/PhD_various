import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from equations import *

# ============================================================
# APS / REVTeX Standard Style Configuration
# ============================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix',
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'axes.titlesize': 8.5,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.0,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.figsize': (3.37, 5.8),  # Fits single column width
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Helper function to select the fixed point with the largest absolute value (magnitude)
def get_largest_fixed_point(fps):
    fps = np.atleast_2d(fps)
    magnitudes = np.linalg.norm(fps, axis=1)
    return fps[np.argmax(magnitudes)]


# ============================================================
# Data Generation for PANEL (a): Deviation vs Sigma
# ============================================================

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

baseconfig_SI = {
    'N': N,
    'Gammas': np.ones(N) * 10,
    'tau': 1 / 5000,
    'power': 200e-6,
    'detuning': -3e6,
    'd': 18e-9,
}

baseconfig = to_unitless(baseconfig_SI)
baseconfig['xi'] = np.ones(N)
baseconfig['alpha'] = 0.5
baseconfig['delta'] = 0

sigmas = np.linspace(30, 100, 100)

fixed_points_a = np.zeros((N + 1, len(sigmas)))
deviations_a = np.zeros((N + 1, len(sigmas)))

print("Computing Panel (a)...")
for i, sigma in tqdm(enumerate(sigmas)):
  config = baseconfig.copy()
  config['sigma'] = sigma

  fp_0th = fixed_points_0th_order(config)
  fps_num = fixed_points_num(
      system_numba_hidde,
      (config, profiles, W3, W4, use_3d, use_4d),
      num_tries=100,
  )

  # Select the fixed point solution with highest absolute magnitude
  fp_num = get_largest_fixed_point(fps_num)

  fixed_points_a[:N, i] = fp_num[:N]
  fixed_points_a[-1, i] = fp_num[-1]
  deviations_a[:, i] = np.abs(fixed_points_a[:, i] - fp_0th) / fp_0th

# ============================================================
# Data Generation for PANEL (b): Heatmap in delta-alpha plane
# ============================================================

deltas = np.linspace(-5, 2, 200)
alphas = np.linspace(0, 1.0, 200)
DELTA, ALPHA = np.meshgrid(deltas, alphas)

max_error_b = np.zeros((len(alphas), len(deltas)))

print("Computing Panel (b)...")
config_b = baseconfig.copy()
config_b['sigma'] = 30.0

for i, alpha in tqdm(enumerate(alphas)):
  for j, delta in enumerate(deltas):
    config_b['alpha'] = alpha
    config_b['delta'] = delta

    # Create full 2N+1 0th-order state vector in phase space
    fp_0th_scalar = np.max(fixed_points_0th_order(config_b))

    fp_0th_vector = np.zeros(2 * N + 1)
    fp_0th_vector[:N] = fp_0th_scalar  # Mode static displacements x_i*
    fp_0th_vector[N : 2 * N] = 0.0  # Mode static velocities y_i* = 0
    fp_0th_vector[-1] = fp_0th_scalar  # Center height z*

    try:
        fps_num = fixed_points_num(
            system_numba_hidde,
            (config_b, profiles, W3, W4, use_3d, use_4d),
            num_tries=50,
        )

        # Select numerical fixed point solution with highest magnitude
        fp_num = get_largest_fixed_point(fps_num)

        # 1. Full Euclidean distance in 2N+1 phase space [%]
        norm_0th = np.linalg.norm(fp_0th_vector)
        euclidean_error = (
            np.linalg.norm(fp_num - fp_0th_vector) / norm_0th
        ) * 100

        euclidean_error = (
            np.linalg.norm(np.sum(fp_num[:N]) - N*fp_0th_scalar) / (N*fp_0th_scalar)
        ) * 100

        max_error_b[i, j] = euclidean_error

    except Exception:
        max_error_b[i, j] = np.nan
    # ============================================================
# PLOTTING
# ============================================================

fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.37, 5.8))

# ------------------------------------------------------------
# Panel (a): Sigma sweep
# ------------------------------------------------------------

cmap_a = cm.get_cmap('viridis')
bounds = np.arange(0.5, N + 1.5, 1)
norm_a = colors.BoundaryNorm(bounds, cmap_a.N)

for i in range(N):
  ax_a.plot(sigmas, deviations_a[i, :] * 100, color=cmap_a(norm_a(i + 1)))

ax_a.plot(
    sigmas,
    deviations_a[N, :] * 100,
    color='black',
    linestyle='--',
    lw=1.2,
    label=r'$\frac{\left|\,z^*-\overline{x}\,\right|}{\overline{x}}$',
)

sm_a = cm.ScalarMappable(cmap=cmap_a, norm=norm_a)
sm_a.set_array([])
cbar_a = fig.colorbar(sm_a, ax=ax_a, pad=0.02)
cbar_a.set_label(r'Mode index $i$')
cbar_a.set_ticks([1, 4, 8, 12, 15])

ax_a.set_xlabel(r'$\sigma$')
ax_a.set_ylabel(r'$\frac{\left|\,x_i^*-\overline{x}\,\right|}{\overline{x}}$ [$\%$]')
ax_a.set_xlim(np.min(sigmas), np.max(sigmas))
ax_a.grid(True, linestyle='--', alpha=0.4)
ax_a.legend(fontsize=10, loc='upper right', framealpha=0.9)
#ax_a.set_yscale('log')

ax_a.text(
    0.06,
    0.92,
    r'(a) $\delta=0$, $\alpha=0.5$',
    transform=ax_a.transAxes,
    fontsize=10,
    fontweight='bold',
    va='top',
    ha='left',
)

# ------------------------------------------------------------
# Panel (b): Heatmap in (delta, alpha) plane
# ------------------------------------------------------------

# Linear pcolormesh rendering
'''
im = ax_b.pcolormesh(
    DELTA,
    ALPHA,
    max_error_b,
    shading='auto',
    cmap='magma',
    vmin=0,  # Explicit linear minimum
    vmax=np.nanmax(max_error_b),  # Linear maximum
)'''
# log-scale pcolormesh rendering
im = ax_b.pcolormesh(
    DELTA,
    ALPHA,
    max_error_b,
    shading='auto',
    cmap='magma',
    norm=colors.LogNorm(
        vmin=np.nanmin(max_error_b[max_error_b > 0]),
        vmax=np.nanmax(max_error_b),
    ),
    rasterized=True
)

cbar_b = fig.colorbar(im, ax=ax_b, pad=0.02)
cbar_b.set_label(r'$\frac{\left|\,X^*-N\overline{x}\,\right|}{N\overline{x}}$ [$\%$]')

ax_b.set_xlabel(r'$\delta$')
ax_b.set_ylabel(r'$\alpha$')

ax_b.plot(deltas, upper_boundary(N, deltas), color='k', linestyle='-', lw=1.0, label='0th order\nBifurcation boundary')
ax_b.plot(deltas, lower_boundary(N, deltas), color='k', linestyle='-', lw=1.0)

ax_b.set_xlim(np.min(deltas), np.max(deltas))
ax_b.set_ylim(np.min(alphas), np.max(alphas))

ax_b.legend(framealpha=0.9)

ax_b.text(
    0.12,
    0.92,
    r'(b) $\sigma=30$',
    transform=ax_b.transAxes,
    fontsize=10,
    fontweight='bold',
    va='top',
    ha='left',
    color='k',
)

plt.tight_layout()
plt.savefig('fig_fixed_point_errors.pdf', dpi=300)
plt.show()