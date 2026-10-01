from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Added ConnectionPatch for inset lines
from matplotlib.patches import ConnectionPatch
import numpy as np
from bruteforce.equations import *

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
    'figure.figsize': (7.0, 7.5),  # Two-column width format
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def calculate_mode_locked_fraction(
    ranges_list_threshold, classification, T_sim, T_eval
):
  """Calculates the total time fraction spent in mode-locked intervals."""
  if classification != 'CHAOTIC':
    return 0.0

  if (
      ranges_list_threshold is None
      or len(ranges_list_threshold) == 0
      or T_eval <= 0
  ):
    return 0.0

  total_ml_time = 0.0

  for t_start, t_stop in ranges_list_threshold:
    if t_start is not None and t_stop is not None:
      total_ml_time += t_stop - t_start
    elif t_start is not None and t_stop is None:
      total_ml_time += T_eval + T_sim - t_start

  return min(1.0, total_ml_time / T_eval)


def load_and_process_layer(
    filename, threshold_idx=3, mode_locking_metrics=True
    ):  # Default threshold_idx 3 = 80%
    """Loads NPZ layer data and reshapes arrays into 2D grids for pcolormesh."""
    data = np.load(filename, allow_pickle=True)

    alphas = data['alphas_axis']
    deltas = data['deltas_axis']
    coords = data['coords']
    base_config_SI = data['base_config_SI'].item()  # Convert from 0-d array to dict
    classifications = data['classifications']
    print(classifications)
    for c in set(classifications):
        count = np.sum(classifications == c)
        print(f"Category '{c}': {count} occurrences")
    if mode_locking_metrics:
        ranges_list = data['ranges_list']
    T_sim = float(data['T_sim'])
    T_eval = float(data['T_eval'])
    print(T_sim, T_eval)
    print(np.min(alphas), np.max(alphas), np.min(deltas), np.max(deltas))
    shape = (len(alphas), len(deltas))

    # 1. Classification map
    class_map = np.zeros(shape, dtype=int)
    class_dict = {
        'BELOW THRESHOLD': 0,
        'SINGLE MODE LASING': 1,
        'MULTI MODE LASING': 2,
        'CHAOTIC': 3,
        'MODE LOCKED': 3,
    }

    # 2. Mode-locked fraction map
    ml_fraction_map = np.zeros(shape)
  
    for idx, c in enumerate(coords):
        # Find grid indices
        i = np.argmin(np.abs(alphas - c[1]))
        j = np.argmin(np.abs(deltas - c[2]))

        cat = classifications[idx]

        class_map[i, j] = class_dict.get(cat, 0)


        # Compute ML fraction from ranges_list at target threshold
        if mode_locking_metrics and ranges_list[idx] is not None and len(ranges_list[idx]) > threshold_idx:
            r_thresh = ranges_list[idx][threshold_idx]
            ml_fraction_map[i, j] = calculate_mode_locked_fraction(
                r_thresh, cat, T_sim, T_eval
            )

    return deltas, alphas, base_config_SI, class_map, ml_fraction_map


# ============================================================
# PLOTTING
# ============================================================

N = 15
sigmas = [30, 40, 50]
threshold_idx = 3  # Index corresponding to threshold = 80%

chi_ijk = np.load('tensors/chi_ijk.npy')
chi_ijkl = np.load('tensors/chi_ijkl.npy')

# Setup 3x2 Grid
fig, axes = plt.subplots(3, 2, sharex='col', sharey='col', gridspec_kw={'wspace': 0.08, 'hspace': 0.08})

# Discrete map configuration for Column 1
categories = [
    'Below Threshold',
    'Single-Mode',
    'Multi-Mode',
    'Chaos / Phase-Locked',
]
cmap_discrete = colors.ListedColormap(
    ['#d9d9d9', '#2b83ba', '#abdda4', '#d7191c']
)
norm_discrete = colors.BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap_discrete.N)

# Continuous map configuration for Column 2
cmap_continuous = cm.get_cmap('magma')

for idx, sigma in enumerate(sigmas):
  ax_left = axes[idx, 0]
  ax_right = axes[idx, 1]

  # Move right column y-ticks and y-label to the right side
  ax_right.yaxis.tick_right()
  ax_right.yaxis.set_label_position('right')

  # File naming conventions (Adjust prefix if necessary)
  file_overview = f'./dummydata/bruteforce_sweep_results_sigma_{sigma}_N=15.npz'
  file_zoom = f'./dummydata/bruteforce_zoom_results_sigma_{sigma}_N=15.npz'

  # Load Left Column Data (Overview)
  d_left, a_left, base_config_SI, class_map, _ = load_and_process_layer(
      file_overview, threshold_idx, mode_locking_metrics=False
  )
  print(class_map)
  config = to_unitless(base_config_SI, verbose=False)
  config['sigma'] = sigma
  config['chi_ijk'] = chi_ijk
  config['chi_ijkl'] = chi_ijkl
  config['xi'] = np.ones(N)
  D_left, A_left = np.meshgrid(d_left, a_left)

  # Plot Left Column: Dynamical Classification
  im_left = ax_left.pcolormesh(
      D_left,
      A_left,
      class_map,
      cmap=cmap_discrete,
      norm=norm_discrete,
      shading='auto',
      rasterized=True,
  )

  ax_left.plot(
      d_left, upper_boundary(N, d_left), color='k', linestyle='-', lw=0.8
  )
  ax_left.plot(
      d_left, lower_boundary(N, d_left), color='k', linestyle='-', lw=0.8
  )

  exact_thresholds = []
    
  print("Calculating exact numerical thresholds (this will take a moment)...")
  print(config)
  for d in d_left:
    config['delta'] = d
    alpha_c = exact_numerical_lasing_threshold(config, d)
    exact_thresholds.append(alpha_c)
  print(exact_thresholds)

  ax_left.plot(
      d_left, exact_thresholds, color='k', linestyle='--', lw=0.8
  )
  ax_left.set_xlim(np.min(d_left), np.max(d_left))
  ax_left.set_ylim(np.min(a_left), np.max(a_left))

  ax_left.set_ylabel(r'$\alpha$')
  ax_left.text(
      0.03,
      0.90,
      f'({chr(97 + idx)}) $\sigma = {sigma}$',
      transform=ax_left.transAxes,
      fontsize=9,
      fontweight='bold',
      bbox=dict(
          boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'
      ),
  )

  # Load Right Column Data (Zoomed-in Region)
  d_right, a_right, base_config_SI, _, ml_frac_map = load_and_process_layer(
      file_zoom, threshold_idx
  )
  D_right, A_right = np.meshgrid(d_right, a_right)

  # Plot Right Column: Mode-locked fraction
  im_right = ax_right.pcolormesh(
      D_right,
      A_right,
      ml_frac_map,
      cmap=cmap_continuous,
      vmin=0.0,
      vmax=1.0,
      shading='auto',
      rasterized=True,
  )

  ax_right.plot(
      d_right, upper_boundary(15, d_right), color='white', linestyle='-', lw=0.8
  )
  ax_right.plot(
      d_right, lower_boundary(15, d_right), color='white', linestyle='-', lw=0.8
  )
  ax_right.plot(
      d_right, lasing_threshold(config, d_right, return_all=False), color='white', linestyle='--', lw=0.8
  )

  ax_right.set_ylabel(r'$\alpha$')
  ax_right.text(
      0.03,
      0.90,
      f'({chr(100 + idx)}) $\sigma = {sigma}$',
      transform=ax_right.transAxes,
      fontsize=9,
      fontweight='bold',
      color='white',
      bbox=dict(
          boxstyle='round,pad=0.2', facecolor='black', alpha=0.5, edgecolor='none'
      ),
  )

  ax_right.set_xlim(-3.5, np.max(d_right))
  ax_right.set_ylim(np.min(a_right), 0.5)

  # Highlight zoomed boundary region on the left overview panel
  x_min, x_max = np.min(d_right), np.max(d_right)
  y_min, y_max = np.min(a_right), np.max(a_right)

  rect = plt.Rectangle(
      (x_min, y_min),
      x_max - x_min,
      y_max - y_min,
      linewidth=1.0,
      edgecolor='black',
      facecolor='none',
      linestyle='--',
  )
  ax_left.add_patch(rect)

  # Add connection lines connecting the inset box corners to the zoom plot
  # Top connector (Top-Right of inset box -> Top-Left of zoom plot)
  '''
  con_top = ConnectionPatch(
      xyA=(x_max, y_max),
      coordsA=ax_left.transData,
      xyB=(0, np.max(a_left)),
      coordsB=ax_right.transAxes,
      color='black',
      linestyle=':',
      linewidth=0.8,
  )
  fig.add_artist(con_top)

  # Bottom connector (Bottom-Right of inset box -> Bottom-Left of zoom plot)
  con_bottom = ConnectionPatch(
      xyA=(x_max, y_min),
      coordsA=ax_left.transData,
      xyB=(0, 0),
      coordsB=ax_right.transAxes,
      color='black',
      linestyle=':',
      linewidth=0.8,
  )
  fig.add_artist(con_bottom)'''

# Axis formatting
axes[2, 0].set_xlabel(r'$\delta$')
axes[2, 1].set_xlabel(r'$\delta$')

#plt.tight_layout()

# Adjust layout spacing to make room for horizontal colorbars
plt.subplots_adjust(
    left=0.08, right=0.92, top=0.96, bottom=0.14, wspace=0.08, hspace=0.08
)

# ------------------------------------------------------------
# Bottom Horizontal Colorbars
# ------------------------------------------------------------

# 1. Colorbar for Left Column (Discrete Classification)
pos_left = axes[2, 0].get_position()
cax_left = fig.add_axes([pos_left.x0, 0.05, pos_left.width, 0.02])
cbar_left = fig.colorbar(
    im_left, cax=cax_left, orientation='horizontal', ticks=[0, 1, 2, 3]
)
cbar_left.ax.set_xticklabels(categories, fontsize=6.5)
cbar_left.ax.tick_params(size=0)

# 2. Colorbar for Right Column (Continuous Mode-locked Fraction)
pos_right = axes[2, 1].get_position()
cax_right = fig.add_axes([pos_right.x0, 0.05, pos_right.width, 0.02])
cbar_right = fig.colorbar(im_right, cax=cax_right, orientation='horizontal')
cbar_right.set_label(
    r'Mode-locked time fraction ($R_{\rm HPL} > 0.80$)', fontsize=7.5
)
cbar_right.ax.tick_params(labelsize=6.5)

plt.savefig('classification_and_modelocking_grid.pdf', dpi=300)
plt.show()