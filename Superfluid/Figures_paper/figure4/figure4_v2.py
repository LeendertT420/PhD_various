import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

from bruteforce.equations import *

# =====================================================================
# 1. HELPER FUNCTION FOR MODE-LOCKED FRACTION
# =====================================================================
def calculate_mode_locked_fraction(
    ranges_list_threshold, classification, T_sim, T_eval
):
    """Calculates the total time fraction spent in mode-locked intervals."""
    if classification == 'BELOW THRESHOLD':
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


# =====================================================================
# 2. MAIN PLOTTING SCRIPT (2x2 Grid for 4 Sigma Values)
# =====================================================================
def create_phase_diagram_subplots(output_prefix='bruteforce_sweep_results', target_sigmas=[30, 40, 50, 60]):
    pattern = f"./data/{output_prefix}_sigma_*_N=*.npz"
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    def extract_sigma(filename):
        match = re.search(r'sigma_([0-9.]+)_N=', filename)
        return float(match.group(1)) if match else 0.0

    selected_files = []
    for sig in target_sigmas:
        matching = [f for f in files if np.isclose(extract_sigma(f), sig, atol=1e-2)]
        if matching:
            selected_files.append(matching[0])

    if not selected_files:
        selected_files = sorted(files, key=extract_sigma)[:4]

    num_panels = len(selected_files)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    viridis = plt.get_cmap('viridis')
    n_bins = 256
    viridis_colors = viridis(np.linspace(0, 1, n_bins))
    
    custom_colors = [
        'white',   # Below threshold
        'orange',  # Single mode
        'red',     # Multi mode
    ]
    combined_colors = custom_colors + list(viridis_colors)
    
    bounds = [0, 1, 2, 3] + list(np.linspace(3, 4, n_bins + 1)[1:])
    cmap = LinearSegmentedColormap.from_list("hybrid_cmap", combined_colors, N=len(bounds)-1)
    norm = BoundaryNorm(bounds, cmap.N)

    for idx, filename in enumerate(selected_files):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        data = np.load(filename, allow_pickle=True)
        
        coords = data['coords']
        classifications = data['classifications']
        ranges_list_all = data['ranges_list']
        T_sim = float(data['T_sim'])
        T_eval = float(data['T_eval'])
        sigma_val = float(data['sigma'])
        
        alphas_axis = data['alphas_axis']
        deltas_axis = data['deltas_axis']
        
        n_alpha = len(alphas_axis)
        n_delta = len(deltas_axis)

        grid_data = np.zeros((n_alpha, n_delta))
        
        for i, (v1, v2, v3) in enumerate(coords):
            alpha_idx = np.abs(alphas_axis - v2).argmin()
            delta_idx = np.abs(deltas_axis - v3).argmin()
            
            cls = classifications[i]
            
            thresh_idx = 3
            ranges_thresh = None
            if ranges_list_all[i] is not None and len(ranges_list_all[i]) > thresh_idx:
                ranges_thresh = ranges_list_all[i][thresh_idx]
            
            
            
            ml_fraction = calculate_mode_locked_fraction(
                ranges_thresh, cls, 1500, 8000
            )

            if v1 == 40 and v2> 0.25 and v2<0.3 and v3>-2.8 and v3<-2.6 and cls != 'BELOW THRESHOLD':
                print(coords[i], cls, ranges_thresh, ml_fraction, T_eval, T_sim, '\n\n')
            
            if ml_fraction > 0:
                grid_data[alpha_idx, delta_idx] = 3.0 + ml_fraction
            elif cls == 'BELOW THRESHOLD':
                grid_data[alpha_idx, delta_idx] = 0.5
            elif cls == 'SINGLE MODE LASING':
                grid_data[alpha_idx, delta_idx] = 1.5
            elif cls == 'MULTI MODE LASING':
                grid_data[alpha_idx, delta_idx] = 2.5
            elif cls == 'CHAOTIC':
                grid_data[alpha_idx, delta_idx] = 3.0
            else:
                grid_data[alpha_idx, delta_idx] = 0.5

        ax.imshow(
            grid_data, 
            origin='lower', 
            extent=[deltas_axis[0], deltas_axis[-1], alphas_axis[0], alphas_axis[-1]], 
            aspect='auto',
            cmap=cmap,
            norm=norm,
            interpolation='nearest'  # Turns off smoothing/interpolation
        )
        
        ax.set_title(f"$\\sigma$ = {sigma_val:.1f}")
        ax.set_xlabel("$\\delta$ (Delta)")
        ax.set_ylabel("$\\alpha$ (Alpha)")

        ax.plot(deltas_axis, upper_boundary(15, deltas_axis), 'k--', linewidth=1.5, label='bif Boundary')
        ax.plot(deltas_axis, lower_boundary(15, deltas_axis), 'k--', linewidth=1.5)

        ax.set_xlim(deltas_axis[0], deltas_axis[-1])
        ax.set_ylim(alphas_axis[0], alphas_axis[-1])

        config_SI = data['base_config_SI'].item()
        config_SI['d'] = to_SI({'sigma': sigma_val})['d']
        config = to_unitless(config_SI)


    for idx in range(num_panels, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    
    cbar.set_ticks([0.5, 1.5, 2.5, 3.0, 3.5, 4.0])
    cbar.set_ticklabels([
        'Below Threshold', 
        'Single Mode', 
        'Multi-Mode', 
        'Chaos (0.0)', 
        'Chaos (0.5)', 
        'Chaos (1.0)'
    ])
    cbar.set_label('Regime / Mode-Locked Fraction')

    fig.subplots_adjust(right=0.9, wspace=0.2, hspace=0.25)
    plt.savefig("phase_diagram_sigma_sweep_2x2.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    create_phase_diagram_subplots(target_sigmas=[30, 40, 50, 60])