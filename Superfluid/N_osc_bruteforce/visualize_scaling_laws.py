import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from equations import upper_boundary, lower_boundary

def analyze_intermittency_sweep(filepath="bruteforce_sweep_results4_N=15.npz", T_sim=1000, T_eval=8000.0):
    """
    Loads single-sigma sweep results and generates a 3-panel plot 
    in the (alpha, delta) plane with interactive threshold selection.
    """
    # 1. LOAD DATA ARCHIVE
    try:
        data = np.load(filepath, allow_pickle=True)
        sigmas = data['sigmas_axis']
        alphas = data['alphas_axis']
        deltas = data['deltas_axis']
        classifications = data['classifications']
        ranges_list = data['ranges_list']
        thresholds = data['thresholds'] if 'thresholds' in data else np.array([0.80, 0.85, 0.90, 0.95])
        R_ML_mean = data['R_ML_mean']
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
        return

    n_alphas, n_deltas = len(alphas), len(deltas)
    n_thresholds = len(thresholds)
    n_grid_pts = n_alphas * n_deltas  # Assuming 1 sigma value

    # Reshape arrays to 2D (alpha, delta)
    grid_class = classifications.reshape((n_alphas, n_deltas))
    grid_ml_mean = np.asarray(R_ML_mean.reshape((n_alphas, n_deltas)), dtype=np.float64)
    grid_ml_mean[grid_class != 'CHAOTIC'] = np.nan

    # Reshape ranges_list safely
    if ranges_list.size == n_grid_pts * n_thresholds:
        ranges_grid = ranges_list.reshape((n_alphas, n_deltas, n_thresholds))
        is_4d = True
    else:
        ranges_grid = ranges_list.reshape((n_alphas, n_deltas))
        is_4d = False

    # 2. PRE-COMPUTE METRICS ACROSS THRESHOLDS
    grid_mean_dt = np.full((n_thresholds, n_alphas, n_deltas), np.nan)
    grid_fraction_ml = np.zeros((n_thresholds, n_alphas, n_deltas))

    print("Pre-computing intermittency statistics across (alpha, delta) grid...")
    for th_idx in range(n_thresholds):
        for j in range(n_alphas):
            for k in range(n_deltas):
                if grid_class[j, k] != 'CHAOTIC':
                    grid_mean_dt[th_idx, j, k] = np.nan
                    grid_fraction_ml[th_idx, j, k] = 0.0
                else:
                    intervals = ranges_grid[j, k, th_idx] if is_4d else ranges_grid[j, k][th_idx]
                    
                    if intervals is not None and len(intervals) > 0:
                        starts = []
                        stops = []
                        for r in intervals:
                            starts.append(r[0])
                            stops.append(T_eval+T_sim if (r[1] is None or np.isnan(r[1])) else r[1])

                        # Metric 1: Mean time between burst starts
                        if len(starts) >= 2:
                            grid_mean_dt[th_idx, j, k] = np.mean(np.diff(starts))

                        # Metric 2: Fraction of simulation time spent mode-locked
                        total_duration = sum(max(0.0, sp - st) for st, sp in zip(starts, stops))
                        grid_fraction_ml[th_idx, j, k] = min(total_duration / T_eval, 1.0)

    # 3. PLOTTING SETUP
    delta_grid, alpha_grid = np.meshgrid(deltas, alphas)

    # Analytical boundaries
    try:
        up_bound = upper_boundary(15, deltas)
        low_bound = lower_boundary(15, deltas)
        has_boundaries = True
    except Exception:
        has_boundaries = False

    # Colormaps
    max_dt = np.nanmax(grid_mean_dt) if not np.all(np.isnan(grid_mean_dt)) else 10.0
    cmap_dt = plt.get_cmap('plasma').copy()
    cmap_dt.set_bad(color='lightgrey')

    cmap_frac = plt.get_cmap('viridis').copy()
    cmap_frac.set_bad(color='lightgrey')

    cmap_ml = plt.get_cmap('magma').copy()
    cmap_ml.set_bad(color='lightgrey')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(bottom=0.22, top=0.88, left=0.06, right=0.98, wspace=0.28)

    curr_th = [1]  # Default to threshold index 1 (0.85)

    def draw_plots():
        th = curr_th[0]
        
        # Plot 1: Mean Inter-Burst Time
        axes[0].clear()
        pcm0 = axes[0].pcolormesh(delta_grid, alpha_grid, grid_mean_dt[th], 
                                  cmap=cmap_dt, vmin=0, vmax=max_dt, shading='nearest')
        axes[0].set_title(f"$\langle \Delta T \\rangle$ | Thresh: {thresholds[th]:.2f}")
        axes[0].set_xlabel("$\delta$")
        axes[0].set_ylabel("\\alpha")

        # Plot 2: Mode-Locked Time Fraction
        axes[1].clear()
        pcm1 = axes[1].pcolormesh(delta_grid, alpha_grid, grid_fraction_ml[th], 
                                  cmap=cmap_frac, vmin=0, vmax=1.0, shading='nearest')
        axes[1].set_title(f"$F_{{\\text{{ML}}}}$ | Thresh: {thresholds[th]:.2f}")
        axes[1].set_xlabel("$\delta$")
        axes[1].set_ylabel("\\alpha")

        # Plot 3: Amplitude-Weighted Order Parameter
        axes[2].clear()
        pcm2 = axes[2].pcolormesh(delta_grid, alpha_grid, grid_ml_mean, 
                                  cmap=cmap_ml, vmin=0, vmax=1.0, shading='nearest')
        axes[2].set_title("$\langle R_{\\text{ML}} \\rangle$ (Mean Order Parameter)")
        axes[2].set_xlabel("$\delta$")
        axes[2].set_ylabel("\\alpha")

        # Overlay boundaries
        if has_boundaries:
            for ax in axes:
                ax.plot(deltas, up_bound, c='k', lw=1.2, ls='--')
                ax.plot(deltas, low_bound, c='k', lw=1.2, ls='--')

        return pcm0, pcm1, pcm2

    pcm0, pcm1, pcm2 = draw_plots()

    # Colorbars
    cbar0 = fig.colorbar(pcm0, ax=axes[0], pad=0.02)
    cbar0.set_label("Mean Inter-Burst Time $\langle \Delta T \\rangle$")
    
    cbar1 = fig.colorbar(pcm1, ax=axes[1], pad=0.02)
    cbar1.set_label("Mode-Locked Fraction $F_{\\text{ML}}$")
    
    cbar2 = fig.colorbar(pcm2, ax=axes[2], pad=0.02)
    cbar2.set_label("Mean Order Parameter $\langle R_{\\text{ML}} \\rangle$")

    # Interactive Threshold Control
    ax_thresh = fig.add_axes([0.38, 0.03, 0.24, 0.10])
    radio_thresh = RadioButtons(ax_thresh, [f'Thresh: {th:.2f}' for th in thresholds], active=curr_th[0])

    def update_thresh(label):
        curr_th[0] = [f'Thresh: {th:.2f}' for th in thresholds].index(label)
        draw_plots()
        fig.canvas.draw_idle()

    radio_thresh.on_clicked(update_thresh)

    plt.suptitle(f"Intermittency & Mode-Locking ($\sigma = {sigmas[0]:.2f}$)", fontsize=14, fontweight='bold')
    plt.show()

if __name__ == '__main__':
    analyze_intermittency_sweep()