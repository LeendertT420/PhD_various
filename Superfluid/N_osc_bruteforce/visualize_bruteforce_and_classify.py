import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap, BoundaryNorm
from equations import *

def load_and_plot_sweep(filepath="bruteforce_sweep_results_N=15.npz"):
    # 1. LOAD COMPRESSED DATA MATRIX
    try:
        data = np.load(filepath, allow_pickle=True)
        sigmas = data['sigmas_axis']
        alphas = data['alphas_axis']
        deltas = data['deltas_axis']
        peaks_freqs = data['peaks_freqs']
        peaks_amps = data['peaks_amps']
        classifications = data['classifications']
        base_config_SI = data['base_config_SI'].item()
        N = data['N']
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Run the simulation pipeline first.")
        return

    # 2. RECONSTRUCT 3D GRID FRAMEWORK FROM FLATTENED ARRAY
    grid_shape = (len(sigmas), len(alphas), len(deltas))
    grid_class = classifications.reshape(grid_shape)
    grid_freqs = peaks_freqs.reshape(grid_shape)
    grid_amps = peaks_amps.reshape(grid_shape)
    grid_maxfreq = np.zeros(grid_shape)
    
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                cls = grid_class[i, j, k]
                if cls == 'CHAOTIC':
                    grid_maxfreq[i, j, k] = -4
                elif cls == 'NOT CLASSIFIED':
                    grid_maxfreq[i, j, k] = -1
                elif cls == 'MODE LOCKED':
                    grid_maxfreq[i, j, k] = -2
                elif cls == 'BELOW THRESHOLD':
                    grid_maxfreq[i, j, k] = -3
                elif len(grid_amps[i, j, k]) > 0:
                    grid_maxfreq[i, j, k] = grid_freqs[i, j, k][np.argmax(grid_amps[i, j, k])]
                else:
                    grid_maxfreq[i, j, k] = -1

    # 3. DEFINE AND ALIGN COLOR MAPPINGS
    unique_classes = np.unique(classifications)
    class_to_int = {cls: idx for idx, cls in enumerate(unique_classes)}
    grid_int = np.vectorize(class_to_int.get)(grid_class)

    # Core explicit color rules
    state_colors = {
        'BELOW THRESHOLD': 'black',
        'MODE LOCKED': 'orange',
        'CHAOTIC': 'white'
    }

    # Generate custom colors array for original qualitative plot
    num_classes = len(unique_classes)
    base_cmap = plt.get_cmap('tab10', num_classes)
    plot1_colors = []
    
    for cls in unique_classes:
        if cls in state_colors:
            plot1_colors.append(state_colors[cls])
        else:
            plot1_colors.append(base_cmap.colors[class_to_int[cls]])
            
    custom_cmap = ListedColormap(plot1_colors)
    bounds_discrete = np.arange(num_classes + 1) - 0.5
    norm_discrete = BoundaryNorm(bounds_discrete, num_classes)

    # 4. CONSTRUCT HYBRID COLORMAP FOR PLOT 2
    code_mapping = {
        -4: 'CHAOTIC',
        -3: 'BELOW THRESHOLD',
        -2: 'MODE LOCKED',
        -1: 'NOT CLASSIFIED'
    }
    
    disc_colors = []
    for code in [-4, -3, -2, -1]:
        cls_name = code_mapping[code]
        if cls_name in state_colors:
            disc_colors.append(state_colors[cls_name])
        elif cls_name in class_to_int:
            disc_colors.append(base_cmap.colors[class_to_int[cls_name]])
        else:
            disc_colors.append((0.5, 0.5, 0.5)) # Fallback

    # Generate boundaries for Plot 2: 4 discrete labels + segmented continuous bins
    max_freq_val = max(np.max(grid_maxfreq), 1e-5)
    num_freq_bins = 50
    freq_bins = np.linspace(0.0, max_freq_val, num_freq_bins + 1)
    
    hybrid_bounds = np.concatenate([[-4.5, -3.5, -2.5, -1.5], freq_bins])
    
    continuous_cmap = plt.get_cmap('viridis')
    freq_colors = [continuous_cmap(i / num_freq_bins) for i in range(num_freq_bins)]
    hybrid_colors = disc_colors + freq_colors
    
    freq_cmap = ListedColormap(hybrid_colors)
    norm_hybrid = BoundaryNorm(hybrid_bounds, len(hybrid_colors))

    # 5. INITIALIZE INTERACTIVE MATPLOTLIB SIDE-BY-SIDE CANVASES
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.25, wspace=0.25, right=0.88)

    def get_slices(sigma_idx):
        slice_discrete = np.fliplr(grid_int[sigma_idx, :, :])
        slice_freq = np.fliplr(grid_maxfreq[sigma_idx, :, :])
        return slice_discrete, slice_freq

    initial_idx = 0
    init_disc, init_freq = get_slices(initial_idx)
    extent_val = [deltas.min(), deltas.max(), alphas.min(), alphas.max()]
    
    # --- PLOT 1: Discrete Classification ---
    im1 = ax1.imshow(
        init_disc, extent=extent_val, origin='lower',
        cmap=custom_cmap, norm=norm_discrete, aspect='auto'
    )
    ax1.set_xlabel(r'$\delta$', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'$\alpha$', fontsize=11, fontweight='bold')
    ax1.set_title('Classification Map', fontsize=12, pad=10)
    line1_u, = ax1.plot(deltas, upper_boundary(N, deltas), c='k')
    line1_l, = ax1.plot(deltas, lower_boundary(N, deltas), c='k')
    lasing_line1, = ax1.plot(deltas, np.zeros_like(deltas), 'r')

    # --- PLOT 2: Frequency Mapping ---
    im2 = ax2.imshow(
        init_freq, extent=extent_val, origin='lower',
        cmap=freq_cmap, norm=norm_hybrid, aspect='auto'
    )
    ax2.set_xlabel(r'$\delta$', fontsize=11, fontweight='bold')
    ax2.set_title('Dominant Frequency Map', fontsize=12, pad=10)
    line2_u, = ax2.plot(deltas, upper_boundary(N, deltas), c='k')
    line2_l, = ax2.plot(deltas, lower_boundary(N, deltas), c='k')
    lasing_line2, = ax2.plot(deltas, np.zeros_like(deltas), 'r')

    ax1.set_ylim(alphas.min(), alphas.max())
    ax1.set_xlim(deltas.min(), deltas.max())

    suptitle_text = fig.suptitle(f'Phase Map Projections | Sigma = {sigmas[initial_idx]:.2f}', fontsize=14, y=0.96)

    # 6. COLORBAR ATTACHMENTS WITH CORRECT DISCRETE LABELS
    cax1 = fig.add_axes([0.43, 0.25, 0.015, 0.58])
    cbar1 = fig.colorbar(im1, cax=cax1, ticks=np.arange(num_classes))
    cbar1.ax.set_yticklabels(unique_classes, fontsize=8)
    cbar1.ax.tick_params(length=0)

    cax2 = fig.add_axes([0.91, 0.25, 0.015, 0.58])
    cbar2 = fig.colorbar(im2, cax=cax2)
    
    discrete_ticks = [-4.0, -3.0, -2.0, -0.75]
    continuous_ticks = np.linspace(0.0, max_freq_val, 5).tolist()
    
    cbar2.set_ticks(discrete_ticks + continuous_ticks)
    tick_labels = ['CHAOTIC', 'BELOW THRESHOLD', 'MODE LOCKED', 'NOT CLASSIFIED'] + [f"{val:.2f}" for val in continuous_ticks]
    cbar2.ax.set_yticklabels(tick_labels, fontsize=8)

    # 7. CONSTRUCT THE INTERACTIVE SIGMA SLIDER
    ax_slider = plt.axes([0.20, 0.08, 0.60, 0.04])
    sigma_slider = Slider(
        ax=ax_slider, label='Sigma Value ',
        valmin=sigmas.min(), valmax=sigmas.max(), valinit=sigmas[0],
        valfmt='%.2f', valstep=sigmas, color='steelblue'
    )

    # 8. LIVE SLIDER UPDATE CALLBACK LOGIC
    def update(val):
        sigma_idx = np.argmin(np.abs(sigmas - val))
        sigma = sigmas[sigma_idx]

        base_config_SI['d'] = to_SI({'sigma': sigma})['d']
        config = to_unitless(base_config_SI)
        
        updated_disc, updated_freq = get_slices(sigma_idx)
        
        im1.set_data(updated_disc)
        im2.set_data(updated_freq)
        
        threshold_data = lasing_threshold(config, deltas, return_all=False)
        lasing_line1.set_ydata(threshold_data)
        lasing_line2.set_ydata(threshold_data)
        
        suptitle_text.set_text(f'Phase Map Projections | Sigma = {sigma:.2f}')
        fig.canvas.draw_idle()

    sigma_slider.on_changed(update)
    plt.show()
    return sigma_slider

if __name__ == "__main__":
    slider_ref = load_and_plot_sweep()