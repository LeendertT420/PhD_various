import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap, BoundaryNorm
from equations import *
from scipy.integrate import solve_ivp

def simulate_and_compute_fft(sigma, alpha, delta, N, base_config_SI):
    """
    HOOK FUNCTION: Insert your high-dimensional non-linear differential equation 
    solver and Fourier Transform logic here.
    
    Returns:
        t (array): Time vector
        y (array): Time trace values to plot (e.g., intensity, field amplitude)
        freqs (array): Frequency vector for the spectrum
        spectrum (array): Fourier transform magnitudes
    """
    params = to_unitless(base_config_SI)
    params['sigma'] = sigma
    params['alpha'] = alpha
    params['delta'] = delta

    X0 = np.concatenate([np.full(N,0), np.full(N, 0), [0]])
    T = 800
    t = np.linspace(0, T, int(T*50))

    sol = solve_ivp(
            lambda t, X: system_numba(t, X, params),
            (0, T), X0, t_eval=t, method="RK45", rtol=1e-6, atol=1e-9
        )

    X = np.sum(sol.y[:N, :], axis=0)

    fft_values = np.fft.rfft(X-np.mean(X))
    frequencies = np.fft.rfftfreq(len(X), d=1/50)

    amplitude = 2.0 * np.abs(fft_values) / len(X)
    amplitude[0] /= 2.0
    if len(X) % 2 == 0:
        amplitude[-1] /= 2.0

    # -------------------------------------------------------------------------
    
    return t, X, frequencies, amplitude


def load_and_plot_sweep(filepath="bruteforce_sweep_results_N=15.npz"):
    # 1. LOAD COMPRESSED DATA MATRIX
    try:
        data = np.load(filepath, allow_pickle=True)
        sigmas = data['sigmas_axis']
        alphas = data['alphas_axis']
        deltas = data['deltas_axis']
        peaks_freqs = data['peaks_freqs']
        peaks_amps = data['peaks_amps']
        classifications = data['classifications'].astype(object)
        base_config_SI = data['base_config_SI'].item()
        N = base_config_SI['N']
        base_config_SI['chi_ijk'] = np.load('./tensors/chi_ijk.npy')[:N, :N, :N]
        base_config_SI['chi_ijkl'] = np.load('./tensors/chi_ijkl.npy')[:N, :N, :N]
        base_config_SI['xi'] = np.ones(N)
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
                    grid_maxfreq[i, j, k] = -3
                elif cls == 'BELOW THRESHOLD':
                    grid_maxfreq[i, j, k] = -2
                elif cls == 'MODE LOCKED':
                    grid_maxfreq[i, j, k] = -1
                elif len(grid_amps[i, j, k]) > 0:
                    grid_maxfreq[i, j, k] = grid_freqs[i, j, k][np.argmax(grid_amps[i, j, k])]
                else:
                    grid_maxfreq[i, j, k] = -3

    # 3. DEFINE AND ALIGN COLOR MAPPINGS
    unique_classes = np.unique(grid_class)
    unique_classes = np.array([c for c in unique_classes if c != 'NOT CLASSIFIED'])
    class_to_int = {cls: idx for idx, cls in enumerate(unique_classes)}
    grid_int = np.vectorize(lambda x: class_to_int.get(x, 0))(grid_class)

    state_colors = {
        'SINGLE MODE LASING': 'lightblue',
        'MULTI MODE LASING': 'darkblue',
        'CHAOTIC': 'white',
        'BELOW THRESHOLD': 'black',
        'MODE LOCKED': 'magenta'
    }

    num_classes = len(unique_classes)
    base_cmap = plt.get_cmap('tab10', num_classes)
    plot1_colors = [state_colors.get(cls, base_cmap.colors[class_to_int[cls] % num_classes]) for cls in unique_classes]
    custom_cmap = ListedColormap(plot1_colors)
    bounds_discrete = np.arange(num_classes + 1) - 0.5
    norm_discrete = BoundaryNorm(bounds_discrete, num_classes)

    # 4. CONSTRUCT HYBRID COLORMAP FOR PLOT 2
    code_mapping = {-3: 'CHAOTIC', -2: 'BELOW THRESHOLD', -1: 'MODE LOCKED'}
    disc_colors = [state_colors.get(code_mapping[code], (0.5, 0.5, 0.5)) for code in [-3, -2, -1]]

    max_freq_val = max(np.max(grid_maxfreq), 1e-5)
    num_freq_bins = 50
    freq_bins = np.linspace(0.0, max_freq_val, num_freq_bins + 1)
    hybrid_bounds = np.concatenate([[-3.5, -2.5, -1.5], freq_bins])
    
    continuous_cmap = plt.get_cmap('viridis')
    freq_colors = [continuous_cmap(i / num_freq_bins) for i in range(num_freq_bins)]
    hybrid_colors = disc_colors + freq_colors
    
    freq_cmap = ListedColormap(hybrid_colors)
    norm_hybrid = BoundaryNorm(hybrid_bounds, len(hybrid_colors))

    # 5. INITIALIZE INTERACTIVE 4-PANEL LAYOUT
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], bottom=0.22, top=0.92, left=0.08, right=0.88, hspace=0.35, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    def get_slices(sigma_idx):
        return np.fliplr(grid_int[sigma_idx, :, :]), np.fliplr(grid_maxfreq[sigma_idx, :, :])

    initial_sigma_idx = 0
    init_disc, init_freq = get_slices(initial_sigma_idx)
    extent_val = [deltas.min(), deltas.max(), alphas.min(), alphas.max()]
    
    # Render maps
    im1 = ax1.imshow(init_disc, extent=extent_val, origin='lower', cmap=custom_cmap, norm=norm_discrete, aspect='auto')
    ax1.set_ylabel(r'$\alpha$', fontsize=11, fontweight='bold')
    ax1.set_title('Classification Map', fontsize=12, pad=10)
    line1_u, = ax1.plot(deltas, upper_boundary(N, deltas), c='k')
    line1_l, = ax1.plot(deltas, lower_boundary(N, deltas), c='k')
    lasing_line1, = ax1.plot(deltas, np.zeros_like(deltas), 'r')

    im2 = ax2.imshow(init_freq, extent=extent_val, origin='lower', cmap=freq_cmap, norm=norm_hybrid, aspect='auto')
    ax2.set_title('Dominant Frequency Map', fontsize=12, pad=10)
    line2_u, = ax2.plot(deltas, upper_boundary(N, deltas), c='k')
    line2_l, = ax2.plot(deltas, lower_boundary(N, deltas), c='k')
    lasing_line2, = ax2.plot(deltas, np.zeros_like(deltas), 'r')

    ax1.set_ylim(alphas.min(), alphas.max())
    ax1.set_xlim(deltas.min(), deltas.max())

    # Interactive coordinates marker points
    current_alpha = (alphas.min() + alphas.max()) / 2.0
    current_delta = (deltas.min() + deltas.max()) / 2.0
    cross1, = ax1.plot(current_delta, current_alpha, 'rx', markersize=10, mew=2)
    cross2, = ax2.plot(current_delta, current_alpha, 'rx', markersize=10, mew=2)

    # Initialize 1D data line structures
    t_dummy, y_dummy, f_dummy, s_dummy = simulate_and_compute_fft(sigmas[initial_sigma_idx], current_alpha, current_delta, N, base_config_SI)
    trace_line, = ax3.plot(t_dummy, y_dummy, color='blue', lw=1.5)
    ax3.set_title('Time Domain Trace', fontsize=11, pad=8)
    ax3.set_xlabel('Time ($t$)', fontsize=10)
    ax3.set_ylabel('Amplitude', fontsize=10)
    
    fft_line, = ax4.plot(f_dummy, s_dummy, color='red', lw=1.5)
    ax4.set_title('Power Spectrum (FFT)', fontsize=11, pad=8)
    ax4.set_xlabel('Frequency ($\omega$)', fontsize=10)
    ax4.set_ylabel('Magnitude', fontsize=10)

    suptitle_text = fig.suptitle(f'Interactive Phase Space Dynamics | Sigma = {sigmas[initial_sigma_idx]:.2f}', fontsize=14, y=0.97)

    # Colorbars
    cax1 = fig.add_axes([0.44, 0.58, 0.015, 0.34])
    cbar1 = fig.colorbar(im1, cax=cax1, ticks=np.arange(num_classes))
    cbar1.ax.set_yticklabels(unique_classes, fontsize=8)
    cbar1.ax.tick_params(length=0)

    cax2 = fig.add_axes([0.91, 0.58, 0.015, 0.34])
    cbar2 = fig.colorbar(im2, cax=cax2)
    discrete_ticks = [-3.0, -2.0, -0.75]
    continuous_ticks = np.linspace(0.0, max_freq_val, 5).tolist()
    cbar2.set_ticks(discrete_ticks + continuous_ticks)
    cbar2.ax.set_yticklabels(['CHAOTIC', 'BELOW THRESHOLD', 'MODE LOCKED'] + [f"{val:.2f}" for val in continuous_ticks], fontsize=8)

    # Sliders axes definitions
    ax_slider_sigma = plt.axes([0.15, 0.12, 0.65, 0.025])
    ax_slider_alpha = plt.axes([0.15, 0.07, 0.65, 0.025])
    ax_slider_delta = plt.axes([0.15, 0.02, 0.65, 0.025])

    slider_sigma = Slider(ax=ax_slider_sigma, label='Sigma ', valmin=sigmas.min(), valmax=sigmas.max(), valinit=sigmas[0], valfmt='%.2f', valstep=sigmas, color='steelblue')
    slider_alpha = Slider(ax=ax_slider_alpha, label='Alpha ', valmin=alphas.min(), valmax=alphas.max(), valinit=current_alpha, valfmt='%.3f', color='darkorange')
    slider_delta = Slider(ax=ax_slider_delta, label='Delta ', valmin=deltas.min(), valmax=deltas.max(), valinit=current_delta, valfmt='%.3f', color='crimson')

    # Global master frame synchronization update routine
    def update_plots(change_sigma=False):
        sigma = slider_sigma.val
        alpha = slider_alpha.val
        delta = slider_delta.val

        # Synchronize indicators
        cross1.set_data([delta], [alpha])
        cross2.set_data([delta], [alpha])

        if change_sigma:
            sigma_idx = np.argmin(np.abs(sigmas - sigma))
            base_config_SI['d'] = to_SI({'sigma': sigma})['d']
            config = to_unitless(base_config_SI)
            
            updated_disc, updated_freq = get_slices(sigma_idx)
            im1.set_data(updated_disc)
            im2.set_data(updated_freq)
            
            threshold_data = lasing_threshold(config, deltas, return_all=False)
            lasing_line1.set_ydata(threshold_data)
            lasing_line2.set_ydata(threshold_data)
            suptitle_text.set_text(f'Interactive Phase Space Dynamics | Sigma = {sigma:.2f}')

        # Reconstruct 1D trajectories using hook function
        t, y, freqs, spectrum = simulate_and_compute_fft(sigma, alpha, delta, N, base_config_SI)
        
        trace_line.set_data(t, y)
        ax3.set_xlim(t[-1000], t[-1])
        
        fft_line.set_data(freqs, spectrum)
        ax4.set_xlim(0, 2.4)
        ax4.set_ylim(0, np.max(spectrum))

        fig.canvas.draw_idle()

    # Event callbacks
    def on_sigma_changed(val): update_plots(change_sigma=True)
    def on_parameter_changed(val): update_plots(change_sigma=False)
    
    slider_sigma.on_changed(on_sigma_changed)
    slider_alpha.on_changed(on_parameter_changed)
    slider_delta.on_changed(on_parameter_changed)

    # Click handling for top axes matrices
    def on_click(event):
        if event.inaxes in [ax1, ax2]:
            # Temporarily block standard change events to update markers cleanly
            slider_delta.eventson = False
            slider_alpha.eventson = False
            slider_delta.set_val(event.xdata)
            slider_alpha.set_val(event.ydata)
            slider_delta.eventson = True
            slider_alpha.eventson = True
            update_plots(change_sigma=False)

    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.show()
    return slider_sigma, slider_alpha, slider_delta

if __name__ == "__main__":
    sliders_ref = load_and_plot_sweep()