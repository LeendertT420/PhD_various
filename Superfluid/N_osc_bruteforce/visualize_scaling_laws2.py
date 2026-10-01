import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
import matplotlib.ticker as ticker

import numpy as np
from scipy.optimize import curve_fit


def power_law_shift_log(x, A, b, x_c):
    """
    Evaluates: ln(y) = ln(a) + b * ln(|x - x_c|)
    Fitting in log-space weights all orders of magnitude equally.
    """
    dist = (x - x_c)  # Avoid log(0)
    return A*dist**b


def fit_power_law(x_data, y_data, threshold=1):
    """
    Fits ln(y) = ln(a) + b * ln(|x - x_c|) in log-log space,
    then back-calculates a = exp(log_a), b, and x_c.
    """
    # Filter non-finite and non-positive data required for log(y)
    mask = (
        np.isfinite(x_data) & np.isfinite(y_data) & (y_data > 0) & (y_data < threshold)
    )
    x_valid = x_data[mask]
    y_valid = y_data[mask]

    if len(x_valid) < 4:
        return None  # Need at least 4 points to fit 3 parameters safely

    # Sort data for consistent output
    sort_idx = np.argsort(x_valid)
    x_valid = x_valid[sort_idx]
    y_valid = y_valid[sort_idx]

    x_min, x_max = x_valid[0], x_valid[-1]

    # Transform y to log-space
    
    x_c_init = -2.5
    b_init = -1.0
    A_init = 10000
    p0 = [A_init, b_init, x_c_init]

    # Bounds: [log_a, b, x_c]
    bounds = (
        [-np.inf, -1000, -5],  # Lower bounds
        [np.inf, -.01, -2],  # Upper bounds (keeps x_c < x_min)
    )

    try:
        popt, pcov = curve_fit(
            power_law_shift_log,
            x_valid,
            y_valid,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )

        if np.any(np.isinf(pcov)) or np.any(np.isnan(pcov)):
            return None

        A_fit, b_fit, x_c_fit = popt

        # --- BACK-CALCULATE PARAMETERS ---
        variance = np.mean(np.sqrt(np.diag(pcov)))

        # Generate smooth fitted curve back in linear space
        x_f = np.linspace(x_min, x_max, 200)
        y_f = power_law_shift_log(x_f, A_fit, b_fit, x_c_fit)

        return b_fit, A_fit, x_c_fit, variance, x_f, y_f

    except (RuntimeError, ValueError):
        return None


def plot_scaling_laws(filepath="bruteforce_sweep_results6_N=15.npz", T_eval=300.0):
    # 1. LOAD DATA ARCHIVE
    try:
        data = np.load(filepath, allow_pickle=True)
        sigmas = data['sigmas_axis']
        alphas = data['alphas_axis']
        deltas = data['deltas_axis']
        ranges_list = data['ranges_list']
        thresholds = data['thresholds'] if 'thresholds' in data else np.array([0.80, 0.85, 0.90, 0.95])
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
        return

    n_sigmas, n_alphas, n_deltas = len(sigmas), len(alphas), len(deltas)
    grid_shape = (n_sigmas, n_alphas, n_deltas)
    n_thresholds = len(thresholds)
    n_grid_pts = n_sigmas * n_alphas * n_deltas

    # Pre-allocate 4D arrays (sigmas, alphas, deltas, thresholds)
    grid_mean_dt = np.full((*grid_shape, n_thresholds), np.nan)
    grid_fraction_ml = np.zeros((*grid_shape, n_thresholds))
    grid_mean_dur = np.full((*grid_shape, n_thresholds), np.nan)

    # Reshape unrolled array safely
    if ranges_list.size == n_grid_pts * n_thresholds:
        ranges_grid = ranges_list.reshape((*grid_shape, n_thresholds))
        is_4d = True
    else:
        ranges_grid = ranges_list.reshape(grid_shape)
        is_4d = False

    print("Computing intermittency metrics...")
    for th_idx in range(n_thresholds):
        for i in range(n_sigmas):
            for j in range(n_alphas):
                for k in range(n_deltas):
   
                    if is_4d:
                        intervals = ranges_grid[i, j, k, th_idx]
                    else:
                        cell = ranges_grid[i, j, k]
                        intervals = cell[th_idx] if (cell is not None and len(cell) > th_idx) else None

                    if intervals is not None and len(intervals) > 0:
                        starts = np.array([r[0] for r in intervals])
                        stops = np.array([5000 if (r[1] is None or np.isnan(r[1])) else r[1] for r in intervals])
                        durations = stops-starts#starts[1:] - stops[:-1]

                        # Metric 1: Mean inter-burst time (requires >= 2 starts)
                        if len(starts) >= 2:
                            grid_mean_dt[i, j, k, th_idx] = np.mean(np.diff(starts))

                        # Metric 2: Total Mode-Locked fraction
                        grid_fraction_ml[i, j, k, th_idx] = np.sum(durations) / 4000

                        # Metric 3: Mean duration per mode-locked interval
                        grid_mean_dur[i, j, k, th_idx] = np.mean(durations)

    # 2. CONSTRUCT Matplotlib FIGURE & SUBPLOTS
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))
    plt.subplots_adjust(bottom=0.32, top=0.88, left=0.06, right=0.97, wspace=0.28)

    # State variables
    state = {
        'var_x': 'alpha',     # X-axis variable: 'sigma', 'alpha', or 'delta'
        'th_idx': 1,          # Default threshold: index 1 (0.85)
        'i_sigma': n_sigmas // 2,
        'j_alpha': n_alphas // 2,
        'k_delta': n_deltas // 2
    }

    def update_plots():
        def set_custom_log_xticks(ax, x_data, num_ticks=3):
            ticks = np.geomspace(np.min(x_data), np.max(x_data), num_ticks)
            
            # Set major ticks and labels
            ax.set_xticks(ticks, labels=[f"{v:.3g}" for v in ticks])
            
            # Disable minor ticks on the x-axis
            ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax1.clear()
        ax2.clear()
        ax3.clear()

        th = state['th_idx']
        i, j, k = state['i_sigma'], state['j_alpha'], state['k_delta']


        # Determine 1D Slice coordinates and data vectors based on active X-axis variable
        if state['var_x'] == 'sigma':
            x_vals = sigmas
            x_label = r"$\sigma$"
            y_dt = grid_mean_dt[:, j, k, th]
            y_frac = grid_fraction_ml[:, j, k, th]
            y_dur = grid_mean_dur[:, j, k, th]
            title_ctx = f"Fixed: $\\alpha={alphas[j]:.3f}$, $\\delta={deltas[k]:.2f}$"
        elif state['var_x'] == 'alpha':
            x_vals = alphas
            x_label = r"$\alpha$"
            y_dt = grid_mean_dt[i, :, k, th]
            y_frac = grid_fraction_ml[i, :, k, th]
            y_dur = grid_mean_dur[i, :, k, th]
            title_ctx = f"Fixed: $\sigma={sigmas[i]:.2f}$, $\\delta={deltas[k]:.2f}$"
        else:  # 'delta'
            x_vals = deltas
            x_label = r"$|\delta|$"
            y_dt = grid_mean_dt[i, j, :, th]
            y_frac = grid_fraction_ml[i, j, :, th]
            y_dur = grid_mean_dur[i, j, :, th]
            title_ctx = f"Fixed: $\sigma={sigmas[i]:.2f}$, $\\alpha={alphas[j]:.3f}$"

        # --- PANEL 1: Inter-Burst Time ---
        ax1.plot(x_vals, y_dt, 'o-', color='crimson', label='Data Points', linewidth=0, markersize=5)
        fit1 = fit_power_law(x_vals, y_dt)
        if fit1:
            gamma, A, x_c, r2, x_f, y_f = fit1
            ax1.plot(x_f, y_f, '--', color='black', label=f'Fit: $~ |x-{{{x_c:.2f}}}|^{{{gamma:.1f}}}$')
        #ax1.set_xscale('symlog', linthresh=0.01)
        #ax1.set_yscale('log')
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel(r"Mean Inter-Burst Time $\langle \Delta T \rangle$", fontsize=11)
        ax1.set_title(f"Inter-Burst Scaling\n({title_ctx})", fontsize=10)
        ax1.grid(True, which="major", ls=":", alpha=0.5)  # <--- Change 'both' to 'major'
        set_custom_log_xticks(ax1, x_vals, num_ticks=3)
        ax1.legend(loc='best', fontsize=9)

        # --- PANEL 2: Mode-Locked Time Fraction ---
        ax2.plot(x_vals, y_frac, 's-', color='teal', label='Data Points', linewidth=0, markersize=5)
        fit2 = fit_power_law(x_vals, y_frac, threshold=.8)
        if fit2:
            gamma, A, x_c, r2, x_f, y_f = fit2
            ax2.plot(x_f, y_f, '--', color='black', label=f'Fit: $~ |x-{{{x_c:.2f}}}|^{{{gamma:.1f}}}$')
        #ax2.set_xscale('symlog', linthresh=0.01)
        #ax2.set_yscale('log')
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_ylabel(r"Mode-Locked Fraction $F_{\text{ML}}$", fontsize=11)
        ax2.set_title(f"Mode-Locked Fraction\nThreshold = {thresholds[th]:.2f}", fontsize=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, which="major", ls=":", alpha=0.5)  # <--- Change 'both' to 'major'
        set_custom_log_xticks(ax2, x_vals, num_ticks=3)

        # --- PANEL 3: Mode-Locked Burst Duration ---
        ax3.plot(x_vals, y_dur, 'd-', color='darkorange', label='Data Points', linewidth=0, markersize=5)
        fit3 = fit_power_law(x_vals, y_dur, threshold=9000)
        print(fit3)
        if fit3:
            gamma, A, x_c, r2, x_f, y_f = fit3
            ax3.plot(x_f, y_f, '--', color='black', label=f'Fit: $~ |x-{{{x_c:.2f}}}|^{{{gamma:.1f}}}$')
        #ax3.set_xscale('symlog', linthresh=0.01)
        #ax3.set_yscale('log')
        ax3.set_xlabel(x_label, fontsize=12)
        ax3.set_ylabel(r"Mean Burst Duration $\langle \tau \rangle$", fontsize=11)
        ax3.set_title("Average Duration per Burst", fontsize=10)
        ax3.grid(True, which="both", ls=":", alpha=0.5)
        ax3.grid(True, which="major", ls=":", alpha=0.5)  # <--- Change 'both' to 'major'
        ax3.legend(loc='best', fontsize=9)
        set_custom_log_xticks(ax3, x_vals, num_ticks=3)

        fig.canvas.draw_idle()

    # Initial Draw
    update_plots()

    # 3. INTERACTIVE CONTROLS (SLIDERS & RADIO BUTTONS)
    # Control Panel Positioning
    ax_xvar = fig.add_axes([0.06, 0.05, 0.12, 0.18])
    radio_xvar = RadioButtons(ax_xvar, ('Vary $\sigma$', 'Vary $\\alpha$', 'Vary $\delta$'), active=1)

    ax_thresh = fig.add_axes([0.21, 0.05, 0.14, 0.18])
    radio_thresh = RadioButtons(ax_thresh, [f'Thresh: {th:.2f}' for th in thresholds], active=state['th_idx'])

    ax_s_sigma = fig.add_axes([0.45, 0.16, 0.48, 0.03])
    ax_s_alpha = fig.add_axes([0.45, 0.10, 0.48, 0.03])
    ax_s_delta = fig.add_axes([0.45, 0.04, 0.48, 0.03])

    s_sigma = Slider(ax_s_sigma, r'Fixed $\sigma$', 0, n_sigmas - 1, valinit=state['i_sigma'], valstep=1)
    s_alpha = Slider(ax_s_alpha, r'Fixed $\alpha$', 0, n_alphas - 1, valinit=state['j_alpha'], valstep=1)
    s_delta = Slider(ax_s_delta, r'Fixed $\delta$', 0, n_deltas - 1, valinit=state['k_delta'], valstep=1)

    # Event Callbacks
    def on_slider_change(val):
        state['i_sigma'] = int(s_sigma.val)
        state['j_alpha'] = int(s_alpha.val)
        state['k_delta'] = int(s_delta.val)

        s_sigma.valtext.set_text(f"{sigmas[int(s_sigma.val)] :.3f}")
        s_alpha.valtext.set_text(f"{alphas[int(s_alpha.val)] :.3f}")
        s_delta.valtext.set_text(f"{deltas[int(s_delta.val)] :.3f}")
        update_plots()

    def on_xvar_change(label):
        if 'sigma' in label:
            state['var_x'] = 'sigma'
        elif 'alpha' in label:
            state['var_x'] = 'alpha'
        else:
            state['var_x'] = 'delta'
        update_plots()

    def on_thresh_change(label):
        state['th_idx'] = [f'Thresh: {th:.2f}' for th in thresholds].index(label)
        update_plots()

    s_sigma.on_changed(on_slider_change)
    s_alpha.on_changed(on_slider_change)
    s_delta.on_changed(on_slider_change)
    radio_xvar.on_clicked(on_xvar_change)
    radio_thresh.on_clicked(on_thresh_change)

    plt.suptitle("1D Power-Law Scaling Analysis (Log-Log Scale)", fontsize=14, fontweight='bold')
    plt.show()

if __name__ == '__main__':
    plot_scaling_laws()