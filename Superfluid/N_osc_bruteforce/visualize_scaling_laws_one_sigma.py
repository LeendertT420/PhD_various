import os
import re
import glob

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import RadioButtons, Slider

from equations import upper_boundary, lower_boundary


def analyze_intermittency_sweep(
    pattern="bruteforce_sweep_results_sigma_*_N=15.npz",
    T_sim=1000,
    T_eval=8000.0,
    N=15,
):

    # ------------------------------------------------------------------
    # Discover files and extract sigma values
    # ------------------------------------------------------------------

    files = glob.glob(pattern)

    if len(files) == 0:
        print(f"No files found matching:\n{pattern}")
        return

    sigma_files = []

    for f in files:
        match = re.search(
            r"sigma_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            os.path.basename(f),
        )

        if match:
            sigma = float(match.group(1))
            sigma_files.append((sigma, f))

    if len(sigma_files) == 0:
        print("Could not extract sigma values from filenames.")
        return

    sigma_files.sort(key=lambda x: x[0])
    print(sigma_files)
    sigma_values = np.array([s for s, _ in sigma_files])
    print(sigma_values)
    print(f"Found {len(sigma_values)} sigma files.")

    # ------------------------------------------------------------------
    # Lazy-loading cache
    # ------------------------------------------------------------------

    dataset_cache = {}

    def process_dataset(filepath):

        data = np.load(filepath, allow_pickle=True)

        

        sigmas = data["sigmas_axis"]
        alphas = data["alphas_axis"]
        deltas = data["deltas_axis"]

        classifications = data["classifications"]
        ranges_list = data["ranges_list"]

        thresholds = (
            data["thresholds"]
            if "thresholds" in data
            else np.array([0.80, 0.85, 0.90, 0.95])
        )

        R_ML_mean = data["R_ML_mean"]

        n_alphas = len(alphas)
        n_deltas = len(deltas)

        n_thresholds = len(thresholds)

        n_grid_pts = n_alphas * n_deltas

        # --------------------------------------------------------------
        # Reshape
        # --------------------------------------------------------------

        grid_class = classifications.reshape(
            (n_alphas, n_deltas)
        )

        grid_ml_mean = np.asarray(
            R_ML_mean.reshape((n_alphas, n_deltas)),
            dtype=np.float64,
        )

        grid_ml_mean[grid_class != "CHAOTIC"] = np.nan

        if ranges_list.size == n_grid_pts * n_thresholds:

            ranges_grid = ranges_list.reshape(
                (n_alphas, n_deltas, n_thresholds)
            )

            is_4d = True

        else:

            ranges_grid = ranges_list.reshape(
                (n_alphas, n_deltas)
            )

            is_4d = False

        # --------------------------------------------------------------
        # Compute intermittency metrics
        # --------------------------------------------------------------

        grid_mean_dt = np.full(
            (n_thresholds, n_alphas, n_deltas),
            np.nan,
        )

        grid_fraction_ml = np.zeros(
            (n_thresholds, n_alphas, n_deltas)
        )

        print(
            f"Processing sigma={sigmas[0]:.6f}"
        )

        for th_idx in range(n_thresholds):

            for j in range(n_alphas):

                for k in range(n_deltas):

                    if grid_class[j, k] != "CHAOTIC":
                        continue

                    intervals = (
                        ranges_grid[j, k, th_idx]
                        if is_4d
                        else ranges_grid[j, k][th_idx]
                    )

                    if intervals is None:
                        continue

                    if len(intervals) == 0:
                        continue

                    starts = []
                    stops = []

                    for r in intervals:

                        starts.append(r[0])

                        if (
                            r[1] is None
                            or np.isnan(r[1])
                        ):
                            stops.append(
                                T_sim + T_eval
                            )
                        else:
                            stops.append(r[1])

                    # --------------------------------------
                    # Mean burst duration
                    # --------------------------------------

                    durations = [
                        max(0.0, sp - st)
                        for st, sp in zip(
                            starts,
                            stops,
                        )
                    ]

                    if len(durations) > 0:
                        grid_mean_dt[
                            th_idx,
                            j,
                            k,
                        ] = np.mean(durations)

                    # --------------------------------------
                    # Fraction mode locked
                    # --------------------------------------

                    total_duration = np.sum(
                        durations
                    )

                    grid_fraction_ml[
                        th_idx,
                        j,
                        k,
                    ] = min(
                        total_duration / T_eval,
                        1.0,
                    )

        return {
            "sigma": float(sigmas[0]),
            "alphas": alphas,
            "deltas": deltas,
            "thresholds": thresholds,
            "grid_mean_dt": grid_mean_dt,
            "grid_fraction_ml": grid_fraction_ml,
            "grid_ml_mean": grid_ml_mean,
        }
    
    datasets = []

    for i, (sigma, filepath) in enumerate(sigma_files):

        print(
            f"[{i+1}/{len(sigma_files)}] "
            f"Processing sigma={sigma:.6f}"
        )

        ds = process_dataset(filepath)

        datasets.append(ds)

    def get_dataset(idx):
        return datasets[idx]

    # ------------------------------------------------------------------
    # Load first dataset
    # ------------------------------------------------------------------

    current_sigma_idx = [0]
    current_threshold_idx = [1]

    ds0 = datasets[0]

    # ------------------------------------------------------------------
    # Analytical boundaries
    # ------------------------------------------------------------------

    try:
        up_bound = upper_boundary(
            N,
            ds0["deltas"],
        )

        low_bound = lower_boundary(
            N,
            ds0["deltas"],
        )

        has_boundaries = True

    except Exception:

        has_boundaries = False

    
    print("\nPreprocessing all sigma files...")
    print(f"Found {len(sigma_files)} files")
    global_max_dt = np.nanmax([
        np.nanmax(ds["grid_mean_dt"])
        for ds in datasets
    ])

    global_max_ml = np.nanmax([
        np.nanmax(ds["grid_ml_mean"])
        for ds in datasets
    ])

    if np.isnan(global_max_dt):
        global_max_dt = 1.0

    if np.isnan(global_max_ml):
        global_max_ml = 1.0

    

    print("Finished preprocessing all datasets.\n")


    

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
        sharex=True,
        sharey=True,
    )

    plt.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.28,
        wspace=0.28,
    )

    cmap_dt = plt.get_cmap(
        "plasma"
    ).copy()
    cmap_dt.set_bad("lightgrey")

    cmap_frac = plt.get_cmap(
        "viridis"
    ).copy()
    cmap_frac.set_bad("lightgrey")

    cmap_ml = plt.get_cmap(
        "magma"
    ).copy()
    cmap_ml.set_bad("lightgrey")

    colorbars = [None, None, None]

    # ------------------------------------------------------------------
    # Drawing routine
    # ------------------------------------------------------------------

    def draw_plots():

        ds = get_dataset(
            current_sigma_idx[0]
        )

        th = min(
            current_threshold_idx[0],
            len(ds["thresholds"]) - 1,
        )

        alphas = ds["alphas"]
        deltas = ds["deltas"]

        delta_grid, alpha_grid = np.meshgrid(
            deltas,
            alphas,
        )

        mean_dt = ds["grid_mean_dt"][th]
        frac_ml = ds["grid_fraction_ml"][th]
        ml_mean = ds["grid_ml_mean"]

        sigma = ds["sigma"]

        max_dt = np.nanmax(
            ds["grid_mean_dt"]
        )

        if np.isnan(max_dt):
            max_dt = 1.0

        for ax in axes:
            ax.clear()

        # --------------------------------------------------------------
        # Plot 1
        # --------------------------------------------------------------

        pcm0 = axes[0].pcolormesh(
            delta_grid,
            alpha_grid,
            mean_dt,
            cmap=cmap_dt,
            vmin=0,
            vmax=max_dt,
            shading="nearest",
        )

        axes[0].set_title(
            rf"$\langle \Delta T \rangle$ | "
            rf"Thresh={ds['thresholds'][th]:.2f}"
        )

        axes[0].set_xlabel(r"$\delta$")
        axes[0].set_ylabel(r"$\alpha$")

        # --------------------------------------------------------------
        # Plot 2
        # --------------------------------------------------------------

        pcm1 = axes[1].pcolormesh(
            delta_grid,
            alpha_grid,
            frac_ml,
            cmap=cmap_frac,
            vmin=0,
            vmax=1,
            shading="nearest",
        )

        axes[1].set_title(
            rf"$F_{{ML}}$ | "
            rf"Thresh={ds['thresholds'][th]:.2f}"
        )

        axes[1].set_xlabel(r"$\delta$")
        axes[1].set_ylabel(r"$\alpha$")

        # --------------------------------------------------------------
        # Plot 3
        # --------------------------------------------------------------

        pcm2 = axes[2].pcolormesh(
            delta_grid,
            alpha_grid,
            ml_mean,
            cmap=cmap_ml,
            vmin=0,
            vmax=1,
            shading="nearest",
        )

        axes[2].set_title(
            r"$\langle R_{ML}\rangle$"
        )

        axes[2].set_xlabel(r"$\delta$")
        axes[2].set_ylabel(r"$\alpha$")

        # --------------------------------------------------------------
        # Boundaries
        # --------------------------------------------------------------

        if has_boundaries:

            try:

                up = upper_boundary(
                    N,
                    deltas,
                )

                low = lower_boundary(
                    N,
                    deltas,
                )

                for ax in axes:

                    ax.plot(
                        deltas,
                        up,
                        "k--",
                        lw=1.2,
                    )

                    ax.plot(
                        deltas,
                        low,
                        "k--",
                        lw=1.2,
                    )

            except Exception:
                pass

        # --------------------------------------------------------------
        # Colorbars
        # --------------------------------------------------------------

        for cb in colorbars:
            if cb is not None:
                cb.remove()

        colorbars[0] = fig.colorbar(
            pcm0,
            ax=axes[0],
            pad=0.02,
        )

        colorbars[0].set_label(
            r"$\langle \Delta T\rangle$"
        )

        colorbars[1] = fig.colorbar(
            pcm1,
            ax=axes[1],
            pad=0.02,
        )

        colorbars[1].set_label(
            r"$F_{ML}$"
        )

        colorbars[2] = fig.colorbar(
            pcm2,
            ax=axes[2],
            pad=0.02,
        )

        colorbars[2].set_label(
            r"$\langle R_{ML}\rangle$"
        )

        fig.suptitle(
            rf"Intermittency & Mode-Locking "
            rf"($\sigma={sigma_values[0]:.4f}$)",
            fontsize=14,
            fontweight="bold",
        )

    # ------------------------------------------------------------------
    # Initial draw
    # ------------------------------------------------------------------
    

    draw_plots()

    # ------------------------------------------------------------------
    # Threshold selector
    # ------------------------------------------------------------------

    ax_thresh = fig.add_axes(
        [0.37, 0.03, 0.25, 0.12]
    )

    thresholds0 = ds0["thresholds"]

    radio_thresh = RadioButtons(
        ax_thresh,
        [
            f"{t:.2f}"
            for t in thresholds0
        ],
        active=min(
            1,
            len(thresholds0) - 1,
        ),
    )

    def update_threshold(label):

        ds = get_dataset(
            current_sigma_idx[0]
        )

        labels = [
            f"{t:.2f}"
            for t in ds["thresholds"]
        ]

        current_threshold_idx[0] = (
            labels.index(label)
        )

        draw_plots()

        fig.canvas.draw_idle()

    radio_thresh.on_clicked(
        update_threshold
    )

    # ------------------------------------------------------------------
    # Sigma slider
    # ------------------------------------------------------------------

    ax_sigma = fig.add_axes(
        [0.15, 0.18, 0.70, 0.03]
    )

    sigma_slider = Slider(
        ax=ax_sigma,
        label=r"$\sigma$ index",
        valmin=0,
        valmax=len(sigma_values) - 1,
        valinit=0,
        valstep=1,
    )

    sigma_slider.valtext.set_text(
        f"{sigma_values[0]:.4f}"
    )

    def update_sigma(val):

        idx = int(
            sigma_slider.val
        )

        current_sigma_idx[0] = idx

        ds = get_dataset(idx)

        current_threshold_idx[0] = min(
            current_threshold_idx[0],
            len(ds["thresholds"]) - 1,
        )

        sigma_slider.valtext.set_text(
            f"{sigma_values[idx]:.4f}"
        )

        draw_plots()

        fig.canvas.draw_idle()

    sigma_slider.on_changed(
        update_sigma
    )

    plt.show()


if __name__ == "__main__":

    analyze_intermittency_sweep()