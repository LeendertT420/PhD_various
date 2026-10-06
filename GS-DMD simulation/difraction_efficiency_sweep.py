import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from dmd_simulation import DMD_simulation


def build_interactive_diffraction_plot(N_spots: int = 1, seed: int = 42):
    """
    Simulates Lee holography across carrier periods (2..200 px).
    Dynamically tracks the physical movement of the 0th, +1st, and -1st diffraction 
    orders so integration circles move along with the light spots.
    """
    # 1. Initialize Simulation with Fixed Random Seed
    np.random.seed(seed)

    sim = DMD_simulation(
        resolution=(1140, 912),
        pitch=7.6e-6,
        wavelength=930e-9,
        f=0.1,
    )

    print(f"Generating target for N_spots = {N_spots} (Seed={seed})...")
    sim.generate_input_and_target(N_spots=N_spots, spacing="random", plot=False)

    print("Running Gerchberg-Saxton phase retrieval...")
    sim.run_gerchberg_saxton(N_iterations=15)

    carrier_periods = np.arange(2, 201, 1)
    orientations = ["horizontal", "vertical", "diagonal"]
    integration_radius_px = 5

    # Integration circle radius in physical units (mm)
    dfx_mm = (sim.fx[1] - sim.fx[0]) * 1e3
    radius_mm = integration_radius_px * dfx_mm

    # Nominal unshifted target spot coordinates [m]
    nominal_x_m = sim.positions_x.copy()
    nominal_y_m = sim.positions_y.copy()

    # Precomputation containers
    precomputed = {
        orient: {
            "eff_1st": [],
            "eff_m1st": [],
            "eff_0th": [],
            "eff_bg": [],
            "farfields": [],
            "centers_1st": [],   # List of list of (x, y) tuples in mm
            "centers_m1st": [],  # List of list of (x, y) tuples in mm
        }
        for orient in orientations
    }

    print("Precalculating far fields and tracking order positions for T = 2..200 px...")
    for orient in orientations:
        for period in carrier_periods:
            # 1. Generate Lee Hologram for current carrier period & orientation
            sim.generate_lee_hologram(
                carrier_period=period, carrier_orientation=orient
            )

            # 2. Compute carrier shift offset vector [m]
            carrier_shift = (sim.wavelength * sim.f) / (period * sim.pitch)

            if orient == "horizontal":
                shift_x, shift_y = carrier_shift, 0.0
            elif orient == "vertical":
                shift_x, shift_y = 0.0, carrier_shift
            elif orient == "diagonal":
                shift_x, shift_y = carrier_shift, carrier_shift

            # 3. Compute shifted physical spot locations [mm]
            centers_1st_mm = [
                ((px + shift_x) * 1e3, (py + shift_y) * 1e3)
                for px, py in zip(nominal_x_m, nominal_y_m)
            ]
            centers_m1st_mm = [(-cx, -cy) for cx, cy in centers_1st_mm]

            # 4. Propagate field
            far_field = sim.propagate_to_far_field().astype(np.float32)
            total_power = np.sum(far_field)

            Ny, Nx = far_field.shape
            center_x, center_y = Nx // 2, Ny // 2
            y_grid, x_grid = np.ogrid[:Ny, :Nx]

            # 0th Order Integration (Fixed at DC Center)
            dc_mask = (
                (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
            ) <= integration_radius_px**2
            p_0th = np.sum(far_field[dc_mask])

            # +1st Order Integration (Centered at shifted spot locations)
            mask_1st = np.zeros((Ny, Nx), dtype=bool)
            for cx_mm, cy_mm in centers_1st_mm:
                idx_x = np.argmin(np.abs(sim.fx * 1e3 - cx_mm))
                idx_y = np.argmin(np.abs(sim.fy * 1e3 - cy_mm))
                mask_1st |= (
                    (x_grid - idx_x) ** 2 + (y_grid - idx_y) ** 2
                ) <= integration_radius_px**2
            p_1st = np.sum(far_field[mask_1st])

            # -1st Order Integration (Centered at shifted conjugate locations)
            mask_m1st = np.zeros((Ny, Nx), dtype=bool)
            for cx_mm, cy_mm in centers_m1st_mm:
                idx_x = np.argmin(np.abs(sim.fx * 1e3 - cx_mm))
                idx_y = np.argmin(np.abs(sim.fy * 1e3 - cy_mm))
                mask_m1st |= (
                    (x_grid - idx_x) ** 2 + (y_grid - idx_y) ** 2
                ) <= integration_radius_px**2
            p_m1st = np.sum(far_field[mask_m1st])

            # Background
            p_bg = max(0.0, total_power - (p_1st + p_m1st + p_0th))

            # Store precomputed results
            precomputed[orient]["farfields"].append(far_field)
            precomputed[orient]["centers_1st"].append(centers_1st_mm)
            precomputed[orient]["centers_m1st"].append(centers_m1st_mm)
            precomputed[orient]["eff_1st"].append((p_1st / total_power) * 100)
            precomputed[orient]["eff_m1st"].append((p_m1st / total_power) * 100)
            precomputed[orient]["eff_0th"].append((p_0th / total_power) * 100)
            precomputed[orient]["eff_bg"].append((p_bg / total_power) * 100)

    # 2. Setup Figure Layout (2 Rows x 3 Columns)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.3, wspace=0.25)

    ax_effs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_fart = [fig.add_subplot(gs[1, i]) for i in range(3)]

    vlines = []
    im_handles = []
    circle_patches = []

    extents_mm = [sim.fx[0] * 1e3, sim.fx[-1] * 1e3, sim.fy[0] * 1e3, sim.fy[-1] * 1e3]
    initial_idx = 4  # Start display at T_carrier = 6 px

    # 3. Create Plots & Initial Circle Handles
    for i, orient in enumerate(orientations):
        # Top Row: Diffraction Efficiency Plots
        ax_eff = ax_effs[i]
        ax_eff.plot(
            carrier_periods,
            precomputed[orient]["eff_1st"],
            "-",
            color="crimson",
            label="+1st Order (Target)",
        )
        ax_eff.plot(
            carrier_periods,
            precomputed[orient]["eff_m1st"],
            "--",
            color="dodgerblue",
            label="-1st Order (Conjugate)",
        )
        ax_eff.plot(
            carrier_periods,
            precomputed[orient]["eff_0th"],
            ":",
            color="black",
            label="0th Order (DC)",
        )
        ax_eff.plot(
            carrier_periods,
            precomputed[orient]["eff_bg"],
            "-.",
            color="gray",
            label="Background",
        )

        vline = ax_eff.axvline(
            x=carrier_periods[initial_idx],
            color="red",
            linestyle="--",
            linewidth=1.5,
        )
        vlines.append(vline)

        ax_eff.set_title(
            f"Orientation: {orient.capitalize()}", fontsize=12, fontweight="bold"
        )
        ax_eff.set_xlabel("Carrier Period [px]")
        if i == 0:
            ax_eff.set_ylabel("Diffraction Efficiency [%]")
        ax_eff.grid(True, linestyle="--", alpha=0.5)
        ax_eff.legend(loc="upper right", fontsize=8)

        # Bottom Row: Far-Field Imshow
        ax_ff = ax_fart[i]
        initial_ff = precomputed[orient]["farfields"][initial_idx]

        im = ax_ff.imshow(
            initial_ff + 1e-6,
            extent=extents_mm,
            origin="lower",
            cmap="magma",
            norm=LogNorm(vmin=1e-5, vmax=np.max(initial_ff)),
        )
        im_handles.append(im)

        # 0th Order Circle (Fixed at DC center)
        c_0th = Circle(
            (0.0, 0.0),
            radius_mm,
            edgecolor="yellow",
            facecolor="none",
            lw=1.5,
            label="0th Order (DC)",
        )
        ax_ff.add_patch(c_0th)

        # Dynamic +1st Order Circles
        c_1st_list = []
        for idx_s, pos in enumerate(precomputed[orient]["centers_1st"][initial_idx]):
            c = Circle(
                pos,
                radius_mm,
                edgecolor="cyan",
                facecolor="none",
                lw=1.5,
                label="+1st Order" if idx_s == 0 else None,
            )
            ax_ff.add_patch(c)
            c_1st_list.append(c)

        # Dynamic -1st Order Circles
        c_m1st_list = []
        for idx_s, pos in enumerate(precomputed[orient]["centers_m1st"][initial_idx]):
            c = Circle(
                pos,
                radius_mm,
                edgecolor="magenta",
                facecolor="none",
                lw=1.5,
                label="-1st Order" if idx_s == 0 else None,
            )
            ax_ff.add_patch(c)
            c_m1st_list.append(c)

        circle_patches.append({"+1st": c_1st_list, "-1st": c_m1st_list})

        ax_ff.set_title(f"Far Field Intensity ({orient})", fontsize=11)
        ax_ff.set_xlabel("x [mm]")
        if i == 0:
            ax_ff.set_ylabel("y [mm]")
        ax_ff.legend(
            loc="upper right", fontsize=8, facecolor="black", labelcolor="white"
        )

    # 4. Interactive Slider Setup
    ax_slider = plt.axes([0.25, 0.02, 0.50, 0.025])
    slider = Slider(
        ax=ax_slider,
        label="Carrier Period [px] ",
        valmin=2,
        valmax=200,
        valinit=carrier_periods[initial_idx],
        valstep=1,
        valfmt="%d px",
    )

    def update(val):
        period_idx = int(slider.val) - 2

        for i, orient in enumerate(orientations):
            # Move cursor line in efficiency plots
            vlines[i].set_xdata([slider.val, slider.val])

            # Update far-field image
            ff_data = precomputed[orient]["farfields"][period_idx]
            im_handles[i].set_data(ff_data + 1e-6)

            # Move +1st order circles along with shifted spot centers
            centers_1st = precomputed[orient]["centers_1st"][period_idx]
            for idx_s, c in enumerate(circle_patches[i]["+1st"]):
                c.center = centers_1st[idx_s]

            # Move -1st order circles along with shifted conjugate centers
            centers_m1st = precomputed[orient]["centers_m1st"][period_idx]
            for idx_s, c in enumerate(circle_patches[i]["-1st"]):
                c.center = centers_m1st[idx_s]

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.suptitle(
        f"Dynamic Order-Tracking DMD Diffraction Analysis ($N_{{\\text{{spots}}}} = {N_spots}$)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.show()


if __name__ == "__main__":
    build_interactive_diffraction_plot(N_spots=5)