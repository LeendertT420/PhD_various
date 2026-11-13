"""Interactive plots for the sealed-headspace stepwise xenon experiment."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from xenon_diffusion_solver import Parameters
from xenon_diffusion_solver_stepwise import solve_xenon_stepwise


PARAMS = Parameters(
    R=0.5e-3, L=3.53e-3, A=2.833e-4, V_g=1.27e-6,
    T=310.15, P0=0.0,
    D_l=1.5e-9, D_t=6e-10,
    lambda_l=0.078, lambda_t=0.175,
    N_l=100, N_t=60, tau_end=20.0, n_output=500,
)
PRESSURE_STEP_PSI = 2.0
INTERVAL_MIN = 7.0
N_STEPS = 15
SAMPLES_PER_INTERVAL = 30


def main():
    solution = solve_xenon_stepwise(
        PARAMS, pressure_step_psi=PRESSURE_STEP_PSI,
        interval_min=INTERVAL_MIN, n_steps=N_STEPS,
        samples_per_interval=SAMPLES_PER_INTERVAL,
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(left=.09, right=.97, top=.90, bottom=.22,
                        hspace=.42, wspace=.30)
    ax_l, ax_t, ax_p, ax_avg = axes.flat

    def redraw(time_min):
        i = int(np.argmin(np.abs(solution["time_min"] - time_min)))
        selected_time = solution["time_min"][i]
        par = solution["parameters"]
        for ax in axes.flat:
            ax.clear()

        ax_l.plot(solution["z_m"], solution["C_liquid_molm3"][i], lw=2)
        ax_l.axvline(par.z_interface, ls="--", color="black",
                     label=r"Organoid interface $z=L$")
        ax_l.set(xlabel="Depth z (m)", ylabel=r"$C_\ell$ (mol m$^{-3}$)",
                 title="Liquid concentration profile")
        ax_l.grid(alpha=.25)
        ax_l.legend(fontsize=8)

        ax_t.plot(solution["r_m"], solution["C_tissue_molm3"][i], lw=2)
        ax_t.set(xlabel="Radial position r (m)",
                 ylabel=r"$C_t$ (mol m$^{-3}$)",
                 title="Organoid radial concentration profile")
        ax_t.grid(alpha=.25)

        ax_p.plot(solution["time_min"], solution["pressure_psi"], lw=2,
                  label="Headspace Xe partial pressure")
        ax_p.axvline(selected_time, ls="--", color="black",
                     label="Selected time")
        ax_p.set(xlabel="Time (min)", ylabel="Xenon partial pressure (psi)",
                 title="Sealed headspace pressure")
        ax_p.grid(alpha=.25)
        ax_p.legend(fontsize=8)

        liquid_mean = solution["C_liquid_molm3"].mean(axis=1)
        tissue_mean = solution["tissue_average_dimless"] * (
            par.lambda_t * par.lambda_l * 6894.757293168
            / (8.31446261815324 * par.T)
        )
        ax_avg.plot(solution["time_min"], liquid_mean,
                    label="Liquid spatial mean")
        ax_avg.plot(solution["time_min"], tissue_mean,
                    label="Tissue volume mean")
        ax_avg.axvline(selected_time, ls="--", color="black")
        ax_avg.set(xlabel="Time (min)",
                   ylabel=r"Mean concentration (mol m$^{-3}$)",
                   title="Mean concentrations")
        ax_avg.grid(alpha=.25)
        ax_avg.legend(fontsize=8)

        fig.suptitle(
            f"Stepwise xenon addition | t={selected_time:.1f} min, "
            f"headspace pressure={solution['pressure_psi'][i]:.2f} psi"
        )
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([.17, .09, .66, .035])
    time_slider = Slider(
        slider_ax, "Time (min)", 0.0, float(solution["time_min"][-1]),
        valinit=0.0,
    )
    time_slider.on_changed(redraw)

    print(f"Starting Xe pressure: 0 psi")
    print(f"Additions: {N_STEPS} x {PRESSURE_STEP_PSI:g} psi "
          f"every {INTERVAL_MIN:g} min, first addition at "
          f"{INTERVAL_MIN:g} min")
    print(f"Final simulation time: {solution['time_min'][-1]:g} min")
    redraw(0.0)
    plt.show()


if __name__ == "__main__":
    main()
