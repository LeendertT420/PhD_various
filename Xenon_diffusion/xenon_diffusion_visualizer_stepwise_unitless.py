"""Unitless visualizer for the sealed-headspace stepwise xenon experiment.

Axes use z/L, r/R, and tau = D_t*t/R^2. Plotted concentrations and
headspace pressure are percentages of their final equilibrium values.
Intermediate equilibrium levels scale as k/N_STEPS of the final level
after k equal xenon additions, following c_eq = 1/q.
"""
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
INTERVAL_MIN = 7
N_STEPS = 15
SAMPLES_PER_INTERVAL = 30


def main():
    solution = solve_xenon_stepwise(
        PARAMS, pressure_step_psi=PRESSURE_STEP_PSI,
        interval_min=INTERVAL_MIN, n_steps=N_STEPS,
        samples_per_interval=SAMPLES_PER_INTERVAL,
    )
    final_input_pressure = N_STEPS * PRESSURE_STEP_PSI
    tau_interval = INTERVAL_MIN * 60.0 / PARAMS.time_scale
    equilibrium_levels_pct = 100.0 * np.arange(1, N_STEPS + 1) / N_STEPS

    fig = plt.figure(figsize=(13, 8))
    fig.subplots_adjust(left=.09, right=.97, top=.90, bottom=.24,
                        hspace=.42, wspace=.30)
    ax_l = fig.add_subplot(221)
    ax_t = fig.add_subplot(222)
    ax_p = fig.add_subplot(223)
    ax_avg = fig.add_subplot(224)
    readout = fig.text(0.09, 0.015, "", family="monospace",
                       fontsize=8.5, va="bottom")

    def redraw(tau):
        times = solution["tau"]
        i = int(np.argmin(np.abs(times - tau)))
        i = int(np.searchsorted(times, times[i], side="right") - 1)
        tau_now = times[i]
        q = PARAMS.q

        liquid_pct = 100.0 * q * solution["c_liquid_dimless"][i] / final_input_pressure
        tissue_pct = 100.0 * q * solution["c_tissue_dimless"][i] / final_input_pressure
        pressure_pct = 100.0 * q * solution["p_dimless"] / final_input_pressure
        interface_pct = 100.0 * q * solution["c_interface_dimless"] / final_input_pressure
        liquid_mean_pct = (
            100.0 * q * solution["liquid_average_dimless"] / final_input_pressure
        )
        tissue_mean_pct = (
            100.0 * q * solution["tissue_average_dimless"] / final_input_pressure
        )

        liquid_now = q * solution["c_liquid_dimless"][i] / final_input_pressure
        tissue_now = q * solution["c_tissue_dimless"][i] / final_input_pressure
        pressure_now = q * solution["p_dimless"][i] / final_input_pressure
        interface_now = q * solution["c_interface_dimless"][i] / final_input_pressure
        readout.set_text(
            "Selected-time values (fraction of final equilibrium; 1.000 = 100%):\n"
            f"Headspace: {pressure_now:.4f}  |  "
            f"Liquid: gas side {liquid_now[0]:.4f}, interface side "
            f"{liquid_now[-1]:.4f}, mean "
            f"{q * solution['liquid_average_dimless'][i] / final_input_pressure:.4f}  |  "
            f"Tissue: center {tissue_now[0]:.4f}, surface {tissue_now[-1]:.4f}, "
            f"volume mean "
            f"{q * solution['tissue_average_dimless'][i] / final_input_pressure:.4f}"
        )

        for ax in (ax_l, ax_t, ax_p, ax_avg):
            ax.clear()
            for step, level in enumerate(equilibrium_levels_pct, start=1):
                label = (
                    "Equilibrium after each cumulative addition"
                    if step == 1 else None
                )
                ax.axhline(level, ls=":", color="gray", lw=.8, alpha=.55,
                           label=label)

        ax_l.plot(solution["x"], liquid_pct, lw=2)
        ax_l.axvline(1.0, ls="--", color="black",
                     label=r"Organoid interface $z/L=1$")
        ax_l.set(xlabel=r"Dimensionless depth $z/L$",
                 ylabel="Liquid concentration (% of final equilibrium)",
                 title="Liquid concentration profile")
        ax_l.set_xlim(0.0, 1.0)
        ax_l.set_ylim(bottom=0.0)
        ax_l.grid(alpha=.25)
        ax_l.legend(fontsize=8)

        ax_t.plot(solution["rho"], tissue_pct, lw=2)
        ax_t.set(xlabel=r"Dimensionless radius $r/R$",
                 ylabel="Tissue concentration (% of final equilibrium)",
                 title="Organoid radial concentration profile")
        ax_t.set_xlim(0.0, 1.0)
        ax_t.set_ylim(bottom=0.0)
        ax_t.grid(alpha=.25)
        ax_t.legend(fontsize=8)

        ax_p.plot(solution["tau"], pressure_pct, lw=2,
                  label="Headspace Xe partial pressure")
        for event in range(1, N_STEPS + 1):
            ax_p.axvline(event * tau_interval, ls=":", color="gray", alpha=.5)
        ax_p.axvline(tau_now, ls="--", color="black", label="Selected time")
        ax_p.set(xlabel=r"Dimensionless time $\tau=D_t t/R^2$",
                 ylabel="Headspace pressure (% of final equilibrium)",
                 title="Headspace xenon partial pressure")
        ax_p.set_ylim(bottom=0.0)
        ax_p.grid(alpha=.25)
        ax_p.legend(fontsize=8)

        ax_avg.plot(solution["tau"], liquid_mean_pct,
                    label="Liquid spatial mean")
        ax_avg.plot(solution["tau"], tissue_mean_pct,
                    label="Tissue volume mean")
        ax_avg.plot(solution["tau"], interface_pct,
                    label="Interface concentration")
        ax_avg.axvline(tau_now, ls="--", color="black", label="Selected time")
        ax_avg.set(xlabel=r"Dimensionless time $\tau=D_t t/R^2$",
                   ylabel="Mean concentration (% of final equilibrium)",
                   title="Mean concentrations")
        ax_avg.set_ylim(bottom=0.0)
        ax_avg.grid(alpha=.25)
        ax_avg.legend(fontsize=8)

        fig.suptitle(
            f"Unitless stepwise xenon diffusion | "
            f"$\\tau$={tau_now:.3f} | {N_STEPS} additions"
        )
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([.17, .135, .66, .035])
    tau_values = solution["tau"]
    time_slider = Slider(
        slider_ax, r"Dimensionless time $\tau$", 0.0,
        float(tau_values[-1]), valinit=0.0,
        valstep=float(tau_values[1] - tau_values[0]),
    )
    time_text = fig.text(.84, .135, "", fontsize=9, va="center")

    def update_time(tau):
        time_text.set_text(
            f"t = {tau * PARAMS.time_scale / 60.0:.1f} min"
        )
        redraw(tau)

    time_slider.on_changed(update_time)

    print(f"q={PARAMS.q:.6g}, delta={PARAMS.delta:.6g}")
    print(f"Dimensionless interval between additions: tau={tau_interval:.6g}")
    print("Cumulative equilibrium levels follow c_eq=1/q and scale linearly "
          "with the total xenon added.")
    print("All plotted positions and time are dimensionless; vertical values "
          "are percentages of final equilibrium.")
    update_time(0.0)
    plt.show()


if __name__ == "__main__":
    main()
