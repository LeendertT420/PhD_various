"""SI-unit visualizer for the sealed-headspace stepwise xenon experiment."""
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
    interval_s = INTERVAL_MIN * 60.0
    fig = plt.figure(figsize=(13, 8))
    fig.subplots_adjust(left=.09, right=.97, top=.90, bottom=.32,
                        hspace=.42, wspace=.30)
    ax_l = fig.add_subplot(221)
    ax_t = fig.add_subplot(222)
    ax_p = fig.add_subplot(223)
    ax_avg = fig.add_subplot(224)
    readout = fig.text(.09, .015, "", family="monospace",
                       fontsize=8.5, va="bottom")

    def redraw(time_s):
        times = solution["time_s"]
        i = int(np.argmin(np.abs(times - time_s)))
        i = int(np.searchsorted(times, times[i], side="right") - 1)
        selected_time = times[i]
        par = solution["parameters"]

        liquid = solution["C_liquid_molm3"][i]
        tissue = solution["C_tissue_molm3"][i]
        liquid_mean = float(np.mean(liquid))
        tissue_mean = float(solution["tissue_average_dimless"][i] *
                            par.lambda_t * par.lambda_l *
                            6894.757293168 / (8.31446261815324 * par.T))
        interface_liquid = solution["C_interface_liquid_molm3"][i]
        interface_tissue = solution["C_interface_tissue_molm3"][i]
        readout.set_text(
            f"Selected time: {selected_time/60.0:.3f} min | "
            f"Headspace Xe pressure: {solution['pressure_pa'][i]:.3f} Pa\n"
            f"Liquid C [mol/m^3]: gas side {liquid[0]:.6g}, "
            f"interface side {liquid[-1]:.6g}, mean {liquid_mean:.6g}, "
            f"interface (liquid basis) {interface_liquid:.6g}\n"
            f"Tissue C [mol/m^3]: center {tissue[0]:.6g}, "
            f"surface {tissue[-1]:.6g}, volume mean {tissue_mean:.6g}, "
            f"interface (tissue basis) {interface_tissue:.6g}"
        )

        for ax in (ax_l, ax_t, ax_p, ax_avg):
            ax.clear()

        ax_l.plot(solution["z_m"], liquid, lw=2)
        ax_l.axvline(par.z_interface, ls="--", color="black",
                     label=r"Organoid interface $z=L$")
        ax_l.set(xlabel="Depth z (m)", ylabel=r"$C_\ell$ (mol m$^{-3}$)",
                 title="Liquid concentration profile")
        ax_l.grid(alpha=.25)
        ax_l.legend(fontsize=8)

        ax_t.plot(solution["r_m"], tissue, lw=2)
        ax_t.set(xlabel="Radial position r (m)",
                 ylabel=r"$C_t$ (mol m$^{-3}$)",
                 title="Organoid radial concentration profile")
        ax_t.grid(alpha=.25)

        ax_p.plot(solution["time_s"], solution["pressure_pa"], lw=2,
                  label="Headspace Xe partial pressure")
        for event in range(1, N_STEPS + 1):
            ax_p.axvline(event * interval_s, ls=":", color="gray", alpha=.5)
        ax_p.axvline(selected_time, ls="--", color="black",
                     label="Selected time")
        ax_p.set(xlabel="Time (s)", ylabel="Xenon partial pressure (Pa)",
                 title="Sealed headspace pressure")
        ax_p.grid(alpha=.25)
        ax_p.legend(fontsize=8)

        liquid_means = solution["C_liquid_molm3"].mean(axis=1)
        tissue_means = solution["tissue_average_dimless"] * (
            par.lambda_t * par.lambda_l * 6894.757293168
            / (8.31446261815324 * par.T)
        )
        ax_avg.plot(solution["time_s"], liquid_means,
                    label="Liquid spatial mean")
        ax_avg.plot(solution["time_s"], tissue_means,
                    label="Tissue volume mean")
        ax_avg.plot(solution["time_s"],
                    solution["C_interface_liquid_molm3"],
                    label="Interface, liquid basis")
        ax_avg.plot(solution["time_s"],
                    solution["C_interface_tissue_molm3"],
                    label="Interface, tissue basis")
        ax_avg.axvline(selected_time, ls="--", color="black",
                       label="Selected time")
        ax_avg.set(xlabel="Time (s)",
                   ylabel=r"Concentration (mol m$^{-3}$)",
                   title="Mean and interface concentrations")
        ax_avg.grid(alpha=.25)
        ax_avg.legend(fontsize=8)

        fig.suptitle(
            f"Stepwise xenon diffusion | t={selected_time:.2f} s, "
            f"headspace pressure={solution['pressure_pa'][i]:.2f} Pa"
        )
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([.17, .245, .66, .035])
    time_values = solution["time_s"]
    time_slider = Slider(
        slider_ax, "Time (s)", 0.0, float(time_values[-1]), valinit=0.0,
        valstep=float(time_values[1] - time_values[0]),
    )
    time_slider.on_changed(redraw)

    print(f"Starting Xe pressure: 0 Pa")
    print(f"Additions: {N_STEPS} x {PRESSURE_STEP_PSI:g} psi "
          f"every {INTERVAL_MIN:g} min; first addition at "
          f"{interval_s:g} s")
    print(f"Final simulation time: {time_values[-1]:g} s")
    redraw(0.0)
    plt.show()


if __name__ == "__main__":
    main()
