"""Interactive SI-unit plots for xenon_solver.py.
Requires numpy, scipy, matplotlib. Run this file in a desktop Python session.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from xenon_diffusion_solver import Parameters, solve_xenon

# All physical and numerical parameters are hard-coded here.
PARAMS = Parameters(
    R=.5e-3, L=3.53e-3, A=2.833e-4, V_g=1e-6,
    T=310.15, P0=101325.0,
    D_l=1.55e-9, D_t=1.00e-9,
    lambda_l=0.083, lambda_t=0.175,
    N_l=100, N_t=60, tau_end=12.0, n_output=500,
)
P_MIN, P_MAX = 0.0, 2.0*101325.0


def main():
    state = {"solution": solve_xenon(PARAMS)}
    fig = plt.figure(figsize=(13, 8))
    fig.subplots_adjust(left=.09, right=.97, top=.90, bottom=.24,
                        hspace=.42, wspace=.30)
    ax_l = fig.add_subplot(221)
    ax_t = fig.add_subplot(222)
    ax_p = fig.add_subplot(223)
    ax_avg = fig.add_subplot(224)
    fig.suptitle("Closed xenon headspace–liquid–organoid system")

    def nearest_index(sol, t):
        return int(np.argmin(np.abs(sol["time_s"]-t)))

    def redraw(sol, i):
        par = sol["parameters"]
        t = sol["time_s"][i]
        for ax in (ax_l, ax_t, ax_p, ax_avg):
            ax.clear()

        ax_l.plot(sol["z_m"], sol["C_liquid_molm3"][i], lw=2)
        ax_l.axvline(par.z_interface, ls="--", color="black",
                     label=r"Organoid interface $z=L$")
        ax_l.set(xlabel="Depth z (m)", ylabel=r"$C_\ell$ (mol m$^{-3}$)",
                 title="Liquid concentration profile")
        ax_l.grid(alpha=.25); ax_l.legend(fontsize=8)

        ax_t.plot(sol["r_m"], sol["C_tissue_molm3"][i], lw=2)
        ax_t.set(xlabel="Radial position r (m)",
                 ylabel=r"$C_t$ (mol m$^{-3}$)",
                 title="Organoid radial concentration profile")
        ax_t.grid(alpha=.25)

        ax_p.plot(sol["time_s"], sol["pressure_pa"], lw=2,
                  label="Headspace Xe partial pressure")
        ax_p.axvline(t, ls="--", color="black", label="Selected time")
        ax_p.axhline(sol["equilibrium_pressure_pa"], ls=":",
                     label="Equilibrium prediction")
        ax_p.set(xlabel="Time (s)", ylabel="Pressure (Pa)",
                 title="Headspace partial pressure")
        ax_p.grid(alpha=.25); ax_p.legend(fontsize=8)

        liquid_mean = sol["C_liquid_molm3"].mean(axis=1)
        tissue_mean = sol["tissue_average_dimless"] * (
            par.lambda_t*par.lambda_l*par.P0/(8.31446261815324*par.T))
        ax_avg.plot(sol["time_s"], liquid_mean, label="Liquid spatial mean")
        ax_avg.plot(sol["time_s"], tissue_mean, label="Tissue volume mean")
        ax_avg.axvline(t, ls="--", color="black")
        ax_avg.set(xlabel="Time (s)", ylabel=r"Mean concentration (mol m$^{-3}$)",
                   title="Mean concentrations")
        ax_avg.grid(alpha=.25); ax_avg.legend(fontsize=8)

        fig.suptitle(f"Xenon diffusion | t={t:.0f} s ({t/3600:.2f} h), "
                     f"initial $P_0$={par.P0:.0f} Pa")
        fig.canvas.draw_idle()

    ax_time = fig.add_axes([.17, .135, .66, .035])
    ax_press = fig.add_axes([.17, .075, .66, .035])
    initial = state["solution"]
    time_slider = Slider(ax_time, "Time (s)", 0,
                         float(initial["time_s"][-1]), valinit=0,
                         valstep=float(initial["time_s"][1]-initial["time_s"][0]))
    pressure_slider = Slider(ax_press, "Initial Xe pressure (Pa)",
                             P_MIN, P_MAX, valinit=PARAMS.P0,
                             valstep=500.0)

    def update_time(value):
        sol = state["solution"]
        redraw(sol, nearest_index(sol, value))

    def update_pressure(value):
        # The pressure slider changes P0 and recalculates the dimensional
        # solution. Dimensionless profiles are unchanged for this linear model.
        sol = solve_xenon(PARAMS, P0=value)
        state["solution"] = sol
        redraw(sol, nearest_index(sol, time_slider.val))

    time_slider.on_changed(update_time)
    pressure_slider.on_changed(update_pressure)

    reset_ax = fig.add_axes([.86, .075, .08, .04])
    reset = Button(reset_ax, "Reset")
    reset.on_clicked(lambda event: (time_slider.reset(), pressure_slider.reset()))

    p = PARAMS
    print("Hard-coded model parameters:")
    print(f"R={p.R:g} m, L={p.L:g} m, A={p.A:g} m^2, V_g={p.V_g:g} m^3")
    print(f"D_l={p.D_l:g} m^2/s, D_t={p.D_t:g} m^2/s")
    print(f"lambda_l={p.lambda_l:g}, lambda_t={p.lambda_t:g}")
    print(f"delta={p.delta:.6g}, beta_l={p.beta_l:.6g}, beta_t={p.beta_t:.6g}")
    print(f"Liquid-tissue interface: z_i=L={p.z_interface:g} m")
    print("Time slider moves through the precomputed solution.")
    print("Pressure slider changes initial headspace Xe partial pressure and reruns.")
    redraw(initial, 0)
    plt.show()


if __name__ == "__main__":
    main()
