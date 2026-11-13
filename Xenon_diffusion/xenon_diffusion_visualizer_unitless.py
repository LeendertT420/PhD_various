"""Unitless interactive visualizer for xenon_solver.py.

Horizontal coordinates: z/L, r/R, and tau = D_t*t/R^2.
Vertical quantities: percentage of their equilibrium value.
For the original model c_eq = 1/q, q = 1 + beta_l + beta_l*beta_t,
so plotted percentages are 100*q times the dimensionless solver outputs.

Requires numpy, scipy, matplotlib. Run from the directory containing
xenon_diffusion_solver.py: python xenon_diffusion_visualizer_unitless.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from xenon_diffusion_solver import Parameters, solve_xenon

PARAMS = Parameters(
    R=1.0e-3, L=5.0e-3, A=2.5e-5, V_g=1.25e-7,
    T=310.15, P0=101325.0,
    D_l=1.55e-9, D_t=1.00e-9,
    lambda_l=0.083, lambda_t=0.175,
    N_l=100, N_t=60, tau_end=12.0, n_output=500,
)
P_REF_PA = 101325.0


def main():
    state = {"solution": solve_xenon(PARAMS)}
    fig = plt.figure(figsize=(13, 8))
    fig.subplots_adjust(left=.09, right=.97, top=.90, bottom=.24,
                        hspace=.42, wspace=.30)
    ax_l = fig.add_subplot(221)
    ax_t = fig.add_subplot(222)
    ax_p = fig.add_subplot(223)
    ax_avg = fig.add_subplot(224)
    # Live numerical readout for the selected time (fractions of equilibrium).
    readout = fig.text(0.09, 0.015, "", family="monospace",
                       fontsize=8.5, va="bottom")

    def nearest_index(sol, tau):
        return int(np.argmin(np.abs(sol["tau"] - tau)))

    def redraw(sol, i):
        par = sol["parameters"]
        tau = sol["tau"][i]
        t_s = sol["time_s"][i]
        q = par.q

        # Since c_eq = 1/q, percentage of equilibrium is 100*q*c.
        liquid_pct = 100.0 * q * sol["c_liquid_dimless"][i]
        tissue_pct = 100.0 * q * sol["c_tissue_dimless"][i]
        pressure_pct = 100.0 * q * sol["p_dimless"]
        liquid_mean_pct = 100.0 * q * sol["liquid_average_dimless"]
        tissue_mean_pct = 100.0 * q * sol["tissue_average_dimless"]

        # Fractions of equilibrium (1.0 means equilibrium).
        cl_now = sol["c_liquid_dimless"][i]
        ct_now = sol["c_tissue_dimless"][i]
        p_fraction = q * sol["p_dimless"][i]
        liquid_gas_side = q * cl_now[0]
        liquid_interface = q * cl_now[-1]
        liquid_mean = q * sol["liquid_average_dimless"][i]
        tissue_center = q * ct_now[0]
        tissue_surface = q * ct_now[-1]
        tissue_mean = q * sol["tissue_average_dimless"][i]
        readout.set_text(
            "Selected-time values (fraction of equilibrium; 1.000 = 100%):\\n"
            f"Headspace pressure: {p_fraction:.4f}   |   "
            f"Liquid: gas side {liquid_gas_side:.4f}, interface side {liquid_interface:.4f}, "
            f"mean {liquid_mean:.4f}   |   "
            f"Tissue: center {tissue_center:.4f}, surface {tissue_surface:.4f}, "
            f"volume mean {tissue_mean:.4f}"
        )

        for ax in (ax_l, ax_t, ax_p, ax_avg):
            ax.clear()

        ax_l.plot(sol["x"], liquid_pct, lw=2)
        ax_l.axvline(1.0, ls="--", color="black",
                     label=r"Organoid interface $z/L=1$")
        ax_l.axhline(100.0, ls=":", color="gray", label="Equilibrium")
        ax_l.set(xlabel=r"Dimensionless depth $z/L$",
                 ylabel="Liquid concentration (% of equilibrium)",
                 title="Liquid concentration profile")
        ax_l.set_xlim(0, 1)
        ax_l.set_ylim(0, 100*PARAMS.q)
        ax_l.grid(alpha=.25)
        ax_l.legend(fontsize=8)

        ax_t.plot(sol["rho"], tissue_pct, lw=2)
        ax_t.axhline(100.0, ls=":", color="gray", label="Equilibrium")
        ax_t.set(xlabel=r"Dimensionless radius $r/R$",
                 ylabel="Tissue concentration (% of equilibrium)",
                 title="Organoid radial concentration profile")
        ax_t.set_xlim(0, 1)
        ax_t.set_ylim(0, 100)
        ax_t.grid(alpha=.25)
        ax_t.legend(fontsize=8)

        ax_p.plot(sol["tau"], pressure_pct, lw=2,
                  label="Headspace Xe partial pressure")
        ax_p.axvline(tau, ls="--", color="black", label="Selected time")
        ax_p.axhline(100.0, ls=":", color="gray", label="Equilibrium")
        ax_p.set(xlabel=r"Dimensionless time $\tau=D_t t/R^2$",
                 ylabel="Headspace pressure (% of equilibrium)",
                 title="Headspace xenon partial pressure")
        ax_p.grid(alpha=.25)
        ax_p.set_ylim(100, 100*PARAMS.q)
        ax_p.legend(fontsize=8)

        ax_avg.plot(sol["tau"], liquid_mean_pct, label="Liquid spatial mean")
        ax_avg.plot(sol["tau"], tissue_mean_pct, label="Tissue volume mean")
        ax_avg.axvline(tau, ls="--", color="black", label="Selected time")
        ax_avg.axhline(100.0, ls=":", color="gray", label="Equilibrium")
        ax_avg.set(xlabel=r"Dimensionless time $\tau=D_t t/R^2$",
                   ylabel="Mean concentration (% of equilibrium)",
                   title="Mean concentrations")
        ax_avg.grid(alpha=.25)
        ax_avg.set_ylim(0, 100)
        ax_avg.legend(fontsize=8)

        fig.suptitle(
            f"Unitless xenon diffusion | $\\tau$={tau:.3f} "
            f"(t={t_s:.0f} s), initial $P_0/P_{{ref}}$={par.P0/P_REF_PA:.2f}"
        )
        fig.canvas.draw_idle()

    ax_time = fig.add_axes([.17, .135, .66, .035])
    ax_press = fig.add_axes([.17, .075, .66, .035])
    initial = state["solution"]
    time_slider = Slider(
        ax_time, r"Dimensionless time $\tau$", 0.0,
        float(initial["tau"][-1]), valinit=0.0,
        valstep=float(initial["tau"][1] - initial["tau"][0])
    )
    time_text = fig.text(0.17, 0.17, "", fontsize=10)

    def update_time_label(tau):
        minutes = tau * PARAMS.R**2 / PARAMS.D_t / 60.0
        time_text.set_text(f"τ = {tau:.3f}  →  t = {minutes:.2f} min")
    pressure_slider = Slider(
        ax_press, r"Initial pressure $P_0/P_{ref}$", 0.0, 2.0,
        valinit=PARAMS.P0 / P_REF_PA, valstep=0.01
    )

    def update_time(value):
        update_time_label(value)
        redraw(state["solution"], nearest_index(state["solution"], value))

    def update_pressure(value):
        # Unitless slider value is converted to Pa only when calling the solver.
        sol = solve_xenon(PARAMS, P0=value * P_REF_PA)
        state["solution"] = sol
        redraw(sol, nearest_index(sol, time_slider.val))
        update_time_label(time_slider.val)

    time_slider.on_changed(update_time)
    pressure_slider.on_changed(update_pressure)
    reset_ax = fig.add_axes([.86, .075, .08, .04])
    reset = Button(reset_ax, "Reset")
    reset.on_clicked(lambda event: (time_slider.reset(), pressure_slider.reset()))

    p = PARAMS
    print(f"delta={p.delta:.6g}, beta_l={p.beta_l:.6g}, "
          f"beta_t={p.beta_t:.6g}, q={p.q:.6g}")
    print(f"Equilibrium dimensionless value: 1/q={1.0/p.q:.6g}")
    print("Plotted vertical values are 100*q times dimensionless solution values.")
    print("Horizontal axes use z/L, r/R, and tau=D_t*t/R^2.")
    redraw(initial, 0)
    update_time_label(0.0)
    plt.show()


if __name__ == "__main__":
    main()
