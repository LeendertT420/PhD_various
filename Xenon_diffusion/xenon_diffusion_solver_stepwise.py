"""Finite-volume solver for repeated xenon additions to a sealed headspace."""
import numpy as np
from scipy.integrate import solve_ivp

from xenon_diffusion_solver import Parameters


PSI_TO_PA = 6894.757293168
GAS_CONSTANT = 8.31446261815324


def solve_xenon_stepwise(params=None, pressure_step_psi=2.0,
                         interval_min=10.0, n_steps=15,
                         samples_per_interval=30):
    """Solve diffusion with fixed-mole xenon additions at regular intervals.

    The initial xenon pressure is zero. Each addition injects enough xenon
    to raise headspace pressure by ``pressure_step_psi`` in the absence of
    gas transfer; the actual pressure subsequently changes as xenon diffuses.
    The first addition occurs after one interval. One additional interval is
    simulated after the final addition.
    """
    p = Parameters() if params is None else params
    if p.R <= 0 or p.L <= 0:
        raise ValueError("Require positive organoid radius and liquid height.")
    if p.N_l < 3 or p.N_t < 3:
        raise ValueError("Use at least 3 cells per domain.")
    if pressure_step_psi <= 0 or interval_min <= 0:
        raise ValueError("Pressure step and interval must be positive.")
    if n_steps < 1:
        raise ValueError("Use at least one pressure addition.")
    if samples_per_interval < 2:
        raise ValueError("Use at least two samples per interval.")

    nl, nt = p.N_l, p.N_t
    dx = 1.0 / nl
    x_faces = np.linspace(0.0, 1.0, nl + 1)
    x = 0.5 * (x_faces[:-1] + x_faces[1:])
    rho_faces = np.linspace(0.0, 1.0, nt + 1)
    rho = 0.5 * (rho_faces[:-1] + rho_faces[1:])
    shell_vol = (rho_faces[1:]**3 - rho_faces[:-1]**3) / 3.0
    dr = 1.0 / nt
    dl_half, dt_half = dx / 2.0, dr / 2.0
    delta, bl, bt = p.delta, p.beta_l, p.beta_t
    K = 3.0 * bt / delta

    def rhs(tau, y):
        pg = y[0]
        cl = y[1:1 + nl]
        ct = y[1 + nl:]

        grad0 = (cl[0] - pg) / dl_half
        J0 = -delta * grad0
        u = (cl[-1] / dl_half + K * ct[-1] / dt_half) / (
            1.0 / dl_half + K / dt_half
        )
        Ji = -delta * (u - cl[-1]) / dl_half

        J = np.empty(nl + 1)
        J[0], J[-1] = J0, Ji
        if nl > 1:
            J[1:-1] = -delta * np.diff(cl) / dx
        dcl = -(J[1:] - J[:-1]) / dx

        G = np.zeros(nt + 1)
        G[-1] = -(u - ct[-1]) / dt_half
        if nt > 1:
            G[1:-1] = -(rho_faces[1:-1]**2) * np.diff(ct) / dr
        dct = -(G[1:] - G[:-1]) / shell_vol
        dpg = bl * delta * grad0
        return np.concatenate(([dpg], dcl, dct))

    pressure_step_pa = pressure_step_psi * PSI_TO_PA
    interval_tau = interval_min * 60.0 / p.time_scale
    # State quantities are normalized to this fixed reference pressure.
    reference_pressure_pa = PSI_TO_PA
    pressure_jump = pressure_step_pa / reference_pressure_pa
    added_moles_per_step = pressure_step_pa * p.V_g / (GAS_CONSTANT * p.T)
    y = np.zeros(1 + nl + nt)
    tau_values = [0.0]
    states = [y.copy()]

    # Each pressure addition follows an interval of diffusion; the extra
    # interval at the end exposes the transient after the last addition.
    for interval_index in range(n_steps + 1):
        tau_start = interval_index * interval_tau
        tau_end = (interval_index + 1) * interval_tau
        tau_eval = np.linspace(tau_start, tau_end, samples_per_interval + 1)
        sol = solve_ivp(
            rhs, (tau_start, tau_end), y, method="BDF", t_eval=tau_eval,
            rtol=p.rtol, atol=p.atol,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        for sample_index in range(1, len(sol.t)):
            tau_values.append(sol.t[sample_index])
            states.append(sol.y[:, sample_index].copy())
        y = sol.y[:, -1].copy()

        if interval_index < n_steps:
            y[0] += pressure_jump
            tau_values.append(tau_end)
            states.append(y.copy())

    tau = np.asarray(tau_values)
    state = np.asarray(states)
    pg = state[:, 0]
    cl = state[:, 1:1 + nl]
    ct = state[:, 1 + nl:]

    interface_conc_dimless = (
        cl[:, -1] / dl_half + K * ct[:, -1] / dt_half
    ) / (1.0 / dl_half + K / dt_half)
    scale_l = p.lambda_l * reference_pressure_pa / (GAS_CONSTANT * p.T)
    scale_t = p.lambda_t * scale_l
    time_s = tau * p.time_scale
    inventory = (pg + bl * np.sum(cl, axis=1) * dx
                 + 3.0 * bl * bt * np.sum(ct * shell_vol, axis=1))
    equilibrium_pressure_pa = (
        n_steps * pressure_step_pa / p.q
    )

    return {
        "parameters": p,
        "tau": tau,
        "time_s": time_s,
        "time_min": time_s / 60.0,
        "x": x,
        "z_m": x * p.L,
        "rho": rho,
        "r_m": rho * p.R,
        "p_dimless": pg,
        "pressure_pa": pg * reference_pressure_pa,
        "pressure_psi": pg * reference_pressure_pa / PSI_TO_PA,
        "c_liquid_dimless": cl,
        "c_tissue_dimless": ct,
        "C_liquid_molm3": cl * scale_l,
        "C_tissue_molm3": ct * scale_t,
        "c_interface_dimless": interface_conc_dimless,
        "C_interface_liquid_molm3": interface_conc_dimless * scale_l,
        "C_interface_tissue_molm3": interface_conc_dimless * scale_t,
        "liquid_average_dimless": cl.mean(axis=1),
        "tissue_average_dimless":
            np.sum(ct * shell_vol, axis=1) / np.sum(shell_vol),
        "inventory_diagnostic": inventory,
        "equilibrium_pressure_pa": equilibrium_pressure_pa,
        "pressure_step_psi": pressure_step_psi,
        "interval_min": interval_min,
        "n_steps": n_steps,
        "added_moles_per_step": added_moles_per_step,
    }


if __name__ == "__main__":
    solution = solve_xenon_stepwise()
    print(f"Pressure additions: {solution['n_steps']} x "
          f"{solution['pressure_step_psi']:g} psi, every "
          f"{solution['interval_min']:g} min")
    print(f"Final simulated time: {solution['time_min'][-1]:g} min")
    print(f"Headspace Xe pressure at end: {solution['pressure_psi'][-1]:.3f} psi")
