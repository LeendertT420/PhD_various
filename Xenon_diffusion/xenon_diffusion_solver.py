"""Finite-volume solver for xenon diffusion: gas headspace, liquid column,
and spherical organoid. Requires numpy and scipy.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Parameters:
    # Geometry [SI]
    R: float = 1.0e-3
    L: float = 5.0e-3
    A: float = 2.5e-5
    V_g: float = 1.25e-7
    # Conditions and properties
    T: float = 310.15
    P0: float = 101325.0
    D_l: float = 1.55e-9
    D_t: float = 1.00e-9       # provisional effective tissue diffusivity
    lambda_l: float = 0.083
    lambda_t: float = 0.175    # provisional tissue/liquid partition proxy
    # Numerical settings
    N_l: int = 100
    N_t: int = 60
    tau_end: float = 12.0
    n_output: int = 500
    rtol: float = 2e-7
    atol: float = 1e-9

    @property
    def V_l(self): return self.A * self.L

    @property
    def V_t(self): return 4*np.pi*self.R**3/3

    @property
    def delta(self): return self.D_l*self.R**2/(self.D_t*self.L**2)

    @property
    def beta_l(self): return self.lambda_l*self.V_l/self.V_g

    @property
    def beta_t(self): return self.lambda_t*self.V_t/self.V_l

    @property
    def q(self):
        # Original idealized model: liquid domain extends to x=1 (z=L).
        return 1+self.beta_l+self.beta_l*self.beta_t

    @property
    def z_interface(self): return self.L

    @property
    def x_interface(self): return 1.0

    @property
    def time_scale(self): return self.R**2/self.D_t


def solve_xenon(params=None, P0=None):
    """Return solution arrays; P0, if supplied, is initial Xe partial pressure [Pa]."""
    p = Parameters() if params is None else params
    if P0 is not None:
        p = replace(p, P0=float(P0))
    if not (p.R > 0 and p.L > 0):
        raise ValueError("Require positive organoid radius and liquid height.")
    if p.N_l < 3 or p.N_t < 3:
        raise ValueError("Use at least 3 cells per domain.")

    nl, nt = p.N_l, p.N_t
    xi = p.x_interface
    dx = xi/nl
    x_faces = np.linspace(0, xi, nl+1)
    x = 0.5*(x_faces[:-1]+x_faces[1:])
    rho_faces = np.linspace(0, 1, nt+1)
    rho = 0.5*(rho_faces[:-1]+rho_faces[1:])
    shell_vol = (rho_faces[1:]**3-rho_faces[:-1]**3)/3
    dr = 1/nt
    dl_half, dt_half = dx/2, dr/2
    delta, bl, bt = p.delta, p.beta_l, p.beta_t
    K = 3*bt/delta

    # State vector: gas pressure ratio, liquid cell averages, tissue cell averages.
    y0 = np.zeros(1+nl+nt)
    y0[0] = 1.0

    def rhs(tau, y):
        pg = y[0]
        cl = y[1:1+nl]
        ct = y[1+nl:]

        # Gas-liquid boundary, c_l(0,tau)=p(tau).
        grad0 = (cl[0]-pg)/dl_half
        J0 = -delta*grad0                 # flux in +x direction

        # Common liquid/tissue interface concentration u, from flux continuity.
        u = (cl[-1]/dl_half + K*ct[-1]/dt_half)/(1/dl_half+K/dt_half)
        grad_liq_i = (u-cl[-1])/dl_half
        grad_tis_s = (u-ct[-1])/dt_half
        Ji = -delta*grad_liq_i

        # Liquid finite-volume balances.
        J = np.empty(nl+1)
        J[0], J[-1] = J0, Ji
        if nl > 1:
            J[1:-1] = -delta*np.diff(cl)/dx
        dcl = -(J[1:]-J[:-1])/dx

        # Spherical tissue finite-volume balances; radial outward flux=-dc/drho.
        G = np.zeros(nt+1)
        G[-1] = -grad_tis_s
        if nt > 1:
            G[1:-1] = -(rho_faces[1:-1]**2)*np.diff(ct)/dr
        dct = -(G[1:]-G[:-1])/shell_vol

        dpg = bl*delta*grad0
        return np.concatenate(([dpg], dcl, dct))

    tau_eval = np.linspace(0, p.tau_end, p.n_output)
    sol = solve_ivp(rhs, (0, p.tau_end), y0, method="BDF",
                    t_eval=tau_eval, rtol=p.rtol, atol=p.atol)
    if not sol.success:
        raise RuntimeError(sol.message)

    pg = sol.y[0]
    cl = sol.y[1:1+nl].T
    ct = sol.y[1+nl:].T

    # Flux-matched concentration at the liquid-tissue interface at each time.
    # Reconstruct using the adjacent liquid and tissue cell-centre values.
    interface_conc_dimless = (
        cl[:, -1]/dl_half + K*ct[:, -1]/dt_half
    ) / (1/dl_half + K/dt_half)

    Rgas = 8.31446261815324
    scale_l = p.lambda_l*p.P0/(Rgas*p.T)
    scale_t = p.lambda_t*scale_l
    time_s = sol.t*p.time_scale

    # Total xenon inventory. The liquid domain spans x in [0, 1], consistent
    # with V_l=A*L and the original idealized interface at z=L.
    inventory = (pg + bl*np.sum(cl, axis=1)*dx
                 + 3*bl*bt*np.sum(ct*shell_vol, axis=1))

    return {
        "parameters": p, "tau": sol.t, "time_s": time_s,
        "time_h": time_s/3600, "x": x, "z_m": x*p.L,
        "rho": rho, "r_m": rho*p.R,
        "p_dimless": pg, "pressure_pa": pg*p.P0,
        "c_liquid_dimless": cl, "c_tissue_dimless": ct,
        "C_liquid_molm3": cl*scale_l, "C_tissue_molm3": ct*scale_t,
        "c_interface_dimless": interface_conc_dimless,
        "C_interface_liquid_molm3": interface_conc_dimless*scale_l,
        "C_interface_tissue_molm3": interface_conc_dimless*scale_t,
        "interface_concentration_percent_equilibrium":
            100*p.q*interface_conc_dimless,
        "liquid_average_dimless": cl.mean(axis=1),
        "tissue_average_dimless": np.sum(ct*shell_vol,axis=1)/np.sum(shell_vol),
        "inventory_diagnostic": inventory,
        "equilibrium_dimless": 1/p.q,
        "equilibrium_pressure_pa": p.P0/p.q,
    }


if __name__ == "__main__":
    s = solve_xenon()
    p = s["parameters"]
    print(f"delta={p.delta:.6g}, beta_l={p.beta_l:.6g}, beta_t={p.beta_t:.6g}")
    print(f"Interface: z_i=L={p.z_interface:.6g} m; x_i={p.x_interface:.6g}")
    print(f"Diffusion time scale R^2/D_t={p.time_scale:.3f} s")
    print(f"Predicted equilibrium pressure={s['equilibrium_pressure_pa']:.3f} Pa")
    print(f"Final pressure={s['pressure_pa'][-1]:.3f} Pa")
    print("Final interface concentration: "
          f"{s['C_interface_liquid_molm3'][-1]:.6g} mol/m^3 "
          f"({s['interface_concentration_percent_equilibrium'][-1]:.3f}% of equilibrium)")
    print(f"Inventory diagnostic range: {s['inventory_diagnostic'].min():.6f} to "
          f"{s['inventory_diagnostic'].max():.6f}")
