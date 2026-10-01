import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
import time
import sys
import json
import os
from datetime import datetime
import logging
from scipy.special import jn_zeros, jnp_zeros, jn
from typing import Tuple
from scipy.interpolate import interp1d 
from scipy.signal import savgol_filter, hilbert
from tqdm import tqdm 

# Configure the logger
LOG_FORMAT = '[OptomechFreqCombsSim] %(asctime)s [%(levelname)s]: %(message)s'
LOG_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# Set the log level (e.g., INFO, DEBUG, WARNING, ERROR)
LOG_LEVEL = logging.DEBUG

# Create a logger
logger = logging.getLogger('OptomechFreqCombsSim_logger')
logger.setLevel(LOG_LEVEL)

# Create a formatter with the custom time format
formatter = logging.Formatter(LOG_FORMAT, LOG_TIME_FORMAT)

# Create a StreamHandler to display log messages on the console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

hbar = 1.055e-34  # J seconds
c = 2.998 * 1e8 #m/s
k_B = 1.380649e-23  # Boltzmann constant in J/K
RHO = 145
ALPHA_VDW = 2.6e-24

def third_sound_speed(superfluid_ratio, alpha_vdw, d):
    return np.sqrt(3 * superfluid_ratio * alpha_vdw / (d**3))
    

def omega_m(m, n, c_3, R, boundary = "free"):
    if boundary == "free":
        chi = jnp_zeros(m, n)[-1]
    if boundary == "fixed":
        chi = jn_zeros(m, n)[-1]


    return (chi * c_3)/R

def calculate_eff_mass(m, n, dens_ratio, R, d, rho):
    zeta = jnp_zeros(m, n)[-1]
    r = np.linspace(0, R, int(1e5))
    eta_r = jn(0, zeta * r/R)
    int_value = np.trapezoid(eta_r**2 * r, r)
    return dens_ratio * (R/d)**2 * 1/zeta**2 * 2 * np.pi * rho * d * int_value / (eta_r[-1]**2)

def calc_zpf(m, n, hbar, alpha_silica, m_eff, d, R):
    c_3 = third_sound_speed(
        superfluid_ratio = 1,
        alpha_vdw = alpha_silica,
        d = d
    )

    zeta = jnp_zeros(m, n)[-1]
    omega = zeta * c_3 / R
    return np.sqrt(hbar / (2 * m_eff * (omega/2 * np.pi)))

def get_beta_2_from_sim(omegas, beta_curve, target_frequency, kind='linear'):
    interpolator = interp1d(omegas, beta_curve, kind=kind, fill_value="extrapolate")
    return interpolator(target_frequency)

def calc_k(r, eta_i, R_probe, d):
    probe_idx = np.argmin(np.abs(r - R_probe))
    eta_probe = eta_i[probe_idx]
    prefactor = 3 * RHO * ALPHA_VDW / d**4 * 2 * np.pi
    integrand = np.trapezoid((eta_i / eta_probe)**2 * r, r)
    return prefactor * integrand

def calc_beta_self(r, eta_i, R_probe, d):
    probe_idx = np.argmin(np.abs(r - R_probe))

    eta_probe = eta_i[probe_idx]

    prefactor = -6 * RHO * ALPHA_VDW / d**5 * 2 * np.pi
    integrand = np.trapezoid((eta_i / eta_probe)**3 * r, r)
    return prefactor * integrand

def calc_beta_pair(r, eta_i, eta_j, R_probe, d):
    probe_idx = np.argmin(np.abs(r - R_probe))

    eta_probe_i = eta_i[probe_idx]
    eta_probe_j = eta_j[probe_idx]

    prefactor = -6 * RHO * ALPHA_VDW / d**5 * 2 * np.pi
    integrand = np.trapezoid((eta_i / eta_probe_i)**2 * (eta_j / eta_probe_j) * r, r)
    return prefactor * integrand

def calc_beta_triplet(r, eta_i, eta_j, eta_k, R_probe, d):
    probe_idx = np.argmin(np.abs(r - R_probe))
    
    eta_probe_i = eta_i[probe_idx]
    eta_probe_j = eta_j[probe_idx]
    eta_probe_k = eta_k[probe_idx]

    prefactor = -6 * RHO * ALPHA_VDW / d**5 * 2 * np.pi
    integrand = np.trapezoid((eta_i / eta_probe_i) * (eta_j / eta_probe_j) * (eta_k / eta_probe_k) * r, r)
    return prefactor * integrand

def calc_beta_four(r, eta_i, eta_j, eta_k, eta_l, R_probe, d):
    probe_idx = np.argmin(np.abs(r - R_probe))

    eta_probe_i = eta_i[probe_idx]
    eta_probe_j = eta_j[probe_idx]
    eta_probe_k = eta_k[probe_idx]
    eta_probe_l = eta_l[probe_idx]

    prefactor = 10 * RHO * ALPHA_VDW / d**6 * 2 * np.pi
    integrand = np.trapezoid((eta_i / eta_probe_i) * (eta_j / eta_probe_j) * (eta_k / eta_probe_k) * (eta_l / eta_probe_l) * r, r)
    return prefactor * integrand


def compute_beta_tensors(r, etas, R_probe, d):
    """
    etas: list of mode profiles [eta_1, eta_2, ..., eta_N]
    Returns:
        beta_self: (N,) array
        beta_pair: (N,N) array, beta_pair[i,j] for x_i^2 x_j
        beta_triplet: (N,N,N) array, beta_triplet[i,j,k] for x_i x_j x_k (i<j<k)
    """
    N = len(etas)
    
    beta_self = np.zeros(N)
    beta_pair = np.zeros((N, N))
    beta_triplet = np.zeros((N, N, N))
    beta_fourth = np.zeros((N, N, N, N))

    # Compute self-cubic
    for i in tqdm(range(N)):
        beta_self[i] = calc_beta_self(r, etas[i], R_probe, d)
    
    # Compute pairwise cubic (i != j)
    for i in tqdm(range(N)):
        for j in range(N):
            if i != j:
                beta_pair[i, j] = calc_beta_pair(r, etas[i], etas[j], R_probe, d)
    
    # Compute triplet cubic coefficients for i<j<k
    # Here we assume symmetry: beta_triplet[i,j,k] = beta_ijk = (overlap integral of eta_i*eta_j*eta_k)
    for i in tqdm(range(N)):
        for j in range(i+1, N):
            for k in range(j+1, N):
                beta_triplet[i,j,k] = calc_beta_triplet(r, etas[i], etas[j], etas[k], R_probe, d)

    for i in tqdm(range(N)):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    beta_fourth[i, j, k, l] = calc_beta_four(r,etas[i], etas[j], etas[k], etas[l], R_probe, d)

    return beta_self, beta_pair, beta_triplet, beta_fourth

def get_bessel_profile(R, N_r, n, boundary):
    m = 0
    if boundary == "free":
        zeta_mn = jnp_zeros(m, n)[-1]  # nth root of J_m
    elif boundary == "fixed":
        zeta_mn = jn_zeros(m, n)[-1]  # nth root of J_m

    r = np.linspace(0, R, N_r)

    return r, jn(0, zeta_mn * r/R)

class OptomechFreqCombSim:
    def __init__(self,
                 l: float,
                 kappa: float,
                 kappa_ext: float,
                 ms: int,
                 ns: int,
                 omegas: float,
                 gammas: float,
                 Gs: float,
                 masses: float,
                 T_bath: float,
                 taus: float,
                 beta_1s: list,
                 beta_2s: list,
                #  beta_self,
                #  beta_pair,
                #  beta_triplet,
                #  beta_fourth,
                 vdw_order:int,
                 etas_scaled_T,
                 W3,
                 W4,              
                 A: float,
                 detuning: Tuple[str, float],
                 input_power: Tuple[str, float],
                 fraction_left: float,
                 sim_time: int,
                 N_eval: int,
                 init_cond: list,
                 plot=True,
                 save=True,
                 directory=None,):

        self.l = l
        self.kappa = kappa
        self.kappa_ext = kappa_ext

        self.ms = ms
        self.ns = ns

        self.omegas = omegas
        self.gammas = gammas
        self.Gs = Gs
        self.masses = masses

        self.T_bath = T_bath
        self.sqrt_2Gm_kBTs = np.sqrt(
            2 * self.gammas * self.masses * k_B * self.T_bath)
        self.taus = taus
        self.beta_1s = beta_1s
        self.beta_2s = beta_2s

        self.vdw_order = vdw_order
        assert self.vdw_order < 5

        self.etas_scaled_T = etas_scaled_T
        self.W3 = W3
        self.W4 = W4

        self.A = A

        self.detuning_type = detuning[0]
        self.Delta = detuning[1]
        self.Delta_start = self.Delta

        self.power_type = input_power[0]
        self.input_power = input_power[1]
        self.fraction_left = fraction_left
        self.power_switched = False

        self.s_in = np.sqrt(self.input_power / (hbar * 2 * np.pi * c/self.l))
        self.s_in_start = self.s_in

        self.sim_time = sim_time
        self.N_eval = N_eval
        self.init_cond = init_cond
        self.N_calls = 0
        self.time_span = np.array([0.0, self.sim_time])
        self.time_eval = np.linspace(
            self.time_span[0], self.time_span[1], N_eval)

        self.plot = plot
        self.save = save
        self.directory = directory

        return

    def simulate(self):
        logger.info("Initializing simulation...")

        self.last_call = time.time()
        self.start_time = time.time()

        x_scale = np.array([1] * len(self.ns))
        v_scale = np.array([1] * len(self.ns))
        z_scale = np.array([1e3])

        # combine scales in the same order as init_cond
        scales = np.concatenate([x_scale, v_scale, z_scale])

        # absolute tolerance proportional to scale
        atol = 1e-9 * scales

        solution = solve_ivp(
            self.ode_solver,
            self.time_span,
            self.init_cond,
            t_eval=self.time_eval,
            method="RK45",
            # rtol=1e-6,
            # atol=atol,
        )

        self.end_time = time.time()

        logger.info("Simulation succesfully completed!")

        if self.plot:
            logger.info("Plotting data...")
            self.plot_data(solution)

        if self.save:
            logger.info("Saving data...")
            self.save_data(solution)

        return

    def plot_data(self, solution):
        N = len(self.masses)
        START = 0
        STOP = None
        time_vals = solution.t[START:STOP]
        x = solution.y[:N]  # positions
        v = solution.y[N:2*N]  # velocities
        z = solution.y[-1][START:STOP]


        # Detunings & powers
        detunings = np.array([get_detuning_numba(t, time_vals[-1]) for t in time_vals])
        powers = np.array([get_power_numba(t, time_vals[-1]) for t in time_vals])

        # ----------------- Compute mechanical phases -----------------
        phases = np.zeros_like(x)
        for i in range(N):
            phases[i] = np.unwrap(np.arctan2(-v[i], omegas[i]*x[i]))
        # Optional: phase differences between first mode and others
        phase_diffs = phases - phases[0]

        dot_Gx = 0.0
        for i in range(N):
            dot_Gx += Gs[i] * x[i]

        detuning_eff = detunings + dot_Gx
        intensity = (kappa_ext * s_in**2) / (0.25 * kappa**2 + detuning_eff**2)

        # ----------------- Begin plotting -----------------
        fig = plt.figure(figsize=(12, 8))
        grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[2, 1])

        # Optical mode intensity
        ax1 = fig.add_subplot(grid[0, 0])
        ax1.plot(time_vals[::10], intensity[::10], color="black", linewidth=1)
        ax1.set_xlabel(r'$t$ [s]')
        ax1.set_ylabel(r'$|a(t)|^2$')
        ax1.set_title("Intracavity optical field")
        ax1.grid()

        # Quadrature plot
        ax2 = fig.add_subplot(grid[0, 1])
        ax2.set_xlabel(r'$I$')
        ax2.set_ylabel(r'$dI/dt$')
        ax2.set_title("Quadrature")
        ax2.grid()

        # Mechanical modes (Second row, left, 2/3 width)
        ax3 = fig.add_subplot(grid[1, 0], sharex=ax1)

        x_tot = np.zeros_like(x[0][START:STOP])

        for i, (m, n) in enumerate(zip(self.ms, self.ns)):
            ax3.plot(time_vals[::10], -x[i][START:STOP][::10], label=f"Mode ({m}, {n})", linewidth=1, alpha = 0.5)
            x_tot += x[i][START:STOP]

        ax3.plot(time_vals[::10], -x_tot[::10], color = "black", linewidth = 1)
        ax3.set_xlabel(r'$t$')
        ax3.set_ylabel(r'$x(t) \ (\AA)$')
        ax3.set_title("Mechanical Modes")
        # ax3.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')        
        ax3.grid()

        # Thermal response
        axz = fig.add_subplot(grid[1, 1], sharex=ax1)
        axz.plot(time_vals[::10], z[::10], color="red", linewidth=1)
        axz.set_xlabel(r'$t$ [s]')
        axz.set_ylabel(r'$z(t)$')
        axz.set_title("Photothermal Response")
        axz.grid()

        # Optical phase space
        ax4 = fig.add_subplot(grid[2, 1])
        # ax4.plot(a_real[::10], a_imag[::10], color="blue", linewidth=1)
        ax4.set_xlabel(r'Re(a)')
        ax4.set_ylabel(r'Im(a)')
        ax4.set_title("Optical Phase Space")
        ax4.grid()

        # FFT of optical mode
        ax5 = fig.add_subplot(grid[2, 0])
        # ax5.plot(freqs[::10]/1e3, amplitude_fft_db[::10], color="black")
        ax5.set_xlabel("Frequency [kHz]")
        ax5.set_ylabel("Magnitude [dB]")
        ax5.set_title("FFT of Optical Mode")
        ax5.grid()

        # Power & detuning
        ax6 = fig.add_subplot(grid[3, 0], sharex=ax1)
        ax6.plot(time_vals[::10], detunings[::10]/(2*np.pi*1e6), color="black")
        ax6.plot(time_vals[::10], (detunings + np.dot(Gs, x[:, START:STOP]))[::10]/(2*np.pi*1e6), color = "blue")
        ax6.axhline(0, 0, 1, color = "blue", linestyle = "--", linewidth = 1)

        ax6.set_xlabel(r'$t$ [s]')
        ax6.set_ylabel("Detuning [MHz]")
        ax6.set_title("Detuning")
        ax6.grid()
        ax7 = ax6.twinx()
        # np.sqrt(input_power / (hbar * 2 * np.pi * f))
        ax7.plot(time_vals[::10], powers[::10]**2*(hbar*2*np.pi* c/self.l)*1e6, color="red")
        ax7.set_ylabel(r"Power [$\mu$W]")

        # ----------------- Mechanical phases -----------------
        ax9 = fig.add_subplot(grid[3, 1], sharex=ax1)
        for i in range(N):
            ax9.plot(time_vals[::10], phases[i][START:STOP][::10], label=f"Mode ({self.ms[i]},{self.ns[i]})", alpha=0.7)

        ax9.set_xlabel(r'$t$ [s]')
        ax9.set_ylabel("Phase [rad]")
        ax9.set_title("Mechanical Phases (unwrapped)")
        ax9.grid()
        # ax8.legend(fontsize=8)

        # # Optional: phase differences
        # ax8 = fig.add_subplot(grid[4, 0], sharex=ax1)
        # for i in range(1, N):
        #     ax8.plot(time_vals[::10], phase_diffs[i][START:STOP][::10], label=f"ΔPhase {i}-0")
        # ax8.set_xlabel(r'$t$ [s]')
        # ax8.set_ylabel("Phase difference [rad]")
        # ax8.set_title("Mechanical Phase Differences vs Mode 0")
        # ax8.grid()
        # ax9.legend(fontsize=8)


        def update_fft_on_xlim_change(event_ax):
            # Get current x-limits from ax1
            xlim = ax1.get_xlim()


            t_min = xlim[0]
            t_max = xlim[1]

            # Find indices corresponding to the current x-limits
            indices = np.where((time_vals >= t_min) & (time_vals <= t_max))[0]

            if len(indices) > 10:  # Only update if there are enough points
                time_window = time_vals[indices]
                # a_window = a[indices]

                # Recompute FFT
                # sampling frequency
                fs_window = 1 / (time_window[1] - time_window[0])
                freqs_window = np.fft.fftshift(
                    np.fft.fftfreq(len(time_window), d=1/fs_window))
                # amplitude_fft_window = np.fft.fftshift(
                #     np.fft.fft(np.abs(a_window)))
                # amplitude_fft_db_window = 20 * \
                #     np.log10(np.abs(amplitude_fft_window))

                # Update the FFT plot
                ax5.clear()
                # ax5.plot(freqs_window / 1e3, amplitude_fft_db_window,
                #          color="black", linewidth=1)
                ax5.set_xlim(-100, 100)
                ax5.set_xlabel("Frequency (kHz)")
                ax5.set_ylabel("Magnitude (dB)")
                ax5.set_title("FFT of Optical Mode")
                ax5.grid()


                # intensity = np.abs(a_window)**2
                # dI_dt = np.gradient(intensity, time_vals)
                # smoothed_intensity = savgol_filter(intensity, window_length=int(self.N_eval / self.sim_time * 1e-5), polyorder=3)
                # dI_dt = np.gradient(smoothed_intensity, time_window)

                ax2.clear()
                # ax2.plot(intensity, dI_dt, color="black", linewidth=1)
                ax2.set_xlabel(r'$I$')
                ax2.set_ylabel(r'$dI/dt$')
                ax2.set_title("Quadrature")
                ax2.grid()

                ax4.clear()
                # ax4.plot(a_window.real, a_window.imag, color="blue", linewidth=1)
                ax4.set_xlabel(r'$\Re(a)$')
                ax4.set_ylabel(r'$\Im(a)$')
                ax4.set_title("Phase Space")
                ax4.grid()

                fig.canvas.draw_idle()

        ax1.callbacks.connect('xlim_changed', update_fft_on_xlim_change)

        plt.tight_layout()
        plt.show()
        return

    def save_data(self, solution):
        N = len(self.masses)

        time_vals = solution.t
        x = solution.y[:N] #unpack all x's and v's of different mechanical modes
        v = solution.y[N:2*N]
        z = solution.y[-1]

        x_tot = x.sum(axis = 0)
        
        try:
            os.mkdir(f"simulations/data/{self.directory}", )
        except Exception:
            logger.warning(
                f"simulations/data/{self.directory} exists, using that directory")


        detunings = []
        powers = []
        for t in time_vals:
            detunings.append(get_detuning_numba(t, time_vals[-1]))
            powers.append(get_power_numba(t, time_vals[-1]))

        dot_Gx = 0.0
        for i in range(N):
            dot_Gx += Gs[i] * x[i]

        detuning_eff = np.array(detunings) + dot_Gx
        intensity = (kappa_ext * s_in**2) / (0.25 * kappa**2 + detuning_eff**2)


        xs = []
        for i in range(x.shape[0]):
            xs.append(x[i, :].tolist())

        vs = []
        for i in range(v.shape[0]):
            vs.append(v[i, :].tolist())

        filename = datetime.now().strftime(
            f"simulations/data/{self.directory}/simulation_one_mode_%Y-%m-%d_%H-%M-%S-%f.json")

        results = {
            "parameters": {
                "l": self.l,
                "kappa": self.kappa,
                "kappa_ext": self.kappa_ext,
                "omegas": list(self.omegas),
                "gammas": list(self.gammas),
                "Gs": list(self.Gs),
                "ms": list(self.masses),
                "T_bath": self.T_bath,
                "taus": list(self.taus),
                "beta_1s": list(self.beta_1s),
                "beta_2s": list(self.beta_2s),
                "A": self.A,
                "detuning_type": str(self.detuning_type),
                "detuning": self.Delta,
                "power_type": str(self.power_type),
                "input_power": self.input_power,
                "fraction_left": self.fraction_left,
                "sim_time": self.sim_time,
                "N_eval": self.N_eval,
                "init_cond": list(self.init_cond),
            },

            "time": time_vals.tolist(),
            "intensity": intensity.tolist(),
            "x": x_tot.tolist(),
            "xs": xs,
            "vs": vs,
            "z": z.tolist(),
            "detunings": detunings,
            "powers": powers,
        }
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n Simulation results saved to {filename}")

    def ode_solver(self, t, y):
        self.N_calls += 1

        #only when the detuning or power types are dynamic we can use the above functions for
        #time-dependent detuning or powers

        if self.N_calls % 50000 == 0:
            elapsed_time = time.time() - self.start_time
            progress = (t - self.time_span[0]) / \
                (self.time_span[1] - self.time_span[0]) * 100
            estimated_total_time = elapsed_time / \
                (progress / 100) if progress > 0 else 0
            remaining_time = estimated_total_time - elapsed_time

            delta_call = time.time() - self.last_call
            self.last_call = time.time()
            power_in = (self.s_in**2) * (hbar * 2 * np.pi * f) * 1e6 #uW

            current_detuning = get_detuning_numba(t, sim_time)
            current_power = get_power_numba(t, sim_time)
            sys.stdout.write(
                f"\r {progress:.2f}% | Elapsed: {elapsed_time:.2f}s | Remaining: {remaining_time:.2f}s | time_per_call: {delta_call/50000*1e6:.2f} us | Delta: {current_detuning/(2 * np.pi * 1e6):.2f} MHz | Power: {current_power**2*(hbar * 2 * np.pi * f)*1e6:.2f} uW")
            sys.stdout.flush()
        return coupled_ode(t,
                        y,
                        self.Delta,
                        self.masses,
                        self.gammas,
                        self.omegas,
                        self.Gs,
                        self.kappa,
                        self.kappa_ext,
                        self.s_in,
                        self.taus,
                        self.beta_1s,
                        self.beta_2s,
                        self.etas_scaled_T, 
                        self.W3, 
                        self.W4,
                        self.vdw_order,
                        self.A)

@njit
def get_power_numba(t, sim_time):
    BEGIN = 67e-6
    MID   = 67e-6
    END   = 67e-6
    t1 = 0.5 * sim_time
    t2 = 1 * sim_time

    if t <= t1:
        power_uw_in = BEGIN + (MID - BEGIN) * (t / t1)
    elif t <= t2:
        power_uw_in = MID + (END - MID) * ((t - t1) / (t2 - t1))
    else:
        power_uw_in = END

    # power_uw_in *= (1 + 0.025 * np.sin(345*t + 123))

    input_power = power_uw_in
    return np.sqrt(input_power / (hbar * 2 * np.pi * f))

@njit
def get_detuning_numba(t, sim_time):
    return -2.4 * 11.45e6/2 * 2 * np.pi
    # return -20.0375e6 * 2 * np.pi

@njit(fastmath=True)
def coupled_ode(t, y, Delta, masses, gammas, omegas, Gs, 
                kappa, kappa_ext, s_in, taus, beta_1s, beta_2s, 
                etas_trans, W3, W4, vdw_order, A):
    
    Delta = get_detuning_numba(t, sim_time)
    s_in = get_power_numba(t, sim_time)

    M, N = etas_trans.shape  # M grid points, N modes
    dydt = np.empty_like(y)  # Allocate output buffer inside function

    # Unpack state
    x = y[:N]
    v = y[N:2*N]
    z = y[2*N]

    # Cavity field intensity
    dot_Gx = 0.0
    for i in range(N):
        dot_Gx += Gs[i] * x[i]
    detuning_eff = Delta + dot_Gx
    intensity = (kappa_ext * s_in**2) / (0.25 * kappa**2 + detuning_eff**2)

    # 1. dx/dt = v
    for i in range(N):
        dydt[i] = v[i]

    # 2. dz/dt
    dydt[2*N] = intensity - z / taus[0]

    # 3. Calculate spatial displacement profile X(r)
    X = etas_trans @ x 
    
    # Vectorized element-wise operations
    X2_W3 = (X**2) * W3
    X3_W4 = (X**3) * W4

    # 4. Accelerations dv/dt
    for i in range(N):
        f_i = -gammas[i] * v[i] - (omegas[i]**2) * x[i] \
              + beta_1s[i] * (hbar * Gs[i] / masses[i]) * intensity \
              + beta_2s[i] * (hbar * Gs[i] * A / (taus[0] * masses[i])) * z

        # 3rd Order Van der Waals Force Projection
        if vdw_order >= 3:
            f3_i = 0.0
            for m in range(M):
                f3_i += etas_trans[m, i] * X2_W3[m]
            f_i += f3_i / masses[i]

        # 4th Order Van der Waals Force Projection
        if vdw_order >= 4:
            f4_i = 0.0
            for m in range(M):
                f4_i += etas_trans[m, i] * X3_W4[m]
            f_i -= f4_i / masses[i]

        dydt[N + i] = f_i

    return dydt

# 1. Pre-computation helper (Run once before ODE simulation)
def prepare_spatial_vdw_weights(r, etas, R_probe, d, RHO, ALPHA_VDW):
    """
    Precomputes mode matrices and spatial integration weights once.
    """
    etas_arr = np.asarray(etas) # Shape (N, M)
    probe_idx = np.argmin(np.abs(r - R_probe))
    
    # Mode profiles scaled at probe position: shape (M, N)
    etas_scaled_T = (etas_arr / etas_arr[:, probe_idx, np.newaxis]).T 
    
    # Integration weights w(r) * r
    w = np.empty_like(r)
    w[0] = 0.5 * (r[1] - r[0])
    w[-1] = 0.5 * (r[-1] - r[-2])
    w[1:-1] = 0.5 * (r[2:] - r[:-2])
    W_r = w * r
    
    # Scaled prefactor weights for 3rd and 4th order spatial integrals
    pref3 = -6.0 * RHO * ALPHA_VDW / (d**5) * 2.0 * np.pi
    pref4 = 10.0 * RHO * ALPHA_VDW / (d**6) * 2.0 * np.pi
    
    W3 = pref3 * W_r * x_0
    W4 = pref4 * W_r * (x_0**2)
    
    return etas_scaled_T, W3, W4

if __name__ == "__main__":
    l = 1064e-9 #nm
    f = c/l #Hz

    R = 3e-3 #m
    L_undercut = 100e-6 #m
    # d = 18.31e-9 #m
    d = 10e-9
    x_0 = 1e-10
    vdw_order = 4

    c_3 = third_sound_speed(superfluid_ratio = 1, alpha_vdw = ALPHA_VDW, d = d)

    #Mechanical modes
    ns = list(np.arange(1,16, 1))
    # ns = np.array([7, 13])
    ms = [0]*len(ns)

    omegas = []
    gammas = []
    masses = []
    Gs = []
    taus = []
    beta_1s = []
    beta_2s = []

    # beta_2_curve = np.loadtxt("simulations\\fp_functions_100mk.txt")
    etas = []

    for i, (m, n) in enumerate(zip(ms, ns)):
        # if i == 0:
        omega = omega_m(m = m, n = n, c_3 = c_3, R = R)
        # else:
            # omega = omegas[0] + (i*150*2*np.pi)
        gamma = 2 * np.pi * 10
        mass = calculate_eff_mass(m = m, n = n, dens_ratio = 1, R = R, d = d, rho = RHO)
        # mass = 1e-3
        G = -2 * np.pi * 20e15 * x_0
        tau = 200e-6
        beta_1 = 1.0
        # beta_2 = get_beta_2_from_sim(omegas = beta_2_curve[0, :], beta_curve = beta_2_curve[1, :], target_frequency = omega)
        # beta_2 = 5e6
        beta_2 = 1.5e6
        
        r, eta = get_bessel_profile(
            R = R,
            N_r = int(1e3),
            n = n,
            boundary = "free"
        )
        etas.append(eta)

        omegas.append(omega)
        gammas.append(gamma)
        masses.append(mass)
        Gs.append(G)
        taus.append(tau)
        beta_1s.append(beta_1/(x_0**2))
        beta_2s.append(beta_2/(x_0**2))


        print(fr"Effective mass for mode ({m}, {n}) @ {omega/(2 * np.pi):.3f} Hz: {mass*1e3:.3f}g. beta = {beta_2:.2e}. tau = {tau:.2e}")

    omegas = np.array(omegas)
    gammas = np.array(gammas)
    masses = np.array(masses)
    Gs = np.array(Gs)
    taus = np.array(taus)
    beta_1s = np.array(beta_1s)
    beta_2s = np.array(beta_2s)

    step = ns[1] - ns[0]

    print(f"Precomputation...")
    etas_scaled_T, W3, W4 = prepare_spatial_vdw_weights(
        r = r,
        etas = etas,
        R_probe=R,
        d = d,
        RHO = RHO,
        ALPHA_VDW=ALPHA_VDW
    )

    # System Parameters
    kappa = 2 * np.pi * 11.45e6  # (rad/s)
    kappa_ext = kappa * 0.5 #assume critical coupling

    T_bath = 100e-3  # Bath temperature in Kelvin

    A = 1  # Absorption factor

    detuning = -kappa/2   # (rad / s)
    input_power = 7.5e-6  # Watt

    fraction_left = 0.5
    input_power = input_power * np.sqrt(fraction_left)
    s_in = np.sqrt(input_power / (hbar * 2 * np.pi * f))

    sim_time = 0.5  # seconds
    N_eval = int(250e3)

    # Full initial condition: [Re(a), Im(a), x_is, v_is, z]
    init_cond = []

    for i in range(len(ns)):
        init_cond += [((2 * np.random.random()) - 1)*1e-12 * 1/x_0] #append x random init position
        # init_cond += [0.0]
    for i in range(len(ns)):
        init_cond += [0.0] # 0 initial velocity

    init_cond += [0.0] #z = 0 initial condition

    print(f"Initial condition: {init_cond}, {len(init_cond)}")

    det_type = "dynamic"
    power_type = "dynamic"

    prev_noise = 0
    prev_time = 0

    sim = OptomechFreqCombSim(
        l=l,
        kappa=kappa,
        kappa_ext=kappa_ext,
        ms = ms,
        ns = ns,
        omegas=omegas,
        gammas=gammas,
        Gs=Gs,
        masses=masses,
        T_bath=T_bath,
        taus=taus,
        beta_1s=beta_1s,
        beta_2s=beta_2s,
        # beta_self=beta_self,
        # beta_pair=beta_pair,
        # beta_triplet=beta_triplet,
        # beta_fourth = beta_fourth,
        vdw_order = vdw_order,
        etas_scaled_T = etas_scaled_T, 
        W3 = W3, 
        W4 = W4,
        A=A,
        detuning=(det_type, detuning),
        input_power=(power_type, input_power),
        fraction_left=fraction_left,
        sim_time=sim_time,
        N_eval=N_eval,
        init_cond=init_cond,
        plot=True,
        save=False,
        directory="stable_mode_lock"
    )

    # if det_type == "dynamic" or power_type == "dynamic":
    #     ts = np.linspace(0, sim_time, N_eval)

    #     dets = []
    #     powers = []

    #     for t in ts:
    #         if det_type == "static":
    #             d = detuning 
    #         else:
    #             d = get_detuning_numba(t, sim_time)

    #         if power_type == "static":
    #             p = input_power
    #         else:
    #             p = get_power_numba(t, sim_time)
            
    #         dets.append(d)
    #         powers.append(p)

    #     dets = np.array(dets)
    #     powers = np.array(powers)

    #     fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)

    #     ax1.plot(ts, dets/1e6)
    #     ax2.plot(ts, powers**2*(hbar * 2 * np.pi * f)*1e6)

    #     ax1.set_xlabel("Time (s)")
    #     ax1.set_ylabel("Detuning (MHz)")
    #     ax2.set_ylabel(r"Power ($\mu$W)")
    #     fig.suptitle("Planned detuning & power curves")
    #     plt.show()

    sim.simulate()