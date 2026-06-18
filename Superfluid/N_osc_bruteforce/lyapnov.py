import numpy as np
import matplotlib.pyplot as plt
from equations import *
from scipy.integrate import solve_ivp

N = 15

base_config = {'N': N,
                   'gamma': np.ones(N),
                   'mu': mu_spectrum(N),
                   'tau': 1.0,
                   'alpha': 1.0,
                   'delta': 1.0,
                   'sigma': 60}

config = to_SI(base_config, verbose=True)

config2 = {'N': N,
                   'Gammas': np.ones(N)*10,
                   'tau': 1/5000,
                   'power': 7.7e-6,
                   'detuning': -15.5e6,
                   'd': 18.31e-9}

to_unitless(config2)



use_3d = True
use_4d = True

parameters = [
    # Chaotic Regimes
    (2.0, 0.0, "chaotic"),
    (1.85, 0.245, "chaotic"),
    (1.528, -1.695, "chaotic"),
    
    # Mode-Locked Regimes
    (1.005, -0.486, "mode-locked"),
    (1.82, -3.663, "mode-locked"),
    (0.402, 1.032, "mode-locked"),
    (0.271, 0.835, "mode-locked"),
    
    # Multi-Mode Regimes
    (1.618, -2.341, "multi-mode"),
    (1.035, -3.663, "multi-mode"),
    
    # Single-Mode Regimes
    (1.075, -2.735, "single-mode"),
    (0.563, -2.735, "single-mode")
]

for alpha, delta, tag in parameters[:3]:
    params = {'N': N,
            'gamma': np.ones(N) * 0.05,
            'mu': mu_spectrum(N),
            'tau': 1.0,
            'alpha': alpha,
            'delta': delta,
            'sigma': 60.0,
            'chi_ijk': np.load('./tensors/chi_ijk.npy')[:N, :N, :N],
            'chi_ijkl': np.load('./tensors/chi_ijkl.npy')[:N, :N, :N, :N],
            'xi': np.ones(N)}


    u0 = np.concatenate([np.random.uniform(-0.5, 0.5, N),
                                np.zeros(N),
                                [0.1]])
        
        # Precompute time evaluation boundaries
    Dt_sim = 1000.0

    Dt_eval = 2*np.pi*25 / np.sqrt(params['mu'][0])

    t_span = (0.0, Dt_sim + Dt_eval)

    t_span_eval = (Dt_sim, Dt_sim + Dt_eval)

    N_points = int(10*25*np.sqrt(params['mu'][-1]/params['mu'][0]))

    t_eval = np.linspace(t_span_eval[0], t_span_eval[1], N_points)



    sol = solve_ivp(
            fun=system_numba,
            t_span=t_span,
            y0=u0,
            args=(params, use_3d, use_4d),
            method='RK45',
            t_eval=t_eval,
            jac=Jacobian_numba,
            rtol=1e-6, atol=1e-8,
            first_step=1e-5)

    epsilon = 1e-8
    init1 = sol.y[:,-1]
    init2 = init1 + epsilon*np.random.rand(2*N+1)

    tspan = (0.0, 1500)
    teval = np.linspace(tspan[0], tspan[1], int(10*tspan[1]*np.sqrt(params['mu'])[-1]/(2*np.pi)))

    sol1 = solve_ivp(
            fun=system_numba,
            t_span=tspan,
            y0=init1,  
            args=(params, use_3d, use_4d),
            method='RK45',            
            t_eval=teval,
            jac=Jacobian_numba,
            rtol=1e-6, atol=1e-8,
            first_step=1e-5)

    sol2 = solve_ivp(
            fun=system_numba,
            t_span=tspan,
            y0=init2, 
            args=(params, use_3d, use_4d),
            method='RK45',           
            t_eval=teval,
            jac=Jacobian_numba,          
            rtol=1e-6, atol=1e-8,
            first_step=1e-5)


    # ==========================================
    # 1. Distance Calculation & Data Masking
    # ==========================================
    distance = np.linalg.norm(sol2.y - sol1.y, axis=0)
    log_distance = np.log(distance)
    time = sol1.t

    saturation_idx = np.where(distance >= 0.85 * np.max(distance))[0][0]

    # To ensure we don't include the "bend" into saturation, we take the index 
    # somewhat earlier (e.g., 70% of the way to saturation index)
    fit_end_idx = int(saturation_idx * 0.80)

    # Fallback: Ensure we have at least a few points to fit
    if fit_end_idx < 5:
        fit_end_idx = min(20, len(time) - 1)

    t_fit_max = time[fit_end_idx]
    print(f"Automatically detected t_fit_max: {t_fit_max:.2f} (Index: {fit_end_idx})")

    # Filter data points where t < t_fit_max for the linear regression
    fit_mask = time <= t_fit_max
    time_fit = time[fit_mask]
    log_dist_fit = log_distance[fit_mask]

    # ==========================================
    # 2. Linear Regression (y = mx + c)
    # ==========================================
    # slope (m) is the MLE, intercept (c) is the estimated log(d0)
    slope, intercept = np.polyfit(time_fit, log_dist_fit, 1)
    mle_estimate = slope

    print(f"Calculated MLE (Slope): {mle_estimate:.4f}")

    # Generate the fitted line values over the fitted time range
    fitted_log_distance = slope * time_fit + intercept
    fitted_distance = np.exp(fitted_log_distance)

    # ==========================================
    # 3. Plotting Trajectories and Fit Line
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    X1 = np.sum(sol1.y[:N, :], axis=0)
    X2 = np.sum(sol2.y[:N, :], axis=0)

    # Left Subplot: The Trajectories (showing the divergence split)
    ax1.plot(time, X1, label='Trajectory 1', color='tab:blue', alpha=0.8)
    ax1.plot(time, X2, label='Trajectory 2', color='tab:red', alpha=0.8)
    ax1.axvline(x=t_fit_max, color='gray', linestyle=':', label=f't_fit boundary ({t_fit_max:.2f})')
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('Total film thickness')
    ax1.set_title(f'Trajectories ({tag})')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(0, 2*t_fit_max)
    ax1.legend()

    # Right Subplot: Log Distance & Linear Fit
    ax2.semilogy(time, distance, color='purple', alpha=0.4, label='Actual Distance')
    ax2.semilogy(time_fit, distance[fit_mask], color='purple', linewidth=2, label='Data used for Fit')
    ax2.semilogy(time_fit, fitted_distance, color='r', linestyle='--', linewidth=2,
                label=f'Linear Fit (Slope/MLE = {mle_estimate:.4f})')

    ax2.axvline(x=t_fit_max, color='gray', linestyle=':', label='t_fit boundary')
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('Euclidean Phase-space-distance')
    ax2.set_title('Log Distance vs. Time')
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()