import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from equations import *
from scipy.integrate import solve_ivp
from tqdm import tqdm


# ============================================================
# APS / REVTeX Standard Style Configuration
# ============================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.figsize': (6.8, 8.5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})





# PREPARE SIMULATION
N = 15
M = int(1e3)
use_3d = True
use_4d = True

profiles = []
for i in range(N):
    r, profile = get_bessel_profile(i+1, N_points=M)
    profiles.append(profile)

profiles = np.array(profiles).T

W3, W4 = prepare_spatial_vdw_weights(r)

baseconfig_SI = {'N': N,
             'Gammas': np.ones(N)*10,
             'tau': 1/5000,
             'power': 200e-6,
             'detuning': -3e6,
             'd': 18e-9}

baseconfig = to_unitless(baseconfig_SI)
baseconfig['xi'] = np.ones(N)



rng = np.random.default_rng(42)
u0 = np.zeros(2*N+1)
u0 = .1*rng.random(2*N+1)

T_equil = 1000
T_eval = 1000
T_plot1 = 800
T_plot2 = 20

t_span = (0.0, T_eval + T_equil)
dt = 1 / (20 * np.sqrt(mu_spectrum(N)[-1]))
N_points = int((T_eval + T_equil) / dt)
t_eval_full = np.linspace(0.0, T_eval + T_equil, N_points)


deltas = [-4,   0,    -1,  3,   -2]
alphas = [0.67, 0.03, .1, 0.8, 0.145]
sigmas = [60,   60,   60, 60,  33]

labels_left = ['a', 'b', 'c', 'd', 'e']
labels_right = ['f', 'g', 'h', 'i', 'j']

spectrum = np.sqrt(mu_spectrum(N))
print(np.diff(spectrum))


print('HIUERO', np.abs(np.mean(np.exp(1j*np.diff(spectrum)))))

# --- FIGURE SETUP WITH GRIDSPEC FOR BROKEN X-AXIS ---
fig = plt.figure()
# Columns 0 & 1 for split Left plot (e.g. 1.5 : 1 ratio), Column 2 for FFT
gs = fig.add_gridspec(5, 3, width_ratios=[1.5, 0.8, 1], wspace=0.05, hspace=0.1)

line_height = None
line_fft = None

for i, (delta, alpha, sigma) in tqdm(enumerate(zip(deltas, alphas, sigmas))):
    
    config = baseconfig
    config['delta'] = delta
    config['alpha'] = alpha
    config['sigma'] = sigma

    config_SI = to_SI(config, verbose=False)
    config_SI['Gammas'] = np.ones(N)*10
    config_SI['tau'] = 1/5000

    config = to_unitless(config_SI, verbose=False)
    config['xi'] = np.ones(N)

    fixed_points = fixed_points_num(system_numba_hidde,
                                    (config, profiles, W3, W4, use_3d, use_4d),
                                    num_tries=100)

    offset = np.sum(fixed_points[-1][:N])

    sol = solve_ivp(
        fun=system_numba_hidde,
        t_span=t_span,
        y0=u0,
        args=(config, profiles, W3, W4, use_3d, use_4d),
        method='RK45',
        t_eval=t_eval_full
    )

    x = sol.y[:N, :]
    y = sol.y[N:2*N, :]
    h_t = np.sum(x, axis=0)

    # ----------------------------------------------------
    # 1. FILM HEIGHT (SPLIT X-AXIS: LEFT COLUMN)
    # ----------------------------------------------------
    ax_left1 = fig.add_subplot(gs[i, 0])
    ax_left2 = fig.add_subplot(gs[i, 1], sharey=ax_left1)

    # Initial transient
    idx_init = t_eval_full <= T_plot1
    line_height, = ax_left1.plot(t_eval_full[idx_init], h_t[idx_init], c='cornflowerblue', label='Film Displacement')
    ax_left1.hlines(offset, 0, T_plot1, color='darkblue', linestyle='--', alpha=0.5)

    # Limiting behavior
    idx_late = t_eval_full >= (T_equil + T_eval - T_plot2)
    ax_left2.plot(t_eval_full[idx_late], h_t[idx_late], c='cornflowerblue')
    ax_left2.hlines(offset, T_equil + T_eval - T_plot2, T_equil + T_eval, color='darkblue', linestyle='--', alpha=0.5)

    # Hide adjacent vertical spines
    ax_left1.spines['right'].set_visible(False)
    ax_left2.spines['left'].set_visible(False)
    ax_left2.tick_params(left=False, labelleft=False)

    # Equal-sized 45-degree slash break marks
    slash_size_in = 0.05 
    bbox1 = ax_left1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox2 = ax_left2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    dx1, dy1 = slash_size_in / bbox1.width, slash_size_in / bbox1.height
    dx2, dy2 = slash_size_in / bbox2.width, slash_size_in / bbox2.height

    kwargs1 = dict(transform=ax_left1.transAxes, color='k', clip_on=False, lw=0.8)
    ax_left1.plot((1 - dx1, 1 + dx1), (-dy1, +dy1), **kwargs1)
    ax_left1.plot((1 - dx1, 1 + dx1), (1 - dy1, 1 + dy1), **kwargs1)

    kwargs2 = dict(transform=ax_left2.transAxes, color='k', clip_on=False, lw=0.8)
    ax_left2.plot((-dx2, +dx2), (-dy2, +dy2), **kwargs2)
    ax_left2.plot((-dx2, +dx2), (1 - dy2, 1 + dy2), **kwargs2)

    # Subplot Label (a-d)
    ax_left1.text(0.05, 0.88, fr'({labels_left[i]}) $(\delta,\,\alpha,\,\sigma) = (${delta}$,\,${alpha}$,\,${sigma}$)$', transform=ax_left1.transAxes, 
                   fontsize=9, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    # ----------------------------------------------------
    # 2. FOURIER TRANSFORM (RIGHT COLUMN)
    # ----------------------------------------------------
    ax_right = fig.add_subplot(gs[i, 2])
    
    idx_fft = t_eval_full >= T_equil
    t_fft = t_eval_full[idx_fft]
    h_fft = h_t[idx_fft]

    h_centered = h_fft - np.mean(h_fft)
    fft_vals = np.fft.rfft(h_centered * np.hanning(len(h_centered)))
    freqs = np.fft.rfftfreq(len(h_centered), d=(t_fft[1] - t_fft[0]))
    fft_mag = np.abs(fft_vals)

    f_max = spectrum[-1] + np.mean(np.diff(spectrum))
    i_max = np.argmin(np.abs(freqs - f_max))
    freqs = freqs[:i_max]
    fft_mag = fft_mag[:i_max]

    line_fft, = ax_right.plot(2*np.pi*freqs, fft_mag, c='indianred', label='FFT Magnitude')
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position('right')
    
    # Subplot Label (e-h)
    ax_right.text(0.05, 0.88, f'({labels_right[i]})', transform=ax_right.transAxes, 
                   fontsize=9, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    ax_right.vlines(spectrum, 0, 1.05*np.max(fft_mag), color='maroon', linestyle='--', alpha=0.5, zorder=100)
    ax_right.set_ylim(0, np.max(fft_mag)*1.05)
    ax_right.set_xlim(0, f_max)

    # ----------------------------------------------------
    # X-AXIS TICKS & LABELS FORMATTING
    # ----------------------------------------------------
    if i < 4:
        ax_left1.tick_params(bottom=False, labelbottom=False)
        ax_left2.tick_params(bottom=False, labelbottom=False)
        ax_right.tick_params(bottom=False, labelbottom=False)
    else:
        ax_left1.set_xlabel(r'Time [$1/\Omega_1$]')
        ax_left2.set_xlabel(r'Time [$1/\Omega_1$]')
        ax_right.set_xlabel(r'Angular Frequency [$\Omega_1$]')

# --- FIGURE LEVEL ANNOTATIONS ---
fig.suptitle('Film Displacement Dynamics', fontsize=10, fontweight='bold', y=0.91)

fig.text(0.065, 0.5, r'Film Displacement $\ \eta(r,\,t)$ [$\kappa/G$]', va='center', ha='center', rotation='vertical', fontsize=10)
fig.text(0.98, 0.5, 'Fourier Magnitude', va='center', ha='center', rotation=-90, fontsize=10)

line_X_star = Line2D([0], [0], color='darkblue', linestyle='--', alpha=0.5, lw=1.0)
line_omega = Line2D([0], [0], color='maroon', linestyle='--', alpha=0.5, lw=1.0)

fig.legend(
    handles=[line_height, line_fft, line_X_star, line_omega],
    labels=[r'Film Displacement $\eta(r,\,t)$', r'Fourier Magnitude', r'$X^*$', r'$\omega_i$'],
    loc='lower center',
    #bbox_to_anchor=(0.5, 0.0),
    ncol=4,
    frameon=True,
    edgecolor='black',
    fontsize=8,
)

#plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.96])
plt.savefig('fig2.pdf', dpi=300)
plt.show()