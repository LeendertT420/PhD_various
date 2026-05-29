import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, CheckButtons
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec

from N_osc_eqs_w_HO import *

# ============================================================
# SETTINGS
# ============================================================
verbose = False
np.set_printoptions(precision=4)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

gamma = 0.05
colors = ['r', 'b', 'g', 'm']

# ============================================================
# PARAMETER GRIDS (BIFURCATION MAP)
# ============================================================
delta_min, delta_max = -5, 2
alpha_min, alpha_max = 0, 1

N_delta = 100
N_alpha = 100

deltas = np.linspace(delta_min, delta_max, N_delta)
deltas_eff = np.linspace(0, delta_max, N_delta)
alphas = np.linspace(alpha_min, alpha_max, N_alpha)

D, A = np.meshgrid(deltas, alphas, indexing='ij')
Z = np.zeros((N_delta, N_alpha))

D_eff, A = np.meshgrid(deltas_eff, alphas, indexing='ij')
Z_eff = np.zeros((N_delta, N_alpha))

N_min, N_max = 3, 25

# ============================================================
# INITIAL PARAMETERS
# ============================================================
N0 = 3
T0 = 100

params = {
        'N': N0,
        'sigma': 20,
        'tau': 1.0,
        'alpha': 0,
        'delta': 0,
        'gamma': np.full(N0, 0.05),
        'mu': mu_spectrum(N0),
        'chi': np.load('./tensors/chi_ijk.npy'),
    }


# ============================================================
# INITIAL CONDITIONS
# ============================================================
x0 = np.zeros(N0)
y0 = np.zeros(N0)
z0 = 0.0

# ============================================================
# FIGURE LAYOUT
# ============================================================
fig = plt.figure(figsize=(16, 10))
fig.subplots_adjust(bottom=0.32)

gs = GridSpec(2, 4, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])  # bifurcation
ax2 = fig.add_subplot(gs[1, 0])  # eigenvalues
ax3 = fig.add_subplot(gs[0, 1])  # time trace
ax4 = fig.add_subplot(gs[1, 1])  # threshold panel
ax5 = fig.add_subplot(gs[0, 2])
ax6 = fig.add_subplot(gs[1, 2])
ax7 = fig.add_subplot(gs[0, 3])
ax8 = fig.add_subplot(gs[1, 3])

# ============================================================
# ============================================================
# 1. BIFURCATION PANEL (ax1)
# ============================================================
# background
ax1.fill_between(deltas, alpha_min, alpha_max, color='lightblue', alpha=0.3)

# curves
bif_lower, = ax1.plot([], [], 'k', lw=2)
bif_upper, = ax1.plot([], [], 'k', lw=2)

lasing_lines = [ax1.plot([], [], 'r', lw=1)[0] for _ in range(4 * N_max - 2)]
point, = ax1.plot([], [], 'ko')

# heatmap (FIXED ORIENTATION)
mesh = ax1.pcolormesh(D, A, Z, shading='auto', cmap='viridis')
cbar = plt.colorbar(mesh, ax=ax1)
cbar.set_label(r'$X^*$')

ax1.set_xlim(delta_min, delta_max)
ax1.set_ylim(alpha_min, alpha_max)
ax1.set_xlabel(r'$\delta$')
ax1.set_ylabel(r'$\alpha$')
ax1.set_title("Bifurcation diagram")
ax1.grid()

# ============================================================
# 2. THRESHOLD PANEL (ax2)
# ============================================================
bif_lower_eff, = ax2.plot([], [], 'k', lw=2)
bif_upper_eff, = ax2.plot([], [], 'k', lw=2)

lasing_lines_eff = [ax2.plot([], [], 'r', lw=1)[0] for _ in range(4 * N_max - 2)]
point_eff, = ax2.plot([], [], 'ko')

# heatmap (FIXED ORIENTATION)
mesh_eff = ax2.pcolormesh(D_eff, A, Z_eff, shading='auto', cmap='viridis')
cbar_eff = plt.colorbar(mesh_eff, ax=ax2)
cbar_eff.set_label(r'$X^*$')

ax2.set_xlabel(r'$\delta_{\mathrm{eff}}$')
ax2.set_ylabel(r'$\alpha_c$')
ax2.set_xlim(0, delta_max)
ax2.set_ylim(alpha_min, alpha_max)
ax2.set_title("Lasing thresholds")

# ============================================================
# 3. EIGENVALUE PANEL (ax2)
# ============================================================
scatters = [ax4.scatter([], [], color=c, s=10) for c in ['r', 'g', 'b']]

ax4.axhline(0, color='gray')
ax4.axvline(0, color='gray')

ax4.set_xlim(-1, 1)
ax4.set_ylim(-1, 18)
ax4.set_title("Eigenvalues")
ax4.set_xlabel('Real')
ax4.set_ylabel('Imaginary')

# ============================================================
# 4. TIME TRACE PANEL (ax3)
# ============================================================
traj1, = ax3.plot([], [], 'k')
offsets = [ax3.axhline(y=0, color=c) for c in ['r', 'g', 'b']]

ax3.set_title("(X, t)")
ax3.set_xlabel(r'$t$')
ax3.set_ylabel(r'$NX$')

# ============================================================
# 4. TIME TRACE PANEL (ax3)
# ============================================================
traj2, = ax7.plot([], [], 'k')
offsets2 = [ax7.axhline(y=0, color=c) for c in ['r', 'g', 'b']]

ax7.set_title("(X, t)")
ax7.set_xlabel(r'$t$')
ax7.set_ylabel(r'$NX$')

# ============================================================
# 4. TIME TRACE PANEL (ax3)
# ============================================================

fourier, = ax8.plot([], [], 'k')
spectrum = ax8.vlines([], ymin=0, ymax=1, color='grey', alpha=0.5)

ax8.set_title("fourier")
ax8.set_xlabel(r'$freq$')
ax8.set_ylabel(r'$amp$')

# ============================================================
# 4. MANIFOLDS
# ============================================================

xbasis = ax5.imshow(
    np.zeros((N0+1, N0+2)),
    aspect='auto',
    origin='lower',
    cmap='viridis'
)

xprimebasis = ax6.imshow(
    np.zeros((2*N0+1, N0+2)),
    aspect='auto',
    origin='lower',
    cmap='viridis'
)

labels = [f"x{i+1}" for i in range(N0)] + ["z", "X"]

ax5.set_xticks(range(N0+2))
ax5.set_xticklabels(labels, rotation=90)

ax6.set_xticks(range(N0+2))
ax6.set_xticklabels(labels, rotation=90)

# ============================================================
# SLIDERS
# ============================================================
box = Rectangle((0.05, 0.05), 0.9, 0.15,
                transform=fig.transFigure,
                fill=False, linewidth=2)
fig.patches.append(box)


def make_slider_with_box(x, y, label, vmin, vmax, vinit, step=None):
    # slider axis
    ax_slider = fig.add_axes([x, y, 0.15, 0.02])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit, valstep=step)

    # textbox axis (to the right of slider)
    ax_box = fig.add_axes([x + 0.154, y - 0.002, 0.06, 0.03])
    textbox = TextBox(ax_box, "", initial=str(vinit))

    # sync: slider → box
    def slider_update(val):
        textbox.set_val(f"{val:.4g}")

    slider.on_changed(slider_update)

    # sync: box → slider
    def box_submit(text):
        try:
            val = float(text)
            if vmin <= val <= vmax:
                slider.set_val(val)
        except ValueError:
            pass  # ignore invalid input

    textbox.on_submit(box_submit)

    return slider, textbox


sA, boxA = make_slider_with_box(0.1, 0.15, r'$\alpha$', alpha_min, alpha_max, params['alpha'])
sD, boxD = make_slider_with_box(0.1, 0.11, r'$\delta$', delta_min, delta_max, params['delta'])
sT, boxT = make_slider_with_box(0.4, 0.15, r'$\tau$', 0.1, 5, params['tau'])
sN, boxN = make_slider_with_box(0.4, 0.11, r'$N$', N_min, N_max, N0, step=1)
sS, boxS = make_slider_with_box(0.7, 0.15, r'$\sigma$', 10, 40, params['sigma'], step=.5)
sTime, boxTime = make_slider_with_box(0.7, 0.11, r'$T$', 1, 500, T0)

sx0, boxx0 = make_slider_with_box(0.1, 0.07, r'$x_0$', -5, 5, 0)
sy0, boxy0 = make_slider_with_box(0.4, 0.07, r'$y_0$', -5, 5, 0)
sz0, boxz0 = make_slider_with_box(0.7, 0.07, r'$z_0$', -5, 5, 0)


# ============================================================
# CHECKBOX
# ============================================================
ax_check = plt.axes([0.85, 0.16, 0.1, 0.04])   # [left, bottom, width, height]

check = CheckButtons(
    ax_check,
    ['Show cmap'],
    [False]      # initially unabled
)

def toggle_cmap(label, mesh, cbar):
    state = check.get_status()[0]

    mesh.set_visible(state)
    cbar.ax.set_visible(state)

    fig.canvas.draw()

# ============================================================
# UPDATE FUNCTION
# ============================================================
def update(val, update_solver=False, update_thresholds=False):

    if verbose: print("Updating...")

    #for scat, offset in zip(scatters, offsets):
    #    scat.set_offsets([[]])
    #    offset.set_ydata([])
    N = int(sN.val)
    params['alpha'] = sA.val
    params['delta'] = sD.val
    params['tau'] = sT.val
    params['sigma'] = sS.val
    params['N'] = N
    params['mu'] = mu_spectrum(N)
    params['gamma'] = np.full(N, gamma)

    T = sTime.val

    x0 = sx0.val
    y0 = sy0.val
    z0 = sz0.val


    
    # --------------------------------------------------------
    # BIFURCATION CURVES
    # --------------------------------------------------------
    bif_lower.set_data(deltas, lower_boundary(N, deltas))
    bif_upper.set_data(deltas, upper_boundary(N, deltas))
    bif_lower.set_zorder(10)
    bif_upper.set_zorder(10)
    if update_thresholds:
        pass
    '''
        thresholds = lasing_threshold(N, deltas, tau, mus, gammas)

        if len(thresholds) != 0:
            for i, line in enumerate(lasing_lines):
                if i < len(thresholds):
                    line.set_data(deltas, thresholds[i])
                else:
                    line.set_data([], [])

            


            thresholds_eff = lasing_threshold(N, deltas, tau, mus, gammas,
                                            as_func_off='delta_eff',
                                            delta_effs=deltas_eff)

            for i, line in enumerate(lasing_lines_eff):
                if i < len(thresholds_eff):
                    line.set_data(np.linspace(0, delta_max, N_delta), thresholds_eff[i])
                else:
                    line.set_data([], [])
        
        else:
            for line, line_eff in zip(lasing_lines, lasing_lines_eff):
                line.set_data([], [])
                line_eff.set_data([], [])

    delta_temp = deltas_eff - N*np.max(z_star(N, alpha, delta))
    bif_lower_eff.set_data(deltas_eff, lower_boundary(N, delta_temp))
    bif_upper_eff.set_data(deltas_eff, upper_boundary(N, delta_temp))
    bif_lower_eff.set_zorder(10)
    bif_upper_eff.set_zorder(10)'''

        

    point.set_data([params['delta']], [params['alpha']])

    '''point_eff.set_data(
        [N * np.max(z_star_numba(N, alpha, delta)) + delta],
        [alpha]
        )'''

    # --------------------------------------------------------
    # EIGENVALUES
    # --------------------------------------------------------
    roots, eigvals, eigvecs = compute_eigs(params)
    print(len(roots), roots)
    #print(roots, eigvals, eigvecs)
    idx = np.argsort(np.real(roots))[::-1]
    #print(idx)
    #eigvals = eigvals[idx]
    #eigvecs = eigvecs[idx]

    for i, (root, vals, vecs, scat) in enumerate(zip(roots, eigvals, eigvecs, scatters)):
        scat.set_offsets(np.c_[vals.real, vals.imag])

        # remove all negative imaginary eigenvalues
        '''mask = vals.imag >= -1e-8

        # Filter
        vals = vals[mask]
        vecs = vecs[:, mask]

        assert len(vals) == N + 1

        # Sort by magnitude of imaginary part
        idx = np.argsort(np.abs(vals.imag))

        vals = vals[idx]
        vecs = vecs[:, idx]

        xbasis_matrix = np.zeros((len(vals), N+1))
        xprimebasis_matrix = np.zeros((len(vals), N+1))


        for k in range(len(vals)):

            # Both basis representations
            vectors = [
                vecs[:, k],
                vecs[:, k]
            ]

            for matrix, vec in zip([xbasis_matrix, xprimebasis_matrix], vectors):

                if np.abs(np.imag(vals[k])) > 1e-10:
                    # 2D eigenspace
                    basis = np.column_stack((np.real(vec), np.imag(vec)))

                    # Orthonormalize
                    Q, _ = np.linalg.qr(basis)

                    # Projection norm of e_i onto plane
                    row = np.sqrt(Q[:N, 0]**2 + Q[:N, 1]**2)

                else:
                    # 1D eigenspace
                    u = np.real(vec)
                    u /= np.linalg.norm(u)

                    # Projection norm of e_i onto line
                    row = np.abs(u[:N])

                matrix[k, :] = row


        xbasis.set_data(xbasis_matrix)
        xbasis.set_extent((-.5, N + .5, -.5, N + .5))
        xbasis.set_clim(vmin=xbasis_matrix.min(), vmax=xbasis_matrix.max())

        ax5.set_yticks(np.arange(N + 1))
        ax5.set_xticks(np.arange(N + 1))

        y_labels = []
        for i in range(N + 1):
            label = r'$\lambda$'
            for digit in list(str(i)):
                label += rf'$_{digit}$'
            y_labels.append(label)

        x_labels = []
        for i in range(1, N + 1):
            label = r'$x$'
            for digit in list(str(i)):
                label += rf'$_{digit}$'
            x_labels.append(label)
        x_labels += [r'$z$']

        ax5.set_yticklabels(y_labels)
        ax5.set_xticklabels(x_labels, rotation=45, ha='right')


        xprimebasis.set_data(xprimebasis_matrix)
        xprimebasis.set_extent((-.5, N + .5, -.5, N + .5))
        xprimebasis.set_clim(vmin=xprimebasis_matrix.min(), vmax=xprimebasis_matrix.max())

        ax6.set_yticks(np.arange(N + 1))
        ax6.set_xticks(np.arange(N + 1))

        x_labels = [r'$X$']
        for i in range(2, N + 1):
            label = r'$u$'
            for digit in list(str(i)):
                label += rf'$_{digit}$'
            x_labels.append(label)
        x_labels += [r'$z$']

        ax6.set_yticklabels(y_labels)
        ax6.set_xticklabels(x_labels, rotation=45, ha='right')'''

    # --------------------------------------------------------
    # TIME EVOLUTION
    # --------------------------------------------------------
    if update_solver:
        
    
        X0 = np.concatenate([np.full(N, x0), np.full(N, y0), [z0]])
        print(X0)
        t_eval = np.linspace(0, T, 5000)

        sol = solve_ivp(
            lambda t, X: system(t, X, params),
            (0, T), X0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9
        )
        print(sol)
        X = np.sum(sol.y[:N, :], axis=0)

        traj1.set_data(t_eval, X)
        #roots_sorted = np.sort(fixed_points_num(params))[::-1]
        #for offset, root in zip(offsets, roots_sorted):
        #    offset.set_ydata([N * root])

        ax3.set_xlim(0, T)
        ax3.set_ylim(np.min(X), np.max(X))

        
        t_eval = np.linspace(T, T+200, 5000)

        sol = solve_ivp(
            lambda t, X: system(t, X, params),
            (T, T+200), sol.y[:,-1], t_eval=t_eval
        )

        X = np.sum(sol.y[:N, :], axis=0)

        traj2.set_data(t_eval, X)
        #roots_sorted = np.sort(z_star_numba(N, alpha, delta))[::-1]
        #for offset, root in zip(offsets, roots_sorted):
        #    offset.set_ydata([N * root])
        
        ax7.set_xlim(T,T+20)
        ax7.set_ylim(np.min(X), np.max(X))
        


        fft_values = np.fft.rfft(X-np.mean(X))
        frequencies = np.fft.rfftfreq(len(X), d=200/5000)

        amplitude = 2.0 * np.abs(fft_values) / len(X)
        amplitude[0] /= 2.0
        if len(X) % 2 == 0:
            amplitude[-1] /= 2.0

        fourier.set_data(frequencies*2*np.pi, amplitude)
        segments = [[[x, 0], [x, np.max(amplitude)]] for x in np.sqrt(params['mu'])]
        spectrum.set_segments(segments)

        ax8.set_xlim(0, np.max(np.sqrt(params['mu']))+np.mean(np.diff(np.sqrt(params['mu']))))
        ax8.set_ylim(0, np.max(amplitude))
        


    '''
    mesh.set_visible(check.get_status()[0])
    cbar.ax.set_visible(check.get_status()[0])
    mesh_eff.set_visible(check.get_status()[0])
    cbar_eff.ax.set_visible(check.get_status()[0])'''


    fig.canvas.draw_idle()

# ============================================================
# CONNECT SLIDERS
# ============================================================
sA.on_changed(lambda val: update(val,
                                 update_solver=True,))
sD.on_changed(lambda val: update(val,
                                 update_solver=True))
sN.on_changed(lambda val: update(val,
                                 update_solver=True,
                                 update_thresholds=True))
sT.on_changed(lambda val: update(val,
                                 update_solver=True,
                                 update_thresholds=True))
sTime.on_changed(lambda val: update(val,
                                 update_solver=True))
sx0.on_changed(lambda val: update(val,
                                 update_solver=True))
sy0.on_changed(lambda val: update(val,
                                 update_solver=True))
sz0.on_changed(lambda val: update(val,
                                 update_solver=True))


# initial draw
update(None, update_solver=True, update_thresholds=True)

plt.show()