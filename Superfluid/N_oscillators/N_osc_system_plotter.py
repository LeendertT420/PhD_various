import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, CheckButtons
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec

from N_osc_eqs import *

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

N_min, N_max = 2, 20

# ============================================================
# INITIAL PARAMETERS
# ============================================================
alpha0, delta0 = 0, 0
tau0 = 1.0
N0 = 2
T0 = 100

mus = mu_spectrum(N0)
gammas = np.full(N0, gamma)

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
box = Rectangle((0.05, 0.05), 0.9, 0.1,
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


sA, boxA = make_slider_with_box(0.1, 0.11, 'α', alpha_min, alpha_max, alpha0)
sD, boxD = make_slider_with_box(0.1, 0.07, r'$\delta$', delta_min, delta_max, delta0)

sT, boxT = make_slider_with_box(0.4, 0.11, 'τ', 0.1, 5, tau0)
sN, boxN = make_slider_with_box(0.4, 0.07, 'N', N_min, N_max, N0, step=1)
sTime, boxTime = make_slider_with_box(0.7, 0.07, 'T', 1, 500, T0)


# ============================================================
# CHECKBOX
# ============================================================
ax_check = plt.axes([0.85, 0.11, 0.1, 0.04])   # [left, bottom, width, height]

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
def update(val, update_cmap=False, update_cmap_eff=True, update_solver=False, update_thresholds=False):

    if verbose: print("Updating...")

    #for scat, offset in zip(scatters, offsets):
    #    scat.set_offsets([[]])
    #    offset.set_ydata([])

    alpha = sA.val
    delta = sD.val
    tau = sT.val
    N = int(sN.val)
    T = sTime.val
    # --------------------------------------------------------
    # SYSTEM PARAMETERS
    # --------------------------------------------------------
    mus = mu_spectrum(N)
    print(mus, np.sqrt(mus))
    gammas = np.full(N, gamma)
    

    # --------------------------------------------------------
    # BIFURCATION HEATMAP
    # --------------------------------------------------------
    if update_cmap:
        for i in range(N_alpha):
            for j in range(N_delta):
                Z[j, i] = np.max(z_star_numba(N, A[j, i], D[j, i]))

        mesh.set_array(Z.ravel(order='C'))
        mesh.set_clim(min([Z.min(), Z_eff.min()]), max([Z.max(), Z_eff.max()]))

    if update_cmap_eff:
        for i in range(N_alpha):
            for j in range(N_delta):
                Z_eff[j, i] = z_star_eff(A[j, i], D_eff[j, i])

        mesh_eff.set_array(Z_eff.ravel(order='C'))
        mesh_eff.set_clim(min([Z.min(), Z_eff.min()]), max([Z.max(), Z_eff.max()]))

    
    # --------------------------------------------------------
    # BIFURCATION CURVES
    # --------------------------------------------------------
    if update_thresholds:
        thresholds = lasing_threshold(N, deltas, tau, mus, gammas)

        if len(thresholds) != 0:
            for i, line in enumerate(lasing_lines):
                if i < len(thresholds):
                    line.set_data(deltas, thresholds[i])
                else:
                    line.set_data([], [])

            bif_lower.set_data(deltas, lower_boundary(N, deltas))
            bif_upper.set_data(deltas, upper_boundary(N, deltas))
            bif_lower.set_zorder(10)
            bif_upper.set_zorder(10)


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
    bif_upper_eff.set_zorder(10)

        

    point.set_data([delta], [alpha])

    point_eff.set_data(
        [N * np.max(z_star_numba(N, alpha, delta)) + delta],
        [alpha]
        )

    # --------------------------------------------------------
    # EIGENVALUES
    # --------------------------------------------------------
    roots, eigvals, eigvecs = compute_eigs(N, mus, alpha, delta, tau, gammas)
    idx = np.argsort(np.real(roots))[::-1]

    eigvals = eigvals[idx]
    eigvecs = eigvecs[idx]

    for i, (root, vals, vecs, scat) in enumerate(zip(roots, eigvals, eigvecs, scatters)):
        scat.set_offsets(np.c_[vals.real, vals.imag])

        # remove all negative imaginary eigenvalues
        mask = vals.imag >= -1e-8

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

        x_idx = np.arange(0, 2*(N+1), 2)

        Tinv = inverse_transform_matrix(N)   # compute once outside loop

        for k in range(len(vals)):

            # Both basis representations
            vectors = [
                Tinv @ vecs[:, k],
                vecs[:, k]
            ]

            for matrix, vec in zip([xbasis_matrix, xprimebasis_matrix], vectors):

                if np.abs(np.imag(vals[k])) > 1e-10:
                    # 2D eigenspace
                    basis = np.column_stack((np.real(vec), np.imag(vec)))

                    # Orthonormalize
                    Q, _ = np.linalg.qr(basis)

                    # Projection norm of e_i onto plane
                    row = np.sqrt(Q[x_idx, 0]**2 + Q[x_idx, 1]**2)

                else:
                    # 1D eigenspace
                    u = np.real(vec)
                    u /= np.linalg.norm(u)

                    # Projection norm of e_i onto line
                    row = np.abs(u[x_idx])

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
        ax6.set_xticklabels(x_labels, rotation=45, ha='right')

    # --------------------------------------------------------
    # TIME EVOLUTION
    # --------------------------------------------------------
    if update_solver:
        y0 = np.zeros(2 * N + 1)
        t_eval = np.linspace(0, T, 5000)

        sol = solve_ivp(
            lambda t, X: system_numba(t, X, alpha, delta, mus, gammas, tau),
            (0, T), y0, t_eval=t_eval
        )

        X = np.sum(sol.y[:-1:2, :], axis=0)

        traj1.set_data(t_eval, X)
        roots_sorted = np.sort(z_star_numba(N, alpha, delta))[::-1]
        for offset, root in zip(offsets, roots_sorted):
            offset.set_ydata([N * root])

        ax3.relim()
        ax3.autoscale_view()


        t_eval = np.linspace(0, 50, 5000)

        sol = solve_ivp(
            lambda t, X: system_numba(t, X, alpha, delta, mus, gammas, tau),
            (0, 50), sol.y[:,-1], t_eval=t_eval
        )

        X = np.sum(sol.y[:-1:2, :], axis=0)

        traj2.set_data(t_eval, X)
        #roots_sorted = np.sort(z_star_numba(N, alpha, delta))[::-1]
        #for offset, root in zip(offsets, roots_sorted):
        #    offset.set_ydata([N * root])

        ax7.relim()
        ax7.autoscale_view()


        fft_values = np.fft.rfft(X-np.mean(X))
        frequencies = np.fft.rfftfreq(len(X), d=50/5000)

        amplitude = 2.0 * np.abs(fft_values) / len(X)
        amplitude[0] /= 2.0
        if len(X) % 2 == 0:
            amplitude[-1] /= 2.0

        fourier.set_data(frequencies, amplitude)

        ax8.set_xlim(0, 3)
        ax8.set_ylim(0, 3)
        



    mesh.set_visible(check.get_status()[0])
    cbar.ax.set_visible(check.get_status()[0])
    mesh_eff.set_visible(check.get_status()[0])
    cbar_eff.ax.set_visible(check.get_status()[0])


    fig.canvas.draw_idle()

# ============================================================
# CONNECT SLIDERS
# ============================================================
sA.on_changed(lambda val: update(val,
                                 update_solver=True,))
sD.on_changed(lambda val: update(val,
                                 update_solver=True))
sN.on_changed(lambda val: update(val,
                                 update_cmap=True,
                                 update_solver=True,
                                 update_thresholds=True))
sT.on_changed(lambda val: update(val,
                                 update_solver=True,
                                 update_thresholds=True))
sTime.on_changed(lambda val: update(val,
                                 update_solver=True))


# initial draw
update(None, update_cmap=True, update_cmap_eff=True, update_solver=True, update_thresholds=True)

plt.show()