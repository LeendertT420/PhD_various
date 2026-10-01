import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


chi = np.load('tensors/chi_ijk.npy')
N_file = np.shape(chi)[0]
N = 20

if N_file <= N:
    N = N_file

chi = chi[:N, :N, :N]


fig = plt.figure(figsize=(14, 10))

# 1. chi_iii
ax1 = fig.add_subplot(2, 2, 1)
modes = np.arange(1, N + 1)
chi_iii = np.einsum('iii->i', chi)
ax1.plot(modes, chi_iii, 'o-', color='crimson', lw=2, markersize=7, label=r'$\chi_{iii}$')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.set_xlabel('Mode index $i$')
ax1.set_ylabel(r'$\chi_{iii}$')
ax1.set_title(r'1. Diagonal Tensor Elements $\chi_{iii}$')
ax1.set_xticks(modes)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. chi_ijj 2D heatmap
ax2 = fig.add_subplot(2, 2, 2)
# chi_ijj: i along y-axis, j along x-axis
chi_ijj = np.einsum('ijj->ij', chi)
im = ax2.imshow(chi_ijj, cmap='seismic', origin='lower', extent=[0.5, N+0.5, 0.5, N+0.5])
cbar = fig.colorbar(im, ax=ax2)
cbar.set_label(r'$\chi_{ijj}$')
ax2.set_xlabel('Mode index $j$')
ax2.set_ylabel('Mode index $i$')
ax2.set_title(r'2. 2D Heatmap of $\chi_{ijj}$')
ax2.set_xticks(modes)
ax2.set_yticks(modes)

# 3. 3D block cut in half using voxels
ax3 = fig.add_subplot(2, 2, 3, projection='3d')

# Define a boolean mask for cut in half (e.g. k <= N//2 or k + j <= N)
# Let's say cut along k-axis: k < N//2 or i >= j >= k
I, J, K = np.indices((N, N, N))
# Mask: remove one corner / half block, e.g., i + j + k > 1.2 * N or k > N//2
filled = (I >= J) & (J >= K)  # cut away top-front quadrant
# Standardize colormap mapping for chi_ijk
norm_val = np.max(np.abs(chi))
colors = plt.cm.seismic((chi + norm_val) / (2 * norm_val))

# Set alpha on cut voxels
colors[..., 3] = 0.85

ax3.voxels(filled, facecolors=colors, edgecolors='k', linewidth=0.3)
ax3.set_xlabel('Mode $i$')
ax3.set_ylabel('Mode $j$')
ax3.set_zlabel('Mode $k$')
ax3.set_title(r'3. 3D Cut-Away Block of $\chi_{ijk}$')
ax3.view_init(elev=25, azim=-50)

# 4. Additional: Slices of chi_ijk for different k
ax4 = fig.add_subplot(2, 2, 4)
chi_k0 = chi[:, :, 0] # k = 1 (index 0)
im4 = ax4.imshow(chi_k0, cmap='viridis', origin='lower', extent=[0.5, N+0.5, 0.5, N+0.5])
cbar4 = fig.colorbar(im4, ax=ax4)
cbar4.set_label(r'$\chi_{ij1}$')
ax4.set_xlabel('Mode index $j$')
ax4.set_ylabel('Mode index $i$')
ax4.set_title(r'4. Off-Diagonal Slice $\chi_{ij,k=1}$')
ax4.set_xticks(modes)
ax4.set_yticks(modes)

plt.tight_layout()
plt.savefig('chi_ijk_visualizations.png', dpi=150)
plt.show()
print("Saved visualization figure successfully!")