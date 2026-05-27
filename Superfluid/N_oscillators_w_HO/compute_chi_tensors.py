import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import jn, jn_zeros
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Setup Parameters and Roots
# -----------------------------------------------------------------------------
N = 20  # Number of modes to compute
zeta = jn_zeros(1, N)

# -----------------------------------------------------------------------------
# 2. Define Stable Integrand Functions (Division inside the power)
# -----------------------------------------------------------------------------
def integrand_chi_i(u, zi):
    ratio = jn(0, zi * u) / jn(0, zi)
    return 4 * (ratio**3) * u

def integrand_chi_ij(u, zi, zj):
    ratio_i = jn(0, zi * u) / jn(0, zi)
    ratio_j = jn(0, zj * u) / jn(0, zj)
    return 4 * (ratio_i**2) * ratio_j * u

def integrand_chi_ijk(u, zi, zj, zk):
    ratio_i = jn(0, zi * u) / jn(0, zi)
    ratio_j = jn(0, zj * u) / jn(0, zj)
    ratio_k = jn(0, zk * u) / jn(0, zk)
    return (4/3) * ratio_i * ratio_j * ratio_k * u

# -----------------------------------------------------------------------------
# 3. Compute the Tensors
# -----------------------------------------------------------------------------
print("Computing tensors...")
chi_i = np.zeros(N)
chi_ij = np.zeros((N, N))
chi_ijk = np.zeros((N, N, N))

for i in tqdm(range(N)):
    chi_i[i], _ = quad(integrand_chi_i, 0, 1, args=(zeta[i],))
    for j in range(N):
        chi_ij[i, j], _ = quad(integrand_chi_ij, 0, 1, args=(zeta[i], zeta[j]))
        for k in range(N):
            chi_ijk[i, j, k], _ = quad(integrand_chi_ijk, 0, 1, args=(zeta[i], zeta[j], zeta[k]))

# -----------------------------------------------------------------------------
# 4. Compute Lambda_i Vector
# -----------------------------------------------------------------------------
print("Computing Lambda_i structural vector...")
lambda_i = np.zeros(N)

for i in range(N):
    # Term 1: chi_i
    term1 = chi_i[i]
    
    # Term 2: 2 * sum_{j != i} chi_ij
    # Mask out the j == i element
    mask_j_not_i = np.ones(N, dtype=bool)
    mask_j_not_i[i] = False
    term2 = 2 * np.sum(chi_ij[i, mask_j_not_i])
    
    # Term 3: sum_{j != i} chi_ji
    term3 = np.sum(chi_ij[mask_j_not_i, i])
    
    # Term 4: sum_{j != k} chi_ijk
    # We create a 2D slice chi_ijk[i, :, :] and sum only where j != k
    slice_jk = chi_ijk[i, :, :]
    mask_j_not_k = ~np.eye(N, dtype=bool)
    term4 = np.sum(slice_jk[mask_j_not_k])
    
    # Combine terms
    lambda_i[i] = term1 + term2 + term3 + term4

# -----------------------------------------------------------------------------
# 5. Save All Data to .npy Files
# -----------------------------------------------------------------------------
np.save('chi_i.npy', chi_i)
np.save('chi_ij.npy', chi_ij)
np.save('chi_ijk.npy', chi_ijk)
np.save('lambda_i.npy', lambda_i)
print("Data successfully saved to .npy files.\n")

# -----------------------------------------------------------------------------
# 6. Load Data and Plot
# -----------------------------------------------------------------------------
print("Reading data and generating plots...")
loaded_chi_i = np.load('chi_i.npy')
loaded_chi_ij = np.load('chi_ij.npy')
loaded_lambda_i = np.load('lambda_i.npy')

# Create a 3-panel figure layout
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
modes = np.arange(1, N + 1)

# Plot 1: chi_i vs Mode Index i
axes[0].plot(modes, loaded_chi_i, 'o-', color='darkblue', linewidth=2, markersize=6)
axes[0].set_title(r'$\chi_i$ vs. Mode Index $i$', fontsize=14)
axes[0].set_xlabel('Mode Index $i$', fontsize=12)
axes[0].set_ylabel(r'$\chi_i$', fontsize=12)
axes[0].set_xticks(modes)
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: chi_ij as a Matrix Color Plot
cax = axes[1].imshow(loaded_chi_ij, cmap='viridis', origin='lower', 
                     extent=[0.5, N + 0.5, 0.5, N + 0.5])
axes[1].set_title(r'$\chi_{ij}$ Interaction Matrix', fontsize=14)
axes[1].set_xlabel('Mode Index $j$', fontsize=12)
axes[1].set_ylabel('Mode Index $i$', fontsize=12)
axes[1].set_xticks(modes)
axes[1].set_yticks(modes)
cbar = fig.colorbar(cax, ax=axes[1], fraction=0.046, pad=0.04)
cbar.set_label(r'Value of $\chi_{ij}$', fontsize=12)

# Plot 3: Lambda_i vs Mode Index i
axes[2].plot(modes, loaded_lambda_i, 's--', color='crimson', linewidth=2, markersize=6)
axes[2].set_title(r'Perturbation Weight $\Lambda_i$ vs. Mode Index $i$', fontsize=14)
axes[2].set_xlabel('Mode Index $i$', fontsize=12)
axes[2].set_ylabel(r'$\Lambda_i$', fontsize=12)
axes[2].set_xticks(modes)
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()