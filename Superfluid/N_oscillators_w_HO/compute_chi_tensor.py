import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import jn, jn_zeros
from tqdm import tqdm
from itertools import combinations_with_replacement, permutations


plot = False

# -----------------------------------------------------------------------------
# 1. Setup Parameters and Roots
# -----------------------------------------------------------------------------
N = 25  # Number of modes to compute
zeta = jn_zeros(1, N)

# -----------------------------------------------------------------------------
# 2. Define Stable Integrand Functions (Division inside the power)
# -----------------------------------------------------------------------------
def integrand_chi_ijk(u, zi, zj, zk):
    ratio_i = jn(0, zi * u) / jn(0, zi)
    ratio_j = jn(0, zj * u) / jn(0, zj)
    ratio_k = jn(0, zk * u) / jn(0, zk)
    return 4 * ratio_i * ratio_j * ratio_k * u

def integrand_chi_ijkl(u, zi, zj, zk, zl):
    ratio_i = jn(0, zi * u) / jn(0, zi)
    ratio_j = jn(0, zj * u) / jn(0, zj)
    ratio_k = jn(0, zk * u) / jn(0, zk)
    ratio_l = jn(0, zl * u) / jn(0, zl)
    return 20/3 * ratio_i * ratio_j * ratio_k * ratio_l * u

# -----------------------------------------------------------------------------
# 3. Compute the Tensors
# -----------------------------------------------------------------------------
print("Computing tensors...")
print(r"$\chi_{ijk}$")
chi_ijk = np.zeros((N, N, N))

for i, j, k in tqdm(combinations_with_replacement(range(N), 3), total=int(N*(N+1)*(N+2)/6)):
    val, _ = quad(integrand_chi_ijk, 0, 1, args=(zeta[i], zeta[j], zeta[k]))

    for p in set(permutations((i, j, k))):
        chi_ijk[p] = val

print(r"$\chi_{ijkl}$")
chi_ijkl = np.zeros((N, N, N, N))

for i, j, k, l in tqdm(combinations_with_replacement(range(N), 4), total=int(N * (N + 1) * (N + 2) * (N + 3) / 24)):
    val, _ = quad(integrand_chi_ijkl, 0, 1, args=(zeta[i], zeta[j], zeta[k], zeta[l]))
    
    for p in set(permutations((i, j, k, l))):
        chi_ijkl[p] = val


# -----------------------------------------------------------------------------
# 5. Save All Data to .npy Files
# -----------------------------------------------------------------------------
np.save('./tensors/chi_ijk.npy', chi_ijk)
np.save('./tensors/chi_ijkl.npy', chi_ijkl)
print("Data successfully saved to .npy files.\n")

# -----------------------------------------------------------------------------
# 6. Load Data and Plot
# -----------------------------------------------------------------------------
if plot:
    print("Reading data and generating plots...")
    loaded_chi_i = np.load('./tensors/chi_i.npy')
    loaded_chi_ij = np.load('./tensors/chi_ij.npy')
    loaded_lambda_i = np.load('./tensors/lambda_i.npy')

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
    axes[2].plot(modes, np.cumsum(loaded_lambda_i)/np.arange(1, N+1), color='crimson', linewidth=3)
    axes[2].set_title(r'Perturbation Weight $\Lambda_i$ vs. Mode Index $i$', fontsize=14)
    axes[2].set_xlabel('Mode Index $i$', fontsize=12)
    axes[2].set_ylabel(r'$\Lambda_i$', fontsize=12)
    axes[2].set_xticks(modes)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()