import os
import numpy as np
import umap
from scipy.cluster.vq import kmeans2
from equations import *
from classification import *

# =====================================================================
# CONFIGURATION
# =====================================================================
AVAILABLE_SIGMAS = np.array([50]) # Update to match your files

def get_raw_filename(sigma_val):
    return f'./results/sweep_results_N=15_sigma={int(sigma_val)}.npz'

def get_metrics_filename(sigma_val):
    return f'./results/sweep_results_N=15_sigma={int(sigma_val)}_metrics.npz'

# =====================================================================
# PRE-COMPUTATION LOOP
# =====================================================================
print("🚀 Starting Batch Metric Extraction Pipeline...")

for sigma_val in AVAILABLE_SIGMAS:
    raw_path = get_raw_filename(sigma_val)
    out_path = get_metrics_filename(sigma_val)
    
    if not os.path.exists(raw_path):
        print(f"⚠️ Skipped: {raw_path} not found.")
        continue
        
    print(f"\n📦 Processing file: {raw_path}")
    data = np.load(raw_path, allow_pickle=True)
    
    # Extract dimensions and add the singleton axis for compatibility
    states = np.expand_dims(data['states'], axis=2)
    above_thresh = np.expand_dims(data['above_threshold'], axis=2)
    exploded = np.expand_dims(data['exploded'], axis=2)
    
    n_alpha, n_delta, n_sigma, total_vars, num_time_steps = states.shape
    N = (total_vars - 1) // 2
    t_real = data['time']
    dt = np.mean(np.diff(t_real))
    
    # 1. Compute Standard Grids
    print("  -> Calculating Spectral Entropy...")
    entropy_grid = compute_grid_spectral_entropy(states, exploded, N)
    
    print("  -> Calculating Peak Details...")
    N_peaks_grid, peak_positions = compute_grid_peak_details(states, exploded, N, dt)
    
    print("  -> Calculating Autocorrelation Metrics...")
    autocorr_grid = compute_grid_autocorrelation_metrics(states, exploded, N, dt)

    print("  -> Calculating classification Metrics...")
    classification_grid = compute_grid_classification(states, exploded, N, dt, epsilon2=2e-2)
    
    # 2. Compute Integrated Clustering & UMAP Projection
    print("  -> Embedding Spectra for Clustering & UMAP...")
    cluster_classification_grid = np.zeros((n_alpha, n_delta, n_sigma), dtype=int)
    cluster_classification_grid[exploded] = 1
    
    umap_coords_grid = np.full((n_alpha, n_delta, n_sigma, 2), np.nan)
    
    spectra_features = []
    valid_coordinates = []
    
    for i in range(n_alpha):
        for j in range(n_delta):
            for k in range(n_sigma):
                if exploded[i, j, k] or not above_thresh[i, j, k]:
                    continue
                X = np.sum(states[i, j, k, 0:N, :], axis=0)
                X_norm = (X - np.mean(X)) / (np.max(X) - np.min(X)) if (np.max(X) - np.min(X)) > 1e-12 else X - np.mean(X)
                fft_vals = np.abs(np.fft.rfft(X_norm))
                if np.sum(fft_vals) > 1e-12:
                    fft_vals /= np.sum(fft_vals)
                
                spectra_features.append(fft_vals)
                valid_coordinates.append((i, j, k))
                
    if len(spectra_features) > 0:
        feature_matrix = np.array(spectra_features)
        
        # K-Means
        centroids, labels = kmeans2(feature_matrix, k=3, minit='points', missing='warn')
        # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_embedding = reducer.fit_transform(feature_matrix)
        
        for idx, (i, j, k) in enumerate(valid_coordinates):
            cluster_classification_grid[i, j, k] = int(labels[idx]) + 2
            umap_coords_grid[i, j, k] = umap_embedding[idx]
            
    # 3. Export to a lightweight compressed file
    print(f"  💾 Saving compressed metrics to: {out_path}")
    np.savez_compressed(
        out_path,
        entropy_grid=entropy_grid,
        N_peaks_grid=N_peaks_grid,
        peak_positions=peak_positions,
        autocorr_grid=autocorr_grid,
        classification_grid=classification_grid,
        cluster_classification_grid=cluster_classification_grid,
        umap_coords_grid=umap_coords_grid
    )

print("\n✅ All metrics successfully pre-compiled and saved!")