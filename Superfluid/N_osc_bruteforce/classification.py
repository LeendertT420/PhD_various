import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks, correlate
from scipy.cluster.vq import kmeans2
from typing import Dict, Tuple, List
from equations import mu_spectrum

def analyze_signal_spectral_entropy(x_total: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Computes the normalized Shannon spectral entropy for a single time trace.
    
    Returns
    -------
    entropy : float
        Spectral entropy normalized between 0.0 (pure single frequency) 
        and 1.0 (white noise / uniform power distribution).
    """
    x_ac = x_total - np.mean(x_total)
    
    # Guard against un-lasing flatlines or dead signals
    if np.max(np.abs(x_ac)) < epsilon:
        return 0.0

    # Compute Power Spectrum via Real FFT
    fft_vals = np.fft.rfft(x_ac)
    power_spectrum = np.abs(fft_vals)**2
    
    # Normalize power spectrum to create a valid probability distribution
    ps_sum = np.sum(power_spectrum)
    if ps_sum > 0:
        power_spectrum /= ps_sum
    else:
        return 0.0
    
    # Calculate Shannon Spectral Entropy
    ps_safe = np.clip(power_spectrum, 1e-12, None)
    entropy = -np.sum(ps_safe * np.log(ps_safe)) / np.log(len(ps_safe))
    
    return float(entropy)



def analyze_signal_peaks(
    x_total: np.ndarray, 
    dt: float, 
    peak_prominence: float = 0.15, 
    peak_height: float = 0.05, 
    epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the variance-normalized FFT of a single time trace and returns detected peaks.
    Can be called completely independently for any individual simulation trace.
    
    Returns
    -------
    peaks : ndarray
        Indices of the detected peaks in the FFT array.
    fft_freqs : ndarray
        The continuous frequency bins corresponding to the FFT axes.
    fft_vals : ndarray
        Variance-normalized single-sided amplitude spectrum values.
    """
    n_points = len(x_total)
    fft_freqs = np.fft.rfftfreq(n_points, d=dt)
    
    # Detrend and check raw threshold limits
    x_ac = x_total - np.mean(x_total)
    if np.max(np.abs(x_ac)) < epsilon:
        return np.array([], dtype=int), fft_freqs, np.zeros_like(fft_freqs)
        
    # Apply variance normalization to stabilize peak prominence
    sigma_x = np.std(x_ac)
    if sigma_x < 1e-12:
        return np.array([], dtype=int), fft_freqs, np.zeros_like(fft_freqs)
        
    x_normalized = x_ac / sigma_x
    fft_vals = (2.0 / n_points) * np.abs(np.fft.rfft(x_normalized))
    
    peaks, _ = find_peaks(fft_vals, prominence=peak_prominence, height=peak_height)
    return peaks, fft_freqs, fft_vals


def analyze_signal_autocorrelation(x_total: np.ndarray) -> float:
    """
    Computes the periodic coherence strength for a single time trace via 
    the Wiener-Khinchin theorem.
    """
    x_ac = x_total - np.mean(x_total)
    if np.max(np.abs(x_ac)) < 1e-6:
        return 0.0
        
    n_points = len(x_total)
    n_fft = 2 * n_points - 1
    
    fft_autocorr = np.fft.fft(x_ac, n=n_fft)
    power_spectrum = np.abs(fft_autocorr)**2
    autocorr_full = np.fft.ifft(power_spectrum).real
    autocorr = autocorr_full[:n_points]
    
    if autocorr[0] > 1e-12:
        autocorr /= autocorr[0]
        corr_peaks, _ = find_peaks(autocorr, prominence=0.01)
        if len(corr_peaks) > 0:
            return float(autocorr[corr_peaks[0]])
            
    return 0.0


def analyze_signal_autocorrelation_scipy(x_total: np.ndarray) -> float:
    """
    Computes the normalized autocorrelation of a signal using SciPy 
    and extracts the value of the first local minimum (negative valley).
    
    Returns:
        float: Close to -1.0 for highly coherent/periodic anti-correlation.
               Close to 0.0 for chaotic decorrelation or flatlines.
    """
    # 1. Mean-center the signal to remove DC offset bias
    x_ac = x_total - np.mean(x_total)
    variance = np.var(x_ac)
    
    # Fast escape fallback for flatline signals
    if variance < 1e-12:
        return 0.0  
        
    # 2. Compute the raw autocorrelation cross-product using SciPy
    # 'full' mode generates an array of size (2 * len(x) - 1)
    raw_corr = correlate(x_ac, x_ac, mode='full')
    
    # 3. Slice array to look at forward time lags only (lag 0 onwards)
    mid_point = len(raw_corr) // 2
    positive_lags = raw_corr[mid_point:]
    
    # 4. Normalize the array so that lag 0 starts exactly at 1.0
    normalized_corr = positive_lags / positive_lags[0]
    
    # 5. Locate the first negative valley using SciPy's peak finder
    # Inverting the signal flips valleys into positive peaks for find_peaks
    valleys, _ = find_peaks(-normalized_corr)
    
    if len(valleys) > 0:
        # Extract the value at the very first structural minimum
        first_valley_value = float(normalized_corr[valleys[0]])
        return first_valley_value
        
    return 0.0


def classify_signal_state(
    x_total: np.ndarray, 
    dt: float, 
    N: int, 
    peak_prominence: float = 0.1, 
    peak_height: float = 0.04, 
    epsilon: float = 1e-6,
    epsilon2: float = 1e-2
) -> int:
    """
    Classifies the operating state of a single trace. 
    Internally calls the standalone 'analyze_signal_peaks' function.
    """
    # Call the standalone peak finder
    peaks, fft_freqs, fft_vals = analyze_signal_peaks(
        x_total, dt, peak_prominence, peak_height, epsilon
    )
    
    if len(peaks) == 0 or np.max(x_total-np.mean(x_total)) < epsilon2:
        return 0  # Below Threshold
    if len(peaks) == 1:
        return 1  # Single-Mode Lasing
    
    #old definition of mode-locked 
    df = fft_freqs[1] - fft_freqs[0]
    active_freqs = np.sort(fft_freqs[peaks])
    eigenfreqs = np.sqrt(mu_spectrum(N)) / (2 * np.pi)
    
    # Check for Mode-Locking (Harmonic Comb check)
    f0 = active_freqs[0]
    margin = 4 * df
    is_mode_locked = f0 > margin
    
    if is_mode_locked:
        for f in active_freqs[1:]:
            nearest_multiple = round(f / f0) * f0
            if np.abs(f - nearest_multiple) > margin:
                is_mode_locked = False
                break
                
    if is_mode_locked:
        return 2  # Mode-Locked State
    
    #new definition
    '''
    df = fft_freqs[1] - fft_freqs[0]
    active_freqs = np.sort(fft_freqs[peaks])

    # Sanity check: Mode-locking requires a comb of at least 2 interacting frequencies
    if len(active_freqs) >= 2:
        f0 = active_freqs[0]
        margin = 4 * df
        max_denom = int(N) 
        
        is_mode_locked = True
        
        # Pre-build our array of allowable denominators [1, 2, ..., N]
        allowed_denominators = np.arange(1, max_denom + 1)
        
        for f in active_freqs[1:]:
            ratio = f / f0
            
            # Find the closest matching numerator for every allowable denominator
            closest_numerators = np.round(ratio * allowed_denominators)
            
            # Calculate the resulting rational frequency ratios
            candidate_ratios = closest_numerators / allowed_denominators
            
            # Find the denominator that minimizes the distance to our target ratio
            errors = np.abs(ratio - candidate_ratios)
            best_idx = np.argmin(errors)
            
            # Reconstruct the physical frequency from the best rational approximation
            expected_f = candidate_ratios[best_idx] * f0
            
            # Check if the actual peak matches our best-fit rational structure
            if np.abs(f - expected_f) > margin:
                is_mode_locked = False
                break
                
        if is_mode_locked:
            return 2  # Mode-Locked State'''
        
    eigenfreqs = np.sqrt(mu_spectrum(N)) / (2 * np.pi)
        
    # Multi-Mode vs Chaos Verification Line
    if len(peaks) < N:
        close_to_eigen = True
        for f, amp in zip(active_freqs, fft_vals[peaks]):
            closest_theoretical = eigenfreqs[np.argmin(np.abs(eigenfreqs - f))]
            if np.abs(f - closest_theoretical) > 3 * df:
                close_to_eigen = False
        if close_to_eigen:
            return 3 # 3 = Multi-Mode, 4 = Chaos
    
    if len(peaks) < N and np.sum(3*fft_vals[peaks]/len(fft_vals) > np.mean(fft_vals)) > len(peaks):
        return 3
    
    if analyze_signal_spectral_entropy(x_total) < 0.1:
        return 3
        
    return 4  # Chaos due to high peak density


# =====================================================================
# PART 2: 3D PARAMETER GRID RUNNERS (FOR PRE-COMPUTATION PIPELINES)
# =====================================================================
def compute_grid_spectral_entropy(
    states: np.ndarray, 
    exploded: np.ndarray, 
    N: int,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Loops over the 3D parameter grid evaluating only spectral entropy values.

    Parameters
    ----------
    states : ndarray
        5D state data tensor shape: (len(alphas), len(deltas), len(sigmas), dim, n_points)
    exploded : ndarray
        3D boolean mask shape: (len(alphas), len(deltas), len(sigmas))
    N : int
        Number of active structural modes to aggregate.
    epsilon : float
        Minimum signal threshold limit.

    Returns
    -------
    entropy_matrix : ndarray
        3D matrix containing the computed spectral entropy values. 
        Exploded states are preserved as np.nan.
    """
    n_alpha, n_delta, n_sigma, _, _ = states.shape
    entropy_matrix = np.full((n_alpha, n_delta, n_sigma), np.nan)
    
    has_exploded_data = exploded is not None and exploded.size > 0
    
    print("Calculating Spectral Entropy across the parameter grid...")
    for i in tqdm(range(n_alpha)):
        for j in range(n_delta):
            for k in range(n_sigma):
                coord = (i, j, k)
                
                # Skip if the simulation exploded (leaves position as np.nan)
                if has_exploded_data and exploded[coord]:
                    continue
                
                # Isolate and sum the mode coordinates for a global signal
                x_total = np.sum(states[i, j, k, 0:N, :], axis=0)
                
                # Call the standalone single-trace calculation engine
                entropy_matrix[coord] = analyze_signal_spectral_entropy(x_total, epsilon)
                
    return entropy_matrix


def compute_grid_peak_details(
    states: np.ndarray, 
    exploded: np.ndarray, 
    N: int, 
    dt: float,
    peak_prominence: float = 0.15, 
    peak_height: float = 0.05,
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]]]:
    """Loops over the parameter grid evaluating only peak allocations."""
    n_alpha, n_delta, n_sigma, _, _ = states.shape
    peak_count_grid = np.zeros((n_alpha, n_delta, n_sigma), dtype=int)
    peak_positions: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
    
    has_exploded_data = exploded is not None and exploded.size > 0
    
    print("Computing Grid Peak Details Separately...")
    for i in tqdm(range(n_alpha)):
        for j in range(n_delta):
            for k in range(n_sigma):
                coord = (i, j, k)
                if has_exploded_data and exploded[coord]:
                    peak_count_grid[coord] = -1
                    peak_positions[coord] = (np.array([]), np.array([]))
                    continue
                    
                x_total = np.sum(states[i, j, k, 0:N, :], axis=0)
                peaks, freqs, vals = analyze_signal_peaks(x_total, dt, peak_prominence, peak_height, epsilon)
                
                peak_count_grid[coord] = len(peaks)
                peak_positions[coord] = (freqs[peaks], vals[peaks])
                
    return peak_count_grid, peak_positions


def compute_grid_autocorrelation_metrics(
    states: np.ndarray, 
    exploded: np.ndarray, 
    N: int, 
    dt: float
) -> np.ndarray:
    """Loops over the parameter grid evaluating only phase coherence bounds."""
    n_alpha, n_delta, n_sigma, _, _ = states.shape
    autocorr_grid = np.zeros((n_alpha, n_delta, n_sigma), dtype=float)
    
    has_exploded_data = exploded is not None and exploded.size > 0
    
    print("Computing Grid Autocorrelation Separately...")
    for i in tqdm(range(n_alpha)):
        for j in range(n_delta):
            for k in range(n_sigma):
                coord = (i, j, k)
                if has_exploded_data and exploded[coord]:
                    autocorr_grid[coord] = 0.0
                    continue
                    
                x_total = np.sum(states[i, j, k, 0:N, :], axis=0)
                autocorr_grid[coord] = analyze_signal_autocorrelation(x_total)
                
    return autocorr_grid


def compute_grid_classification(
    states: np.ndarray, 
    exploded: np.ndarray, 
    N: int, 
    dt: float,
    peak_prominence: float = 0.15, 
    peak_height: float = 0.05,
    epsilon: float = 1e-6,
    epsilon2:float = 1e-3
) -> np.ndarray:
    """Loops over the parameter grid evaluating only operational classifications."""
    n_alpha, n_delta, n_sigma, _, _ = states.shape
    classification_grid = np.zeros((n_alpha, n_delta, n_sigma), dtype=int)
    
    has_exploded_data = exploded is not None and exploded.size > 0
    
    print("Computing Grid Regime Classifications Separately...")
    for i in tqdm(range(n_alpha)):
        for j in range(n_delta):
            for k in range(n_sigma):
                coord = (i, j, k)
                if has_exploded_data and exploded[coord]:
                    classification_grid[coord] = 4
                    continue
                    
                x_total = np.sum(states[i, j, k, 0:N, :], axis=0)
                classification_grid[coord] = classify_signal_state(
                    x_total, dt, N, peak_prominence, peak_height, epsilon, epsilon2
                )
                
    return classification_grid


def compute_all_grid_metrics(
    states: np.ndarray, 
    exploded: np.ndarray, 
    N: int, 
    dt: float,
    peak_prominence: float = 0.15, 
    peak_height: float = 0.05,
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Executes a highly optimized, single-pass pipeline over the 3D parameter grid.
    Computes spectral peaks, positions, operational state classifications, phase 
    coherence, and spectral entropy concurrently without redundant data looping.

    Returns
    -------
    classification_grid : ndarray (int)
        3D map: 0=Below Threshold, 1=Single-Mode, 2=Mode-Locked, 3=Multi-Mode, 4=Chaos
    autocorr_grid : ndarray (float)
        3D matrix storing the height of the first periodic correlation recurrence.
    peak_count_grid : ndarray (int)
        3D matrix containing the number of identified frequency peaks.
    peak_positions : dict
        Keys are (i, j, k) tuples; values are (frequencies, powers) arrays.
    entropy_matrix : ndarray (float)
        3D matrix of Shannon spectral entropy values. Exploded states are np.nan.
    """
    n_alpha, n_delta, n_sigma, _, _ = states.shape
    
    # 1. Preallocate all tracking structures
    classification_grid = np.zeros((n_alpha, n_delta, n_sigma), dtype=int)
    autocorr_grid = np.zeros((n_alpha, n_delta, n_sigma), dtype=float)
    peak_count_grid = np.zeros((n_alpha, n_delta, n_sigma), dtype=int)
    entropy_matrix = np.full((n_alpha, n_delta, n_sigma), np.nan)
    peak_positions: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
    
    # Pre-calculate theoretical internal array values
    eigenfreqs = np.sqrt(mu_spectrum(N)) / (2 * np.pi)
    has_exploded_data = exploded is not None and exploded.size > 0
    
    print("Executing Unified Single-Pass Master Analysis Loop...")
    for i in tqdm(range(n_alpha)):
        for j in range(n_delta):
            for k in range(n_sigma):
                coord = (i, j, k)
                
                # --- Guard 1: Catastrophic Explosion States ---
                if has_exploded_data and exploded[coord]:
                    classification_grid[coord] = 4  # Classify forced chaos
                    autocorr_grid[coord] = 0.0
                    peak_count_grid[coord] = -1
                    peak_positions[coord] = (np.array([]), np.array([]))
                    # entropy_matrix[coord] remains np.nan
                    continue
                
                # Isolate global aggregate simulation signal X(t)
                x_total = np.sum(states[i, j, k, 0:N, :], axis=0)
                
                # --- Guard 2: Idle / Below-Threshold States ---
                x_ac = x_total - np.mean(x_total)
                if np.max(np.abs(x_ac)) < epsilon:
                    classification_grid[coord] = 0
                    autocorr_grid[coord] = 0.0
                    peak_count_grid[coord] = 0
                    peak_positions[coord] = (np.array([]), np.array([]))
                    entropy_matrix[coord] = 0.0
                    continue
                
                # =============================================================
                # METRIC 1 & 2: PEAK DETECTION & POSITION TRACKING
                # =============================================================
                # Call our standalone independent peak engine
                peaks, fft_freqs, fft_vals = analyze_signal_peaks(
                    x_total, dt, peak_prominence, peak_height, epsilon
                )
                
                peak_count_grid[coord] = len(peaks)
                peak_positions[coord] = (fft_freqs[peaks], fft_vals[peaks])
                
                # =============================================================
                # METRIC 3: SPECTRUM ENTROPY (OPTIMIZED)
                # =============================================================
                # Optimization: Reuse the existing fft_vals to build the power distribution
                power_spectrum = fft_vals**2
                ps_sum = np.sum(power_spectrum)
                if ps_sum > 0:
                    power_spectrum /= ps_sum
                    ps_safe = np.clip(power_spectrum, 1e-12, None)
                    entropy_matrix[coord] = -np.sum(ps_safe * np.log(ps_safe)) / np.log(len(ps_safe))
                else:
                    entropy_matrix[coord] = 0.0
                
                # =============================================================
                # METRIC 4: REGIME CLASSIFICATION ENGINE
                # =============================================================
                if len(peaks) == 0:
                    classification_grid[coord] = 0
                elif len(peaks) == 1:
                    classification_grid[coord] = 1
                else:
                    df = fft_freqs[1] - fft_freqs[0]
                    active_freqs = np.sort(fft_freqs[peaks])
                    
                    # Harmonic Comb Filter for Mode-Locking
                    f0 = active_freqs[0]
                    margin = 2 * df
                    is_mode_locked = f0 > margin
                    if is_mode_locked:
                        for f in active_freqs[1:]:
                            nearest_multiple = round(f / f0) * f0
                            if np.abs(f - nearest_multiple) > margin:
                                is_mode_locked = False
                                break
                    
                    if is_mode_locked:
                        classification_grid[coord] = 2
                    elif len(peaks) < N:
                        # Match proximity to structural array internal eigenfrequencies
                        close_to_eigen = True
                        for f in active_freqs:
                            closest_theoretical = eigenfreqs[np.argmin(np.abs(eigenfreqs - f))]
                            if np.abs(f - closest_theoretical) > 3 * df:
                                close_to_eigen = False
                                break
                        classification_grid[coord] = 3 if close_to_eigen else 4
                    else:
                        classification_grid[coord] = 4
                
                # =============================================================
                # METRIC 5: TIME-DOMAIN AUTOCORRELATION
                # =============================================================
                autocorr_grid[coord] = analyze_signal_autocorrelation(x_total)
                
    print("Master Pre-Computation Complete. All pipelines synced.")
    return classification_grid, autocorr_grid, peak_count_grid, peak_positions, entropy_matrix



def compute_grid_unsupervised_classification(
    states: np.ndarray, 
    exploded: np.ndarray, 
    above_thresh: np.ndarray, 
    N: int, 
    num_clusters: int = 3
) -> np.ndarray:
    """
    Embeds the FFT power spectrum of each grid coordinate into a high-dimensional 
    space and uses SciPy's K-Means algorithm to automatically discover 
    and classify the active dynamical regimes.
    
    Args:
        num_clusters (int): Number of active regimes to discover (default 3: 
                            Single-Mode, Mode-Locked, and Chaos).
    """
    n_alpha, n_delta, n_sigma, _, num_time_steps = states.shape
    
    # 1. Create a placeholder to collect high-dimensional features and coordinates
    spectra_features = []
    valid_coordinates = []
    
    for i in range(n_alpha):
        for j in range(n_delta):
            for k in range(n_sigma):
                # Filter out uninteresting states to avoid polluting the embedding space
                if exploded[i, j, k] or not above_thresh[i, j, k]:
                    continue
                
                # Reconstruct total displacement X
                X = np.sum(states[i, j, k, 0:N, :], axis=0)
                
                # Mean-center and scale amplitude so the clustering engine 
                # focuses entirely on the SHAPE of the spectrum, not its magnitude.
                if np.max(X) - np.min(X) > 1e-12:
                    X_norm = (X - np.mean(X)) / (np.max(X) - np.min(X))
                else:
                    X_norm = X - np.mean(X)
                
                # Compute power spectrum profile
                fft_vals = np.abs(np.fft.rfft(X_norm))
                
                # Normalize the spectrum area to 1.0 (acts like a probability density distribution)
                fft_sum = np.sum(fft_vals)
                if fft_sum > 1e-12:
                    fft_vals /= fft_sum
                
                spectra_features.append(fft_vals)
                valid_coordinates.append((i, j, k))
    
    # Initialize the final output classification grid
    # Defaulting to 0 (Below Threshold)
    cluster_grid = np.zeros((n_alpha, n_delta, n_sigma), dtype=int)
    
    # Pre-map explicit physical exceptions
    cluster_grid[exploded] = 1  # 1: Exploded/Unstable
    
    # If no active points cross threshold, skip ML clustering step safely
    if len(spectra_features) == 0:
        return cluster_grid
        
    # 2. Stack lists into a 2D high-dimensional matrix: Shape (M_points, N_frequencies)
    feature_matrix = np.array(spectra_features)
    
    # 3. Apply SciPy K-Means
    # 'minit=points' initializes centroids choosing actual data points randomly
    centroids, labels = kmeans2(feature_matrix, k=num_clusters, minit='points', missing='warn')
    
    # 4. Map the discovered labels back to the 3D space.
    # Shifting labels up by 2 to protect our custom 0 (Idle) and 1 (Exploded) bounds!
    for idx, (i, j, k) in enumerate(valid_coordinates):
        cluster_grid[i, j, k] = int(labels[idx]) + 2
        
    return cluster_grid