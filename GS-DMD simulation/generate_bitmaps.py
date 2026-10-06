import os
import numpy as np
from PIL import Image
from dmd_simulation import DMD_simulation

# --- Experiment Parameters ---
OUTPUT_FOLDER = "dmd_bitmaps_5spots_diagonal"
N_SPOTS = 5
SEED = 42
ORIENTATION = "diagonal"
CARRIER_PERIODS = range(2, 21)  # Integer carrier periods 2 through 20
GS_ITERATIONS = 20

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize DMD simulation instance (Adjust resolution if your physical DMD differs)
sim = DMD_simulation(
    resolution=(1140, 912),  # (Nx, Ny)
    pitch=7.6e-6,
    wavelength=930e-9,
    f=0.1,
)

print(f"Generating DMD Bitmaps for N={N_SPOTS} spots in {ORIENTATION} orientation...")
print(f"Target Directory: {os.path.abspath(OUTPUT_FOLDER)}\n")

for period in CARRIER_PERIODS:
    # Fix seed per iteration to preserve exact target spot locations across periods
    np.random.seed(SEED)

    # 1. Pre-compensate spot coordinates for current carrier period
    sim.generate_precompensated_input_and_target(
        N_spots=N_SPOTS,
        carrier_period=period,
        carrier_orientation=ORIENTATION,
        spacing="random",
        plot=False,
    )

    # 2. Compute phase mask via Gerchberg-Saxton
    sim.run_gerchberg_saxton(N_iterations=GS_ITERATIONS)

    # 3. Generate binary Lee hologram
    hologram = sim.generate_lee_hologram(
        carrier_period=period,
        carrier_orientation=ORIENTATION,
    )

    # 4. Convert binary array (0.0 / 1.0) to 8-bit uint8 bitmap values (0 / 255)
    # 0 = Mirror OFF, 255 = Mirror ON
    hologram_uint8 = (hologram * 255).astype(np.uint8)
    img = Image.fromarray(hologram_uint8, mode="L")

    # 5. Format structured filename and save BMP image
    filename = f"hologram_5spots_{ORIENTATION}_period{period:02d}px.bmp"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    img.save(filepath)

    print(f"[+] Saved: {filename}")

print(f"\nSuccessfully generated {len(CARRIER_PERIODS)} bitmap files.")