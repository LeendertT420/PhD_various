import numpy as np
from typing import Tuple, Literal, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm


class DMD_simulation:
    """
    Simulates DMD pupil plane illumination, target Fourier spot generation,
    carrier shift pre-compensation, Gerchberg-Saxton phase retrieval, 
    binarized Lee Hologram generation, and far-field diffraction efficiency.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1140, 912),
        pitch: float = 7.6e-6,
        wavelength: float = 930e-9,
        f: float = 0.1,
        sigma_input: Optional[float] = None,
    ):
        self.Nx, self.Ny = resolution
        self.pitch = pitch
        self.wavelength = wavelength
        self.f = f

        # --- Beam Waists ---
        if sigma_input is None:
            self.sigma_input = (self.Ny * self.pitch) / 2
        else:
            self.sigma_input = sigma_input

        # --- Coordinate Grids ---
        self.x = (np.arange(self.Nx) - self.Nx / 2) * self.pitch
        self.y = (np.arange(self.Ny) - self.Ny / 2) * self.pitch
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Spatial frequency / Far-field coordinates
        self.fx = (np.arange(self.Nx) - self.Nx / 2) * (
            self.wavelength * self.f / (self.Nx * self.pitch)
        )
        self.fy = (np.arange(self.Ny) - self.Ny / 2) * (
            self.wavelength * self.f / (self.Ny * self.pitch)
        )

        self.span_fx = self.wavelength * self.f / self.pitch
        self.span_fy = self.wavelength * self.f / self.pitch

        # Internal state containers
        self.input_intensity: Optional[np.ndarray] = None
        self.input_amp: Optional[np.ndarray] = None
        self.target_intensity: Optional[np.ndarray] = None
        self.target_amp: Optional[np.ndarray] = None
        self.positions_x: Optional[np.ndarray] = None  # Desired target positions
        self.positions_y: Optional[np.ndarray] = None
        self.gs_positions_x: Optional[np.ndarray] = None  # Pre-shifted GS positions
        self.gs_positions_y: Optional[np.ndarray] = None
        self.phase_mask: Optional[np.ndarray] = None
        self.binary_hologram: Optional[np.ndarray] = None

    def _compute_input_beam(self) -> None:
        """Computes Gaussian input intensity and amplitude in pupil plane."""
        self.input_intensity = np.exp(
            -(self.X**2 + self.Y**2) / (2 * self.sigma_input**2)
        )
        self.input_intensity /= np.sum(self.input_intensity)
        self.input_amp = np.sqrt(self.input_intensity)

    def generate_input_and_target(
        self,
        N_spots: int,
        spacing: Literal["random", "even"] = "random",
        sigma_target: Optional[float] = None,
        sigma_target_mode: Literal["point", "diffrac_limited"] = "point",
        plot: bool = False,
    ) -> np.ndarray:
        """Generates input beam profile and desired target Fourier intensity."""
        if sigma_target is None:
            if sigma_target_mode == "diffrac_limited":
                sigma_target = (self.wavelength * self.f) / (
                    4 * np.pi * self.sigma_input
                )
            elif sigma_target_mode == "point":
                sigma_target = 0.0

        self._compute_input_beam()

        # --- Spot Target Generation (Fourier Plane) ---
        if spacing == "random":
            self.positions_x = np.random.uniform(-0.35, 0.35, size=N_spots) * (
                self.span_fx / 2
            )
            self.positions_y = np.random.uniform(-0.35, 0.35, size=N_spots) * (
                self.span_fy / 2
            )
        elif spacing == "even":
            self.positions_x = np.linspace(-0.35, 0.35, N_spots) * (
                self.span_fx / 2
            )
            self.positions_y = np.linspace(-0.35, 0.35, N_spots) * (
                self.span_fy / 2
            )

        self.gs_positions_x = self.positions_x.copy()
        self.gs_positions_y = self.positions_y.copy()

        # Build delta spot target matrix
        if sigma_target == 0:
            self.target_intensity = np.zeros((self.Ny, self.Nx))
            for x_spot, y_spot in zip(self.positions_x, self.positions_y):
                idx_x = np.argmin(np.abs(self.fx - x_spot))
                idx_y = np.argmin(np.abs(self.fy - y_spot))
                self.target_intensity[idx_y, idx_x] = 1.0
        else:
            raise NotImplementedError("Gaussian targets are not implemented yet.")

        self.target_intensity *= np.sum(self.input_intensity) / np.sum(
            self.target_intensity
        )
        self.target_amp = np.sqrt(self.target_intensity)

        if plot:
            self.plot_input_and_target()

        return self.target_intensity

    def generate_precompensated_input_and_target(
        self,
        N_spots: int,
        carrier_period: float,
        carrier_orientation: Literal[
            "horizontal", "vertical", "diagonal"
        ] = "diagonal",
        spacing: Literal["random", "even"] = "random",
        plot: bool = False,
    ) -> np.ndarray:
        """
        Generates target spot positions pre-compensated for Lee hologram carrier frequency shift.
        
        Subtracts the carrier shift offset vector (lambda * f) / (T_carrier * pitch)
        so that after propagation, the +1st order reconstructed spot lands precisely at
        the desired coordinates.
        """
        self._compute_input_beam()

        # 1. Determine desired final target coordinates
        if spacing == "random":
            self.positions_x = np.random.uniform(-0.35, 0.35, size=N_spots) * (
                self.span_fx / 2
            )
            self.positions_y = np.random.uniform(-0.35, 0.35, size=N_spots) * (
                self.span_fy / 2
            )
        elif spacing == "even":
            self.positions_x = np.linspace(-0.35, 0.35, N_spots) * (
                self.span_fx / 2
            )
            self.positions_y = np.linspace(-0.35, 0.35, N_spots) * (
                self.span_fy / 2
            )

        # 2. Calculate the spatial carrier shift vector [m]
        carrier_shift = (self.wavelength * self.f) / (carrier_period * self.pitch)

        if carrier_orientation == "horizontal":
            shift_x, shift_y = carrier_shift, 0.0
        elif carrier_orientation == "vertical":
            shift_x, shift_y = 0.0, carrier_shift
        elif carrier_orientation == "diagonal":
            shift_x, shift_y = carrier_shift, carrier_shift
        else:
            raise ValueError(f"Invalid orientation: {carrier_orientation}")

        # 3. Pre-shift GS target coordinates (subtract shift vector)
        self.gs_positions_x = self.positions_x - shift_x
        self.gs_positions_y = self.positions_y - shift_y

        # 4. Build pre-shifted target amplitude matrix for GS
        self.target_intensity = np.zeros((self.Ny, self.Nx))
        for x_spot, y_spot in zip(self.gs_positions_x, self.gs_positions_y):
            idx_x = np.argmin(np.abs(self.fx - x_spot))
            idx_y = np.argmin(np.abs(self.fy - y_spot))
            self.target_intensity[idx_y, idx_x] = 1.0

        self.target_intensity *= np.sum(self.input_intensity) / np.sum(
            self.target_intensity
        )
        self.target_amp = np.sqrt(self.target_intensity)

        if plot:
            self.plot_input_and_target()

        return self.target_intensity

    def run_gerchberg_saxton(
        self,
        N_iterations: int = 10,
        initial_phase: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Executes the Gerchberg-Saxton phase retrieval algorithm."""
        if self.input_amp is None or self.target_amp is None:
            raise ValueError(
                "Input and Target amplitudes not initialized. Run target generation first."
            )

        if initial_phase is None:
            phase_beam = (np.random.rand(self.Ny, self.Nx) * 2 - 1) * np.pi
        else:
            phase_beam = initial_phase

        for _ in tqdm(range(N_iterations + 1), desc="Gerchberg-Saxton Loop"):
            field_beam = self.input_amp * np.exp(1j * phase_beam)

            field_focus = np.fft.fftshift(np.fft.fft2(field_beam))
            phase_focus = np.angle(field_focus)

            # Update constraints
            field_focus = self.target_amp * np.exp(1j * phase_focus)
            field_beam = np.fft.ifft2(np.fft.ifftshift(field_focus))
            phase_beam = np.angle(field_beam)

        self.phase_mask = np.angle(np.exp(1j * phase_beam))
        return self.phase_mask

    def generate_lee_hologram(
        self,
        carrier_period: int = 6,
        carrier_orientation: Literal[
            "horizontal", "vertical", "diagonal"
        ] = "diagonal",
    ) -> np.ndarray:
        """Encodes the phase mask into a binary amplitude Lee Hologram."""
        if self.phase_mask is None:
            raise ValueError("Phase mask not found. Run run_gerchberg_saxton first.")

        carrier_freq = 2 * np.pi / carrier_period / self.pitch

        if carrier_orientation == "horizontal":
            carrier_phase = carrier_freq * self.X
        elif carrier_orientation == "vertical":
            carrier_phase = carrier_freq * self.Y
        elif carrier_orientation == "diagonal":
            carrier_phase = carrier_freq * (self.X + self.Y)

        fringes = (1 + np.cos(carrier_phase - self.phase_mask)) / 2

        self.binary_hologram = np.zeros_like(fringes)
        self.binary_hologram[fringes > 0.5] = 1.0

        return self.binary_hologram

    def propagate_to_far_field(
        self, hologram: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Propagates beam through the hologram to the Fourier plane."""
        if hologram is None:
            if self.binary_hologram is None:
                raise ValueError("No binary hologram available. Run generate_lee_hologram first.")
            hologram = self.binary_hologram

        if self.input_amp is None:
            raise ValueError("Input beam amplitude missing.")

        field_dmd = self.input_amp * hologram
        far_field_complex = np.fft.fftshift(np.fft.fft2(field_dmd))
        far_field_intensity = np.abs(far_field_complex) ** 2
        return far_field_intensity

    def calculate_diffraction_efficiency(
        self,
        hologram: Optional[np.ndarray] = None,
        integration_radius_pixels: int = 5,
    ) -> Dict[str, float]:
        """Calculates power ratio across diffraction orders."""
        far_field = self.propagate_to_far_field(hologram)
        total_power = np.sum(far_field)

        if total_power == 0:
            return {"1st_order": 0.0, "-1st_order": 0.0, "0th_order": 0.0, "background": 1.0}

        Ny, Nx = far_field.shape
        center_x, center_y = Nx // 2, Ny // 2
        y_grid, x_grid = np.ogrid[:Ny, :Nx]

        # 1. 0th Order / DC Power
        dc_mask = (
            (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
        ) <= integration_radius_pixels**2
        p_0th = np.sum(far_field[dc_mask])

        # 2. +1st Order (Target Spots at desired positions)
        mask_1st = np.zeros((Ny, Nx), dtype=bool)
        if self.positions_x is not None and self.positions_y is not None:
            for px, py in zip(self.positions_x, self.positions_y):
                idx_x = np.argmin(np.abs(self.fx - px))
                idx_y = np.argmin(np.abs(self.fy - py))
                spot_mask = (
                    (x_grid - idx_x) ** 2 + (y_grid - idx_y) ** 2
                ) <= integration_radius_pixels**2
                mask_1st |= spot_mask

        p_1st = np.sum(far_field[mask_1st])

        # 3. -1st Order (Conjugate Spots)
        mask_minus_1st = np.zeros((Ny, Nx), dtype=bool)
        if self.positions_x is not None and self.positions_y is not None:
            for px, py in zip(self.positions_x, self.positions_y):
                idx_x = np.argmin(np.abs(self.fx - px))
                idx_y = np.argmin(np.abs(self.fy - py))

                sym_idx_x = 2 * center_x - idx_x
                sym_idx_y = 2 * center_y - idx_y

                spot_mask = (
                    (x_grid - sym_idx_x) ** 2 + (y_grid - sym_idx_y) ** 2
                ) <= integration_radius_pixels**2
                mask_minus_1st |= spot_mask

        p_minus_1st = np.sum(far_field[mask_minus_1st])

        # 4. Background
        p_bg = max(0.0, total_power - (p_1st + p_minus_1st + p_0th))

        return {
            "1st_order": float(p_1st / total_power),
            "-1st_order": float(p_minus_1st / total_power),
            "0th_order": float(p_0th / total_power),
            "background": float(p_bg / total_power),
        }

    def plot_input_and_target(self) -> None:
        """Plots input intensity and target Fourier spots."""
        if self.input_intensity is None or self.target_intensity is None:
            raise ValueError("Input or target intensity has not been generated.")

        fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

        im0 = axs[0].imshow(
            self.input_intensity,
            extent=[self.x[0] * 1e3, self.x[-1] * 1e3, self.y[0] * 1e3, self.y[-1] * 1e3],
            origin="lower",
            cmap="viridis",
        )
        axs[0].set_title("Input Intensity (Pupil Plane)", fontsize=12, fontweight="bold")
        axs[0].set_xlabel("x [mm]")
        axs[0].set_ylabel("y [mm]")
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label="Intensity [a.u.]")

        im1 = axs[1].imshow(
            self.target_intensity,
            extent=[self.fx[0] * 1e3, self.fx[-1] * 1e3, self.fy[0] * 1e3, self.fy[-1] * 1e3],
            origin="lower",
            cmap="magma",
        )
        axs[1].set_title("Pre-Shifted Target Fourier Plane", fontsize=12, fontweight="bold")
        axs[1].set_xlabel("x [mm]")
        axs[1].set_ylabel("y [mm]")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="Target Intensity [a.u.]")

        plt.tight_layout()
        plt.show()