"""
filmicrgb - A Python implementation of the filmicrgb IOP.
"""

import numpy as np
from scipy.interpolate import CubicSpline

class Filmicrgb:
    """
    A class-based representation of the filmic RGB module.

    This is a simplified implementation focusing on the core tone mapping
    functionality of filmicrgb. It translates the essential spline-based
    dynamic range compression from the darktable C source code.
    """
    def __init__(self,
                 white_point_ev: float = 4.0,
                 black_point_ev: float = -8.0,
                 contrast: float = 1.5,
                 latitude: float = 0.5,
                 preserve_color: str = "luminance"):
        """
        Initializes the Filmicrgb module with its core parameters.

        Args:
            white_point_ev (float): The relative exposure of the whitest point
                                    in the image that should retain detail.
            black_point_ev (float): The relative exposure of the blackest point.
            contrast (float): The contrast of the 'S' curve. Higher values
                              mean a steeper, more contrasty curve.
            latitude (float): The width of the 'linear' portion of the curve,
                              representing the mid-tones.
            preserve_color (str): The method for preserving color.
                                  Supported: 'luminance', 'max_rgb'.
        """
        self.white_point_ev = white_point_ev
        self.black_point_ev = black_point_ev
        self.contrast = contrast
        self.latitude = latitude
        self.preserve_color = preserve_color

        # These will be computed based on the parameters above.
        # We will implement this logic step-by-step.
        self._spline_coeffs, self._spline_nodes = self._compute_spline()

    def _compute_spline(self):
        """
        Computes the filmic spline coefficients.
        
        This is a Python translation of the core logic from darktable's
        `dt_iop_filmic_rgb_compute_spline` function.
        """
        print("--- Computing Filmic Spline ---")
        
        # 1. Parameter conversion from UI values to internal representation
        # Convert EV values to linear scale
        black_point_lin = 2.0 ** self.black_point_ev
        white_point_lin = 2.0 ** self.white_point_ev
        
        # dynamic_range is the total range in linear values
        dynamic_range = white_point_lin - black_point_lin
        if dynamic_range < 1e-6:
            dynamic_range = 1e-6

        # Latitude defines the width of the linear central part of the curve.
        # It's expressed as a percentage of the dynamic range.
        latitude_range = self.latitude * (white_point_lin - black_point_lin)
        
        # The 'balance' parameter from darktable is implicitly handled here by how
        # the latitude is centered around the mid-gray point (which we assume to be 0.1845).
        # For simplicity, we are centering the latitude logarithmically.
        mid_gray = 0.1845
        log_mid = np.log2(mid_gray)
        log_black = np.log2(black_point_lin)
        
        # Center the latitude range around the mid-gray point
        x0 = np.exp2(log_black)
        x1 = np.exp2(log_mid - latitude_range / 2.0)
        x2 = np.exp2(log_mid + latitude_range / 2.0)
        x3 = white_point_lin
        
        # This defines the 5 x-coordinates of our 4-segment spline
        spline_nodes_x = np.array([x0, x1, x2, x3])
        
        # The y-coordinates are determined by the target display range (0 to 1)
        # and the contrast.
        # The central part of the curve (y1 to y2) is made linear and its slope
        # is determined by the contrast.
        
        # Target display values
        D_black = 0.0
        D_white = 1.0
        D_range = D_white - D_black
        D_mid = (D_black + D_white) / 2.0
        
        # A more robust way to calculate the slope and y-nodes.
        # The user 'contrast' parameter is used to scale the output range of the
        # central linear section of the curve, which ensures y1 and y2 stay
        # within the [0, 1] bounds.
        # A contrast of 1.0 means the output range is the full display range.
        # A contrast of 2.0 means the output range is half the display range, making the slope steeper.
        if self.contrast < 1e-6:
            self.contrast = 1e-6
        output_latitude_width = D_range / self.contrast
        
        # Calculate y1 and y2 based on the compressed output latitude, centered around D_mid.
        y1 = D_mid - output_latitude_width / 2.0
        y2 = D_mid + output_latitude_width / 2.0
        
        spline_nodes_y = np.array([D_black, y1, y2, D_white])
        
        # We must ensure that the control points are monotonic for the spline to be well-behaved.
        spline_nodes_x = np.sort(spline_nodes_x)
        spline_nodes_y = np.sort(spline_nodes_y)

        # For now, we will use a simpler interpolation method rather than solving
        # the full 12x12 matrix system from darktable, which is very complex.
        # We can use CubicSpline to get a similar result with much less code.
        # This is a reasonable first approximation.
        # We use 'clamped' boundary conditions to ensure the ends of the curve are flat.
        spline = CubicSpline(spline_nodes_x, spline_nodes_y, bc_type='clamped')

        print("--- Filmic Spline Computed (using simplified CubicSpline) ---")
        
        # The "coefficients" are now the spline object itself.
        # We also return the node points for potential visualization.
        return spline, (spline_nodes_x, spline_nodes_y)

    def _get_pixel_norm(self, pixel_rgb: np.ndarray) -> np.ndarray:
        """
        (Placeholder) Computes the 'norm' or 'intensity' of a pixel.
        """
        if self.preserve_color == "max_rgb":
            # The 'max RGB' norm is simple and robust against desaturation.
            return np.max(pixel_rgb, axis=-1)
        
        # Default to luminance Y (needs proper color space conversion coefficients)
        # Using a simplified sRGB->Y conversion for now.
        # Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
        return (0.2126 * pixel_rgb[..., 0] +
                0.7152 * pixel_rgb[..., 1] +
                0.0722 * pixel_rgb[..., 2])

    def _apply_spline(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the computed spline to an input value x.
        """
        # The self._spline_coeffs is now the spline object from scipy
        # We need to clip the input to the spline's domain to avoid errors
        spline_domain = self._spline_nodes[0]
        x_clipped = np.clip(x, spline_domain[0], spline_domain[-1])
        return self._spline_coeffs(x_clipped)

    def process(self, image_data: np.ndarray) -> np.ndarray:
        """
        Applies the filmic tone mapping to the image data.
        """
        print("--- Applying Filmic RGB ---")
        
        if self._spline_coeffs is None:
            print("Warning: Spline not computed. Returning original image.")
            return image_data

        # 1. Get the norm (intensity) of each pixel
        old_norm = self._get_pixel_norm(image_data)
        # Add a small epsilon to avoid division by zero
        old_norm_safe = np.maximum(old_norm, 1e-9)

        # 2. Apply the filmic spline to the norm
        new_norm = self._apply_spline(old_norm)
        
        # 3. Re-calculate pixel RGB values to preserve hue
        # new_RGB = old_RGB * (new_norm / old_norm)
        # We need to handle 3 channels, so we expand the dimensions of the ratio.
        ratio = new_norm / old_norm_safe
        processed_image = image_data * ratio[..., np.newaxis]

        print("--- Filmic RGB applied (using placeholders) ---")
        return processed_image


if __name__ == '__main__':
    # A simple test case to demonstrate usage
    # Note: This will not work correctly until _compute_spline and _apply_spline are implemented.
    import os
    import rawpy
    import imageio
    
    # Add project root to path to allow imports
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from iop.exposure import Exposure

    print("--- Running Filmicrgb Module Test ---")

    # File paths
    base_dir = project_root
    input_raw_path = os.path.join(base_dir, '..', 'database', 'sample2.dng')
    output_image_path = os.path.join(base_dir, 'output', 'filmicrgb_test_output.jpg')

    try:
        print(f"1. Reading and decoding RAW file...")
        with rawpy.imread(input_raw_path) as raw:
            raw_image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
        
        linear_image_float = raw_image.astype(np.float32) / 65535.0

        print(f"2. Applying a pre-filmic exposure adjustment.")
        # It's common to adjust exposure before filmic
        exposure_module = Exposure(exposure_ev=-1.0)
        exposed_image = exposure_module.process(linear_image_float)
        
        print(f"3. Initializing Filmicrgb with default settings.")
        # These are just example values
        filmic_module = Filmicrgb(
            white_point_ev=4.0,
            black_point_ev=-9.0,
            contrast=1.6,
            latitude=0.4
        )
        processed_image = filmic_module.process(exposed_image)

        print("4. Applying gamma correction for display.")
        # NOTE: Filmicrgb already outputs to a display-ready-ish state, 
        # but a gamma curve is still needed for sRGB display.
        gamma_corrected_image = np.clip(processed_image, 0, 1) ** (1 / 2.2)
        output_image_uint8 = (gamma_corrected_image * 255).astype(np.uint8)

        print(f"5. Saving result to: {output_image_path}")
        imageio.imwrite(output_image_path, output_image_uint8)
        print("--- Test finished (using placeholders). ---")

    except Exception as e:
        print(f"An error occurred during the test: {e}") 