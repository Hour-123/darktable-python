"""
tonecurve - A Python implementation of the tonecurve IOP.
"""

import numpy as np
from scipy.interpolate import CubicSpline

class Tonecurve:
    """
    A class-based representation of the tone curve module.

    This implementation uses cubic spline interpolation to create a smooth
    curve from a few control points, and then generates a Look-Up Table (LUT)
    for efficient processing, mimicking darktable's approach.
    """
    def __init__(self, control_points_x: list, control_points_y: list, lut_size: int = 65536):
        """
        Initializes the ToneCurve module.

        Args:
            control_points_x (list): A list of x-coordinates for the curve's control points (e.g., [0, 0.25, 0.75, 1]).
                                     Values should be normalized between 0.0 and 1.0.
            control_points_y (list): A list of y-coordinates for the curve's control points (e.g., [0, 0.2, 0.8, 1]).
                                     Values should be normalized between 0.0 and 1.0.
            lut_size (int): The number of entries in the Look-Up Table. 65536 corresponds to 16-bit precision.
        """
        if len(control_points_x) != len(control_points_y) or len(control_points_x) < 2:
            raise ValueError("Control points must have at least 2 points, and x and y lists must be of the same size.")
        
        self.control_points_x = np.array(control_points_x)
        self.control_points_y = np.array(control_points_y)
        self.lut_size = lut_size
        
        # Create the Look-Up Table upon initialization
        self.lut = self._create_lut()

    def _create_lut(self) -> np.ndarray:
        """
        Creates a Look-Up Table (LUT) from the control points using cubic spline interpolation.
        """
        print("Creating Tone Curve LUT...")
        
        # Ensure the curve passes through the exact start and end points
        # bc_type='natural' is a common choice for smooth endings.
        # Alternatively, 'clamped' can be used if endpoint derivatives are known.
        spline = CubicSpline(self.control_points_x, self.control_points_y, bc_type='natural')
        
        # Generate the x-values for the LUT (the "input" values)
        lut_indices = np.linspace(0.0, 1.0, self.lut_size)
        
        # Calculate the interpolated y-values (the "output" values)
        lut_values = spline(lut_indices)
        
        # Clip the LUT values to be within the [0, 1] range to avoid artifacts
        lut_values = np.clip(lut_values, 0.0, 1.0)
        
        print("LUT created successfully.")
        return lut_values

    def process(self, image_data: np.ndarray) -> np.ndarray:
        """
        Applies the tone curve to the image data using the pre-computed LUT.

        Args:
            image_data (np.ndarray): The input image data, expected to be float
                                     and normalized between 0.0 and 1.0.

        Returns:
            np.ndarray: The processed image data.
        """
        if image_data.max() > 1.0 or image_data.min() < 0.0:
            print("Warning: Input image data is not normalized to [0, 1]. Clipping to range.")
            image_data = np.clip(image_data, 0.0, 1.0)

        # Convert the float image data to integer indices for LUT lookup
        # (e.g., a pixel value of 0.5 becomes index 32767 in a 65536-size LUT)
        indices = (image_data * (self.lut_size - 1)).astype(np.uint16)
        
        # Apply the LUT. This is a very fast operation.
        processed_image = self.lut[indices]
        
        return processed_image.astype(np.float32)

if __name__ == '__main__':
    # A simple test case to demonstrate usage
    import os
    import rawpy
    import imageio

    print("--- Running ToneCurve Module Test ---")

    # Define a classic 'S-curve' for increasing contrast
    x_points = [0.0, 0.25, 0.75, 1.0]
    y_points = [0.0, 0.15, 0.85, 1.0]

    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_raw_path = os.path.join(base_dir, '..', 'database', 'sample2.dng')
    output_image_path = os.path.join(base_dir, 'output', 'tonecurve_test_output.jpg')

    try:
        print(f"1. Reading and decoding RAW file: {input_raw_path}")
        with rawpy.imread(input_raw_path) as raw:
            raw_image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
        
        linear_image_float = raw_image.astype(np.float32) / 65535.0

        print(f"2. Initializing ToneCurve with an S-curve for contrast.")
        tone_curve_module = Tonecurve(x_points, y_points)
        processed_image = tone_curve_module.process(linear_image_float)

        print("3. Applying gamma correction for display.")
        gamma_corrected_image = np.clip(processed_image, 0, 1) ** (1 / 2.2)
        output_image_uint8 = (gamma_corrected_image * 255).astype(np.uint8)

        print(f"4. Saving result to: {output_image_path}")
        imageio.imwrite(output_image_path, output_image_uint8)
        print("--- Test successful! ---")

    except Exception as e:
        print(f"An error occurred during the test: {e}") 