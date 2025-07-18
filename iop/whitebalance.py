# -*- coding: utf-8 -*-
"""
White Balance IOP module.

This module provides a class-based interface for applying white balance to an
image based on color temperature and tint, following darktable's implementation.
"""

import numpy as np

try:
    from darktable_python.utils import color_science
except ImportError:
    # Fallback for when running from pipeline
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import color_science

# sRGB D65 conversion matrix, used as a fallback
XYZ_TO_SRGB_D65 = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]
])

class Whitebalance:
    """
    A class to manage and apply white balance settings, providing an API
    consistent with other IOP modules.
    """
    def __init__(self, temp_k=6502.0, tint=1.0, **kwargs):
        """
        Initializes the WhiteBalance module.

        Args:
            temp_k (float): The correlated color temperature in Kelvin.
            tint (float): The tint adjustment.
            image_info (dict, optional): Image metadata. Defaults to None.
        """
        self.temp_k = temp_k
        self.tint = tint
        self.image_info = kwargs if 'image_info' in kwargs else {}
        self.coeffs = self._calculate_coeffs()

    def _calculate_coeffs(self):
        """
        Calculates RGB multipliers from color temperature and tint.

        This method simulates darktable's implementation for post-demosaic images:
        1. Convert temperature and tint to an XYZ illuminant.
        2. Select the appropriate XYZ-to-RGB conversion matrix.
        3. Convert the illuminant's XYZ to camera RGB.
        4. Calculate and normalize the white balance coefficients.
        """
        # 1. Convert temperature and tint to XYZ illuminant
        illuminant_xyz = color_science.temperature_tint_to_xyz(self.temp_k, self.tint)

        # 2. Select XYZ -> RGB matrix
        # Use camera-specific matrix from metadata if available, else use sRGB
        if 'xyz_to_camera' in self.image_info:
            xyz_to_rgb_matrix = self.image_info['xyz_to_camera']
        else:
            xyz_to_rgb_matrix = XYZ_TO_SRGB_D65

        # 3. Convert illuminant from XYZ to camera RGB
        illuminant_rgb = np.dot(xyz_to_rgb_matrix, illuminant_xyz)

        # 4. Calculate and normalize coefficients
        coeffs = 1.0 / np.maximum(1e-8, illuminant_rgb)
        coeffs /= np.maximum(1e-8, coeffs[1])  # Normalize to green channel

        return tuple(coeffs)

    def process(self, image):
        """
        Applies the calculated white balance to an image.

        Args:
            image (np.ndarray): The input RGB image in a linear format.

        Returns:
            np.ndarray: The white-balanced image.
        """
        r_coeff, g_coeff, b_coeff = self.coeffs
        # Create a copy to avoid modifying the original image data
        balanced_image = image.copy()
        balanced_image[:, :, 0] *= r_coeff
        balanced_image[:, :, 1] *= g_coeff
        balanced_image[:, :, 2] *= b_coeff
        return balanced_image

if __name__ == '__main__':
    import imageio.v2 as imageio
    import os

    # --- Configuration for Standalone Verification ---
    # Define paths relative to this script's location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # Path to the input image for testing
    # This should be a linear image, before white balance is applied
    INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'test_suite', 'test_assets', 'whitebalence', 'sample-origin.tif')
    
    # Path to save the output image
    OUTPUT_FLOAT_PATH = os.path.join(PROJECT_ROOT, 'test_suite', 'test_assets', 'whitebalence', 'sample-python-output.tif')
    OUTPUT_VISUAL_PATH = os.path.join(PROJECT_ROOT, 'test_suite', 'test_assets', 'whitebalence', 'sample-python-output-visual.jpg')

    # White balance parameters to test with
    # These would typically come from a config file
    test_params = {
        'temp_k': 6502.0,
        'tint': 1.0
    }

    # Mock image_info dictionary, as we don't have a real RAW file here
    # In a real pipeline, this would be populated from rawpy
    mock_image_info = {
        'xyz_to_camera': None,  # This will force the use of the sRGB matrix
        'camera_wb': [1.0, 1.0, 1.0, 1.0], # Placeholder
        'daylight_wb': [1.0, 1.0, 1.0, 1.0] # Placeholder
    }

    print("--- Running WhiteBalance Module Standalone Verification ---")

    # 1. Load the input image
    print(f"1. Loading image from: {INPUT_IMAGE_PATH}")
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"ERROR: Input image not found at '{INPUT_IMAGE_PATH}'. Please ensure it exists.")
        exit()
    input_image = imageio.imread(INPUT_IMAGE_PATH).astype(np.float32) / 65535.0

    # 2. Initialize the WhiteBalance module
    print(f"2. Initializing WhiteBalance with params: {test_params}")
    wb_module = Whitebalance(temp_k=test_params['temp_k'], tint=test_params['tint'], **mock_image_info)
    print(f"   - Calculated Coeffs: R={wb_module.coeffs[0]:.4f}, G={wb_module.coeffs[1]:.4f}, B={wb_module.coeffs[2]:.4f}")

    # 3. Process the image
    print("3. Applying white balance...")
    output_image = wb_module.process(input_image)

    # 4. Save the 32-bit float output image
    print(f"4. Saving 32-bit float output to: {OUTPUT_FLOAT_PATH}")
    # Ensure the image is in float32 format for direct saving
    output_image_float32 = output_image.astype(np.float32)
    imageio.imwrite(OUTPUT_FLOAT_PATH, output_image_float32)

    # 5. Convert to visual format (8-bit with gamma correction) and save
    print(f"5. Saving visual output to: {OUTPUT_VISUAL_PATH}")
    # Apply gamma correction for visualization
    gamma_corrected_image = np.clip(output_image, 0, 1) ** (1 / 2.2)
    # Convert to 8-bit integer for saving as JPG
    output_image_uint8 = (gamma_corrected_image * 255).astype(np.uint8)
    imageio.imwrite(OUTPUT_VISUAL_PATH, output_image_uint8)

    print("--- Verification Finished ---")
    print(f"Please compare the output file with a baseline image to verify correctness.")