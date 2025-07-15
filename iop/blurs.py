
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from enum import Enum
import imageio.v2 as imageio
import os

class BlurType(Enum):
    GAUSSIAN = "gaussian"
    LENS = "lens"
    MOTION = "motion"

class Blurs:
    """
    Provides a collection of blur effects, including Gaussian, Lens, and Motion blur.
    This is a Python implementation inspired by darktable's 'blurs' module.
    """

    def __init__(self,
                 blur_type: str = "gaussian",
                 radius: int = 10,
                 # Lens blur params
                 blades: int = 5,
                 concavity: float = 1.0,
                 linearity: float = 1.0,
                 rotation: float = 0.0,
                 # Motion blur params
                 angle: float = 0.0,
                 curvature: float = 0.0,
                 offset: float = 0.0):
        """
        Initializes the Blurs operation.

        Args:
            blur_type (BlurType): The type of blur to apply.
            radius (int): General size/radius for the blur effect.
            blades (int): Number of diaphragm blades for lens blur (3-11).
            concavity (float): Concavity of the diaphragm blades for lens blur.
            linearity (float): Linearity of the diaphragm blades for lens blur.
            rotation (float): Rotation of the diaphragm for lens blur.
            angle (float): Angle of the motion blur in degrees.
            curvature (float): Curvature of the motion path.
            offset (float): Offset of the motion path from the center.
        """
        try:
            self.blur_type = BlurType(blur_type)
        except ValueError:
            raise ValueError(f"'{blur_type}' is not a valid blur type. Available types are: {[e.value for e in BlurType]}")
            
        self.radius = radius
        self.blades = blades
        self.concavity = concavity
        self.linearity = linearity
        self.rotation = rotation
        self.angle = angle
        self.curvature = curvature
        self.offset = offset

    def _create_lens_kernel(self) -> np.ndarray:
        """
        Creates a kernel that simulates the shape of a camera's diaphragm (bokeh).
        This is a Python/NumPy translation of darktable's `create_lens_kernel`.
        """
        # Kernel size must be odd
        size = int(self.radius) * 2 + 1
        kernel = np.zeros((size, size))

        # Aliasing reduction
        eps = 1.0 / size
        radius = (size - 1) / 2.0 - 1

        # Center of the kernel
        center = (size - 1) / 2.0

        n = self.blades
        m = self.concavity
        k = self.linearity
        rotation_rad = np.deg2rad(self.rotation)

        for i in range(size):
            for j in range(size):
                # Normalized coordinates in [-1, 1]
                x = (i - center) / radius
                y = (j - center) / radius

                # Polar coordinates
                r = np.sqrt(x**2 + y**2)
                if r > 1.0:
                    continue
                
                theta = np.arctan2(y, x) + rotation_rad
                
                # Formula from https://math.stackexchange.com/a/4160104/498090
                # This complex formula models the diaphragm shape.
                # It's simplified here for clarity and performance. A more direct port:
                term_in_asinf = k * np.cos(n * theta)
                # Clamp the argument for asinf to avoid domain errors
                term_in_asinf = np.clip(term_in_asinf, -1.0, 1.0)
                
                M_numerator = np.cos((2. * np.arcsin(k) + np.pi * m) / (2. * n))
                M_denominator = np.cos((2. * np.arcsin(term_in_asinf) + np.pi * m) / (2. * n))
                
                # Avoid division by zero
                if np.abs(M_denominator) < 1e-9:
                    M = 1.0
                else:
                    M = M_numerator / M_denominator

                if M >= r + eps:
                    kernel[i, j] = 1.0
        
        # Normalize the kernel
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
            
        return kernel

    def _create_motion_kernel(self) -> np.ndarray:
        """
        Creates a kernel that simulates motion blur.
        This is a Python/NumPy translation of darktable's `create_motion_kernel`.
        """
        # Kernel size must be odd
        size = int(self.radius) * 2 + 1
        kernel = np.zeros((size, size))
        center = (size - 1) / 2.0
        
        # Polynomial params from user controls
        A = self.curvature / 2.0
        B = 1.0
        C = -A * self.offset**2 + B * self.offset
        
        # Rotation matrix
        angle_rad = np.deg2rad(self.angle)
        # The C code has a -pi/4 correction, we might need to adjust this based on visual results
        # For now, we'll stick to a direct interpretation of 'angle'
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # Oversample to draw a smoother line/curve
        oversample_factor = 8
        num_points = size * oversample_factor
        
        for i in range(num_points):
            # Normalized coordinates in [-1, 1]
            x_norm = (i / oversample_factor - center) / center
            
            # Build the motion path: 2nd order polynomial
            # X = x_norm - self.offset # C code uses this, let's test a simpler model first
            y_norm = x_norm**2 * A + x_norm * B + C

            # Rotate the point
            vec = np.array([x_norm, y_norm])
            rot_vec = rotation_matrix @ vec
            
            # Convert back to kernel coordinates
            kernel_x = int(round(rot_vec[0] * center + center))
            kernel_y = int(round(rot_vec[1] * center + center))

            # Draw the point on the kernel if it's within bounds
            if 0 <= kernel_x < size and 0 <= kernel_y < size:
                kernel[kernel_y, kernel_x] = 1.0
        
        # Normalize the kernel
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
            
        return kernel

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the selected blur to the image.
        """
        if self.blur_type == BlurType.GAUSSIAN:
            # For gaussian_filter, the 'sigma' is the standard deviation of the
            # Gaussian kernel. A larger radius should correspond to a larger sigma.
            # We can use a simple heuristic, like sigma = radius / 2.
            sigma = self.radius / 2.0
            # The filter is applied to each channel independently.
            return gaussian_filter(image, sigma=(sigma, sigma, 0))
        elif self.blur_type == BlurType.LENS:
            kernel = self._create_lens_kernel()
            # Apply convolution to each channel separately
            result = np.zeros_like(image, dtype=np.float32)
            for channel in range(image.shape[2]):
                result[..., channel] = convolve(image[..., channel], kernel, mode='reflect')
            return result
        elif self.blur_type == BlurType.MOTION:
            kernel = self._create_motion_kernel()
            result = np.zeros_like(image, dtype=np.float32)
            for channel in range(image.shape[2]):
                result[..., channel] = convolve(image[..., channel], kernel, mode='reflect')
            return result
        else:
            raise ValueError(f"Unknown blur type: {self.blur_type}")

if __name__ == '__main__':
    print("--- Running Blurs Module Local Test (Gaussian) ---")

    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    # Use a previously generated output as input for this test
    input_path = os.path.join(project_root, 'output', 'pipeline_with_diffuse_output.jpg')
    gaussian_output_path = os.path.join(project_root, 'output', 'blurs_test_gaussian_output.jpg')
    lens_output_path = os.path.join(project_root, 'output', 'blurs_test_lens_output.jpg')
    motion_output_path = os.path.join(project_root, 'output', 'blurs_test_motion_output.jpg')

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
    else:
        # Load the image
        print(f"Loading image from: {input_path}")
        image = imageio.imread(input_path)
        image_float = image.astype(np.float32) / 255.0

        # --- Test Gaussian Blur ---
        print("\nInitializing Blurs module for Gaussian blur...")
        blurer_gaussian = Blurs(blur_type='gaussian', radius=20)
        print("Processing image with Gaussian blur...")
        processed_gaussian = blurer_gaussian.process(image_float)
        output_gaussian_uint8 = (np.clip(processed_gaussian, 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(gaussian_output_path, output_gaussian_uint8)
        print(f"--- Gaussian test complete. Image saved to: {gaussian_output_path} ---")

        # --- Test Lens Blur ---
        print("\nInitializing Blurs module for Lens blur...")
        # Parameters to create a visible hexagonal bokeh effect
        blurer_lens = Blurs(blur_type='lens', radius=10, blades=6, concavity=1, linearity=1)
        print("Processing image with Lens blur...")
        processed_lens = blurer_lens.process(image_float)
        output_lens_uint8 = (np.clip(processed_lens, 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(lens_output_path, output_lens_uint8)
        print(f"--- Lens blur test complete. Image saved to: {lens_output_path} ---")

        # --- Test Motion Blur ---
        print("\nInitializing Blurs module for Motion blur...")
        # Parameters for a simple 45-degree linear motion blur
        blurer_motion = Blurs(blur_type='motion', radius=30, angle=90, curvature=0.01)
        print("Processing image with Motion blur...")
        processed_motion = blurer_motion.process(image_float)
        output_motion_uint8 = (np.clip(processed_motion, 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(motion_output_path, output_motion_uint8)
        print(f"--- Motion blur test complete. Image saved to: {motion_output_path} ---") 