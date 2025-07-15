import numpy as np
from scipy.ndimage import gaussian_filter

class Highpass:
    """
    Isolates high-frequency details in the image.
    This is achieved by subtracting a blurred version of the image from the
    original. The resulting high-pass layer can then be blended with the
    original image to achieve sharpening (e.g., using an Overlay blend mode).
    """
    def __init__(self, **kwargs):
        """
        Initializes the Highpass IOP.

        Args:
            **kwargs: Keyword arguments for IOP parameters.
                      - radius (float): The radius (sigma) for the Gaussian blur that defines
                                        the frequency cut-off. Smaller radii isolate finer
                                        details. Default: 10.0.
                      - contrast (float): A factor to boost the contrast of the high-pass
                                          layer, making the details more prominent.
                                          Range: 0.0 to 2.0. Default: 1.0 (no change).
        """
        self.params = {
            'radius': 10.0,
            'contrast': 1.0
        }
        self.params.update(kwargs)
        self.name = 'highpass'

    def process(self, image):
        """
        Applies the high-pass filter.

        Args:
            image (np.ndarray): Input image, float values in [0, 1].

        Returns:
            np.ndarray: A high-pass filtered image, centered around 0.5.
        """
        radius = self.params['radius']
        contrast = self.params['contrast']

        # Ensure radius is not zero to avoid division by zero or no-op.
        if radius <= 0:
            # Return a neutral gray image if there's no radius.
            return np.full_like(image, 0.5, dtype=image.dtype)

        # 1. Create the low-pass version of the image using a Gaussian blur.
        #    The blur is applied to each color channel independently.
        lowpass = gaussian_filter(image, sigma=(radius, radius, 0))

        # 2. Subtract the low-pass from the original to get the high-pass layer.
        #    The result is centered around 0.0 (ranging from -1.0 to 1.0).
        highpass_layer = image - lowpass

        # 3. Apply contrast. This amplifies the details.
        #    We are multiplying values centered around 0, so this works as expected.
        highpass_layer *= contrast

        # 4. Add 0.5 to shift the range from [-1.0, 1.0] to [-0.5, 1.5]
        #    with 0 details being represented by 0.5 (neutral gray).
        #    This makes the high-pass layer visible as an image.
        final_image = highpass_layer + 0.5

        # Clip the values to ensure they remain in the [0, 1] range for display.
        final_image = np.clip(final_image, 0.0, 1.0)
        
        print(f"[{self.name}] Applied high-pass filter with radius {radius} and contrast {contrast}.")

        return final_image.astype(image.dtype) 