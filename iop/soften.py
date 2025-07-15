
import numpy as np
from skimage import color
from scipy.ndimage import uniform_filter

class Soften():
    """
    Simulates the Orton effect, creating a softened, glowing look.
    This effect is achieved by blending a blurred, brightened, and
    more saturated version of the image with the original.
    """

    def __init__(self, size: float = 50.0, saturation: float = 100.0,
                 brightness: float = 0.33, amount: float = 50.0):
        """
        Initializes the Soften operation.

        Args:
            size (float): The radius of the blur, controls the diffusion.
                          Range [0.0, 100.0]. Default is 50.0.
            saturation (float): The saturation boost for the blurred layer.
                                Range [0.0, 100.0]. Default is 100.0.
            brightness (float): The brightness boost for the blurred layer.
                                Range [-2.0, 2.0]. Default is 0.33.
            amount (float): The mix amount between the original and softened image.
                            Range [0.0, 100.0]. Default is 50.0.
        """
        
        self.size = np.clip(size, 0.0, 100.0)
        self.saturation = np.clip(saturation, 0.0, 100.0)
        self.brightness = np.clip(brightness, -2.0, 2.0)
        self.amount = np.clip(amount, 0.0, 100.0)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the soften (Orton) effect to the image.

        Args:
            image (np.ndarray): The input image in RGB format, with values in [0, 1].

        Returns:
            np.ndarray: The processed image.
        """
        # Step 1: Create the "overexposed" layer
        hsv_image = color.rgb2hsv(image)

        # Apply brightness and saturation adjustments
        # Brightness adjustment is exponential, mimicking darktable's behavior
        hsv_image[..., 2] *= np.exp2(self.brightness)
        hsv_image[..., 1] *= (self.saturation / 100.0)

        # Clip values to be within the valid range [0, 1]
        np.clip(hsv_image, 0, 1, out=hsv_image)

        # Convert back to RGB
        overexposed_layer = color.hsv2rgb(hsv_image)

        # Step 2: Blur the "overexposed" layer
        # Calculate blur radius based on image size, similar to darktable
        h, w, _ = image.shape
        max_radius = np.sqrt(w*w + h*h) * 0.01
        radius = int(max_radius * (min(100.0, self.size + 1) / 100.0))

        # Apply a uniform box filter if radius is greater than 0
        if radius > 0:
            # The filter size should be odd
            filter_size = 2 * radius + 1
            blurred_layer = uniform_filter(overexposed_layer, size=filter_size, mode='nearest')
        else:
            blurred_layer = overexposed_layer.copy()

        # Step 3: Blend the blurred layer with the original image
        mix_amount = self.amount / 100.0
        output_image = (1.0 - mix_amount) * image + mix_amount * blurred_layer
        
        return np.clip(output_image, 0, 1) 