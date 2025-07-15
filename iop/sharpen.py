import numpy as np
from scipy.ndimage import gaussian_filter

class Sharpen:
    """
    Sharpens the image by applying an 'Unsharp Mask' method, which is based
    on the high-pass filter. It enhances the details (edges) of the image.
    """
    def __init__(self, **kwargs):
        """
        Initializes the Sharpen IOP.

        Args:
            **kwargs: Keyword arguments for IOP parameters.
                      - radius (float): The radius (sigma) of the Gaussian blur used
                                        to create the high-pass layer. This determines
                                        the size of the details being sharpened.
                                        Default: 1.0.
                      - amount (float): The strength of the sharpening effect. It controls
                                        how much the high-pass layer is blended with the
                                        original. Range: 0.0 to 5.0. Default: 1.0.
        """
        self.params = {
            'radius': 1.0,
            'amount': 1.0,
        }
        self.params.update(kwargs)
        self.name = 'sharpen'

    def _overlay_blend(self, base, blend):
        """
        Applies the 'Overlay' blend mode.
        It multiplies or screens the colors, depending on the base color.
        """
        return np.where(base <= 0.5, 
                        2 * base * blend, 
                        1 - 2 * (1 - base) * (1 - blend))

    def process(self, image):
        """
        Applies the sharpening effect.

        Args:
            image (np.ndarray): Input image, float values in [0, 1].

        Returns:
            np.ndarray: The sharpened image.
        """
        radius = self.params['radius']
        amount = self.params['amount']

        if radius <= 0 or amount <= 0:
            return image

        # 1. Create the low-pass version (blurred)
        lowpass = gaussian_filter(image, sigma=(radius, radius, 0))

        # 2. Create the high-pass layer (details)
        highpass = image - lowpass

        # 3. Blend the high-pass layer with the original image using Overlay mode.
        #    The `amount` parameter scales the high-pass layer before blending.
        #    A neutral gray (0.5) in the highpass layer results in no change.
        sharpened_image = self._overlay_blend(image, highpass * amount + 0.5)

        # Clip to maintain valid range
        final_image = np.clip(sharpened_image, 0.0, 1.0)
        
        print(f"[{self.name}] Applied sharpen with radius {radius} and amount {amount}.")

        return final_image.astype(image.dtype) 