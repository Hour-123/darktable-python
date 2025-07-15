
import numpy as np
from numba import njit

@njit
def _dither_loop(img_float, levels):
    """
    The core loop for Floyd-Steinberg dithering, optimized with Numba.
    """
    f = levels - 1
    for y in range(img_float.shape[0]):
        for x in range(img_float.shape[1]):
            old_pixel = img_float[y, x].copy()
            # Quantize the pixel to the nearest color
            new_pixel = np.round(old_pixel * f) / f
            img_float[y, x] = new_pixel
            
            # Calculate the quantization error
            quant_error = old_pixel - new_pixel
            
            # Diffuse the error to neighboring pixels
            if x + 1 < img_float.shape[1]:
                img_float[y, x + 1] += quant_error * 7 / 16
            if y + 1 < img_float.shape[0]:
                if x > 0:
                    img_float[y + 1, x - 1] += quant_error * 3 / 16
                img_float[y + 1, x] += quant_error * 5 / 16
                if x + 1 < img_float.shape[1]:
                    img_float[y + 1, x + 1] += quant_error * 1 / 16
    return img_float

class Dither:
    def __init__(self, levels=8, method='floyd-steinberg'):
        """
        Initializes the Dither/Posterize operator.
        
        Args:
            levels (int): The number of quantization levels per channel.
            method (str): The algorithm to use, 'floyd-steinberg' or 'posterize'.
        """
        if method not in ['floyd-steinberg', 'posterize']:
            raise ValueError("Method must be 'floyd-steinberg' or 'posterize'")
        self.levels = levels
        self.method = method

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply dithering or posterization to the image based on the chosen method.

        Args:
            image (np.ndarray): The input image, as a float32 array in [0, 1] range.

        Returns:
            np.ndarray: The processed image, as a float32 array in [0, 1] range.
        """
        if self.levels <= 1:
            return np.zeros_like(image)
            
        # The input 'image' is already a float32 [0,1] numpy array.
        # We work on a copy to avoid modifying the original array in place.
        img_float = image.copy()
        
        if self.method == 'floyd-steinberg':
            # Call the Numba-optimized loop for dithering
            processed_float = _dither_loop(img_float, self.levels)
        elif self.method == 'posterize':
            # Perform posterization directly
            f = self.levels - 1
            processed_float = np.round(img_float * f) / f
                        
        # Clip values to be in [0, 1] range before returning
        return np.clip(processed_float, 0, 1)

# The standalone posterize function is now obsolete and has been removed.

