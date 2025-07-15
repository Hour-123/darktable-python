import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab, lab2rgb

class Bloom:
    """
    A Python implementation of darktable's 'bloom' module.

    This class creates a bloom or glow effect on the bright areas of an image,
    operating in the Lab color space for better color preservation.
    """
    def __init__(self, strength=0.1, radius=0.02, threshold=90.0, **kwargs):
        """
        Initializes the Bloom module.

        Args:
            strength (float): The intensity of the bloom effect. (0.0 to 1.0)
            radius (float): The radius of the glow, as a fraction of the image's largest dimension.
                            (Practical range: 0.005 to 0.1)
            threshold (float): The brightness threshold (in L* of Lab space) to select pixels that will glow.
                               (Range: 0-100, darktable default is 90).
        """
        self.strength = strength
        self.radius = radius
        self.threshold = threshold

    def process(self, image):
        """
        Applies the bloom effect to the input image.

        Args:
            image (np.ndarray): The input image as a NumPy array (H, W, C) in linear sRGB space.

        Returns:
            np.ndarray: The image with the bloom effect.
        """
        h, w, c = image.shape

        # 1. Convert image to Lab color space
        lab_image = rgb2lab(image)
        
        # Extract the L* channel (luminance)
        luminance = lab_image[:, :, 0]

        # 2. Extract the bright parts based on the L* channel.
        highlights = np.where(luminance >= self.threshold, luminance, 0)

        # 3. Blur the highlights to create the "glow".
        # A more reasonable sigma calculation. It's scaled by the largest dimension
        # of the image to maintain a similar look across different aspect ratios.
        sigma = self.radius * max(h, w)
        blurred_highlights = gaussian_filter(highlights, sigma=sigma)

        # 4. Blend the blurred highlights back onto the original L* channel.
        # Darktable uses a Screen blend mode. The formula is: 1 - (1-base) * (1-blend)
        # Since our L* channel is in range [0, 100], we normalize it for blending.
        l_norm = luminance / 100.0
        blur_norm = blurred_highlights / 100.0
        
        # Apply strength to the blend layer
        # Note: In a true Screen blend, strength is often applied differently,
        # but this simple multiplication is intuitive and effective.
        final_glow = blur_norm * self.strength

        # Screen blend
        blended_l_norm = 1.0 - (1.0 - l_norm) * (1.0 - final_glow)
        
        # Denormalize back to L* range [0, 100]
        lab_image[:, :, 0] = blended_l_norm * 100.0
        
        # 5. Convert back to RGB color space.
        bloom_image_rgb = lab2rgb(lab_image)

        # 6. Clip the result to ensure it stays within the valid [0, 1] range.
        return np.clip(bloom_image_rgb, 0, 1) 