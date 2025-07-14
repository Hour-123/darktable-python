import numpy as np
from scipy.ndimage import gaussian_filter

class Grain:
    """
    A Python implementation of darktable's 'grain' module.

    This class adds photographic grain to an image to simulate a film look.
    """
    def __init__(self, strength=0.3, coarseness=0.5, **kwargs):
        """
        Initializes the Grain module.

        Args:
            strength (float): The intensity of the grain effect. Ranges from 0 to 1.
            coarseness (float): The size of the grain particles. Ranges from 0 to 1.
        """
        self.strength = strength
        self.coarseness = coarseness

    def process(self, image):
        """
        Applies the grain effect to the input image.

        Args:
            image (np.ndarray): The input image as a NumPy array (H, W, C).

        Returns:
            np.ndarray: The image with the grain effect.
        """
        h, w, c = image.shape
        
        # 1. Generate uniform random noise for each channel independently.
        # The noise is centered around 0 (from -0.5 to 0.5).
        noise = np.random.rand(h, w, c) - 0.5
        
        # 2. Control coarseness by blurring the noise.
        # A larger coarseness value means a larger sigma for the Gaussian blur,
        # which makes the noise particles clump together, appearing larger.
        # We scale coarseness to a practical sigma range. A small offset prevents sigma from being zero.
        if self.coarseness > 0:
            sigma = 1 + self.coarseness * 5 # 这里的 5 是经验值，可以调整
            # We apply the blur to each channel of the noise independently.
            noise_channels = [gaussian_filter(noise[:, :, i], sigma=sigma) for i in range(c)]
            noise = np.stack(noise_channels, axis=-1)

        # 3. Apply strength.
        # We scale the noise by the strength factor.
        # The final noise will be added to the image, so it's centered around 0.
        grain = noise * self.strength
        
        # 4. Add the grain to the image.
        # The grain is added to the image. Using addition is a simple but effective
        # way to simulate grain, which is additive in nature on film.
        grained_image = image + grain
        
        # 5. Clip the result to ensure it stays within the valid [0, 1] range.
        return np.clip(grained_image, 0, 1) 