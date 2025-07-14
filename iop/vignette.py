import numpy as np

class Vignette:
    """
    A Python implementation of darktable's 'vignette' module.

    This class applies a vignette effect to an image, darkening the corners
    to draw focus to the center.
    """
    def __init__(self, strength=0.5, radius=0.7, softness=0.5, **kwargs):
        """
        Initializes the Vignette module.

        Args:
            strength (float): The darkness of the vignette. 0 is none, 1 is fully black.
            radius (float): The radius of the vignette circle, as a fraction of the image's half-diagonal.
            softness (float): The softness of the vignette's edge. 0 is a hard edge, 1 is very soft.
        """
        self.strength = strength
        self.radius = radius
        self.softness = softness

    def process(self, image):
        """
        Applies the vignette effect to the input image.

        Args:
            image (np.ndarray): The input image as a NumPy array (H, W, C).

        Returns:
            np.ndarray: The image with the vignette effect.
        """
        h, w, _ = image.shape
        
        # Create a coordinate grid
        y, x = np.ogrid[:h, :w]
        
        # Calculate the center of the image
        center_y, center_x = h / 2, w / 2
        
        # Calculate the distance of each pixel from the center, normalized by the image diagonal
        # This creates an elliptical distance to match image aspect ratio
        dist = np.sqrt(((y - center_y) / h)**2 + ((x - center_x) / w)**2)
        
        # Normalize distance to a more intuitive scale
        dist_norm = dist / np.max(dist)

        # Calculate the vignette mask using a smoothstep-like function
        # The clamp ensures softness doesn't go to zero, preventing division by zero.
        safe_softness = np.maximum(self.softness, 1e-6)
        
        # Calculate the inner and outer radius of the soft edge
        r1 = self.radius * (1 - safe_softness)
        r2 = self.radius
        
        # Create the gradient
        vignette_mask = (dist_norm - r1) / (r2 - r1)
        vignette_mask = np.clip(vignette_mask, 0, 1) # 非常重要，保证了之后 r1 内是透明的，r2 外是黑色的

        # Apply the strength and invert the mask (we want to darken the edges)
        vignette_mask = 1.0 - (vignette_mask * self.strength)
        
        # Apply the mask to the image (multiply on each color channel)
        return image * vignette_mask[:, :, np.newaxis] 