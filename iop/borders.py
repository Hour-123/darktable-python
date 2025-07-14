import numpy as np

class Borders:
    """
    Adds a solid color border around the image.
    This is a Python implementation of a simplified 'borders' module.
    """
    def __init__(self, **kwargs):
        """
        Initializes the Borders IOP.

        Args:
            **kwargs: Keyword arguments for IOP parameters.
                      - size (float): Border size as a percentage of the image's shorter side.
                                      Range: 0.0 to 1.0. Default: 0.1 (10%).
                      - color (list or tuple): RGB color for the border.
                                               Values are in the range [0, 255].
                                               Default: [255, 255, 255] (white).
        """
        # Default parameters
        self.params = {
            'size': 0.1,
            'color': [255, 255, 255]
        }
        
        # Update params with any provided kwargs
        self.params.update(kwargs)
        self.name = 'borders' # for logging

    def process(self, image):
        """
        Applies the border effect to the image.

        Args:
            image (np.ndarray): The input image in RGB format (H, W, C) with float values in [0, 1].

        Returns:
            np.ndarray: The image with the added border.
        """
        h, w, c = image.shape
        
        # Normalize color from [0, 255] to [0, 1]
        border_color = np.array(self.params['color']) / 255.0
        
        # Calculate border size in pixels
        shorter_side = min(h, w)
        border_px = int(shorter_side * self.params['size'])
        
        # Calculate new dimensions
        new_h = h + 2 * border_px
        new_w = w + 2 * border_px
        
        # Create new image (canvas) with border color
        new_image = np.full((new_h, new_w, c), border_color, dtype=np.float32)
        
        # Copy original image into the center of the new image, 
        # replace the original image(central canvas) with the new image
        top = border_px
        left = border_px
        new_image[top:top+h, left:left+w, :] = image
        
        print(f"[{self.name}] Applied a {border_px}px border.")
        return new_image 