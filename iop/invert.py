import numpy as np

class Invert:
    """
    A Python implementation of darktable's 'invert' module.

    This class inverts the colors of an image.
    """
    def __init__(self, **kwargs):
        # This module has no parameters, but we accept kwargs for consistency.
        pass

    def process(self, image):
        """
        Inverts the input image.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: The inverted image.
        """
        # The core logic of the invert module
        return 1.0 - image 