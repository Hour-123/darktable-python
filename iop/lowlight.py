import numpy as np
from skimage.color import rgb2xyz, xyz2rgb, rgb2lab, lab2xyz, xyz2lab

def _create_lut(curve_points):
    """
    Creates a Look-Up Table (LUT) from a set of curve points.
    """
    # Ensure the curve starts at 0 and ends at 1 on the x-axis
    points = sorted(curve_points, key=lambda p: p[0])
    if points[0][0] > 0:
        points.insert(0, [0, points[0][1]])
    if points[-1][0] < 1:
        points.append([1, points[-1][1]])
        
    x_coords, y_coords = zip(*points)
    
    # Generate a LUT with 65536 entries for high precision, matching darktable
    lut_indices = np.linspace(0, 1, 65536)
    lut_values = np.interp(lut_indices, x_coords, y_coords)
    return lut_values.astype(np.float32)

def _lookup(lut, values):
    """
    Looks up values in the LUT.
    'values' should be a NumPy array scaled to [0, 1].
    """
    indices = (values * (len(lut) - 1)).astype(np.int32)
    return lut[indices]

class Lowlight:
    def __init__(self, blueness=20.0, curve_points=None):
        """
        Initializes the Lowlight vision operator.
        
        Args:
            blueness (float): The amount of blue shift in the scotopic white point.
            curve_points (list of [x, y]): Points defining the brightness-to-mix transition.
                                           x is brightness (0-1), y is mix weight (0-1).
        """
        self.blueness = blueness
        
        if curve_points is None:
            # Default curve: linear transition from full effect to no effect.
            self.curve_points = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        else:
            self.curve_points = curve_points
            
        # Create the Look-Up Table
        self.lut = _create_lut(self.curve_points)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Simulates human lowlight vision (Purkinje effect).
        
        Args:
            image (np.ndarray): Input RGB image, float32, range [0, 1].
            
        Returns:
            np.ndarray: Output RGB image, float32, range [0, 1].
        """
        # 1. Define scotopic white point and convert to XYZ
        # In Lab space, a negative 'b*' value is blue.
        lab_scotopic_white = np.array([100.0, 0.0, -self.blueness], dtype=np.float32).reshape(1, 1, 3)
        xyz_scotopic_white = lab2xyz(lab_scotopic_white)

        # 2. Convert input RGB image to Lab and XYZ
        lab_image = rgb2lab(image)
        xyz_image = rgb2xyz(image)

        # 3. Calculate scotopic luminance (V)
        # Empirical formula from darktable's lowlight.c
        coeff = 0.5
        threshold = 0.01
        
        X, Y, Z = xyz_image[..., 0], xyz_image[..., 1], xyz_image[..., 2]
        
        # Avoid division by zero for very dark pixels
        safe_X = np.maximum(X, threshold)
        
        V = Y * (1.33 * (1.0 + (Y + Z) / safe_X) - 1.68)
        V = np.clip(coeff * V, 0, 1)

        # 4. Get blend weights from LUT based on L* channel
        # Normalize L* from [0, 100] to [0, 1] for lookup
        L_star = lab_image[..., 0] / 100.0
        w = _lookup(self.lut, L_star)
        
        # Reshape for broadcasting
        w = w[..., np.newaxis] 
        V = V[..., np.newaxis]

        # 5. Calculate the scotopic (night vision) color
        xyz_scotopic_image = V * xyz_scotopic_white

        # 6. Blend original and scotopic colors
        blended_xyz = w * xyz_image + (1.0 - w) * xyz_scotopic_image
        
        # 7. Convert back to RGB
        final_rgb = xyz2rgb(blended_xyz)
        
        return np.clip(final_rgb, 0, 1) 