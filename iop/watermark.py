import numpy as np
from PIL import Image

class Watermark:
    """
    Overlays a watermark image onto the main image.
    This is a Python implementation of a simplified 'watermark' (overlay) module.
    """
    def __init__(self, **kwargs):
        """
        Initializes the Watermark module.

        Args:
            **kwargs: Keyword arguments for IOP parameters.
                - watermark_file (str): Path to the watermark image file.
                - scale (float): Scaling factor for the watermark, relative to the main image's width. Default: 0.2.
                - opacity (float): Opacity of the watermark (0.0 to 1.0). Default: 0.5.
                - position (str): Corner to place the watermark ('top-left', 'top-right', 
                                  'bottom-left', 'bottom-right', 'center'). Default: 'bottom-right'.
        """
        self.params = {
            'watermark_file': 'darktable_python/assets/logo.png',
            'scale': 0.2,
            'opacity': 0.5,
            'position': 'bottom-right'
        }
        self.params.update(kwargs)
        self.name = 'watermark'

    def process(self, image):
        """
        Applies the watermark to the image.

        Args:
            image (np.ndarray): The input image in RGB format (H, W, C) with float values in [0, 1].

        Returns:
            np.ndarray: The image with the watermark.
        """
        try:
            # Convert main image from numpy [0,1] float to PIL [0,255] uint8
            # RGBA mode: RGB + Alpha channel, 
            # Alpha channel: 0-255, 0 is transparent, 255 is opaque
            main_image_pil = Image.fromarray((image * 255).astype(np.uint8)).convert('RGBA')
            
            # Load watermark image
            watermark = Image.open(self.params['watermark_file']).convert('RGBA')
        except FileNotFoundError:
            print(f"[{self.name}] Watermark file not found at {self.params['watermark_file']}. Skipping.")
            return image

        # --- Scale watermark ---
        main_w, main_h = main_image_pil.size
        base_width = int(main_w * self.params['scale'])
        w_percent = (base_width / float(watermark.size[0]))
        h_size = int((float(watermark.size[1]) * float(w_percent)))
        watermark = watermark.resize((base_width, h_size), Image.Resampling.LANCZOS)
        
        # --- Adjust opacity ---
        if self.params['opacity'] < 1.0:
            alpha = watermark.split()[3]
            alpha = alpha.point(lambda p: p * self.params['opacity'])
            watermark.putalpha(alpha)

        # --- Position watermark ---
        wm_w, wm_h = watermark.size
        pos = self.params['position']
        margin = int(main_w * 0.02) # 2% margin

        if pos == 'bottom-right':
            position = (main_w - wm_w - margin, main_h - wm_h - margin)
        elif pos == 'top-left':
            position = (margin, margin)
        elif pos == 'top-right':
            position = (main_w - wm_w - margin, margin)
        elif pos == 'bottom-left':
            position = (margin, main_h - wm_h - margin)
        elif pos == 'center':
            position = ((main_w - wm_w) // 2, (main_h - wm_h) // 2)
        else: # default to bottom-right
            position = (main_w - wm_w - margin, main_h - wm_h - margin)

        # --- Composite images ---
        main_image_pil.paste(watermark, position, watermark)

        # Convert back to numpy [0,1] float for the pipeline
        final_image_np = np.array(main_image_pil.convert('RGB')) / 255.0
        
        print(f"[{self.name}] Applied watermark from '{self.params['watermark_file']}'.")
        return final_image_np.astype(np.float32) 