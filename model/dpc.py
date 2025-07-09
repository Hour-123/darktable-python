# @Author: Jiahao Huang
# @Date: 2025-07-09 
# @Description: DPC model

import numpy as np
from typing import Dict, Any, Union

class DPC: 
    """
    DPC model

    steps: 
        1. padding the inputs
        2. correct the dead pixel, adopt mean filter to correct the dead pixel
        3. return the corrected image

    usage: 
        dpc = DPC(inputs = img, white_level, dead_pixel_threshold).run()
    """
    def __init__(self, inputs: np.ndarray, **kwargs: Any) -> None: 
        self.inputs = inputs
        self.kwargs = kwargs
        self.bayer_pattern = self.kwargs.get('bayer_pattern', 'RGGB')
        self.white_level = int(self.kwargs.get('white_level', 1023))
        self.dead_pixel_threshold = float(self.kwargs.get('dead_pixel_threshold', 30))
        
    def __padding_inputs(self, inputs: np.ndarray, padding: int) -> np.ndarray: 
        """
        padding the inputs, mode is reflect
        """
        return np.pad(inputs, ((padding, padding), (padding, padding)), mode='reflect')
    
    def __dead_pixel_correction(self) -> np.ndarray: 
        """
        dead pixel correction, adopt mean filter to correct the dead pixel
        """
        H, W = self.inputs.shape
        white_level = self.white_level
        padding = 2

        inputs = self.inputs.astype(np.float32)
        inputs = self.__padding_inputs(inputs, padding)

        mean_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        mean_kernel /= np.sum(mean_kernel)

        for i in range(padding, H + padding, 2): 
            for j in range(padding, W + padding, 2): 
                if (abs(inputs[i - 2 : i + 3 : 2, j - 2 : j + 3 : 2] - inputs[i, j]) > self.dead_pixel_threshold).all(): 
                    inputs[i, j] = np.sum(inputs[i - 2 : i + 3 : 2, j - 2 : j + 3 : 2] * mean_kernel)
        
        return np.clip(inputs, 0, white_level).astype(np.uint16)
    
    def run(self) -> np.ndarray: 
        """
        run the DPC
        """
        return self.__dead_pixel_correction()

if __name__ == '__main__': 
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    white_level = 1023

    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116185548.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
    dpc = DPC(inputs = img, white_level = white_level, dead_pixel_threshold = 30.0).run()
    dpc_output= cv2.demosaicing(dpc, cv2.COLOR_BayerRG2BGR)
    dpc_output_8 = (dpc_output.astype(np.uint32) * 255 // white_level).astype(np.uint8)
    
    cv2.imwrite(root_path / 'outputs' / '01_dpc.png', dpc_output_8)