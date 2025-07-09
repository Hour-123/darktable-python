# @Author: Jiahao Huang
# @Date: 2025-07-09 
# @Description: Black Level Correction Module

import numpy as np
from typing import Dict, Any, Union

class BLC: 
    """
    BLC model

    description: this model is used to correct the black level of the image
    black level offset is set by default, but can be set by user. 
    In this demo, CMOS sensor is set to 10bits, so the white level is 1023, and offset is 64. 

    steps: 
        1. input the bayer pattern
        2. correct the black level
        3. return the corrected image

    usage: 
        blc = BLC(inputs = img, white_level, black_level).run()
    """
    def __init__(self, inputs: np.ndarray, **kwargs: Any) -> None: 
        self.inputs = inputs
        self.kwargs = kwargs
        self.bayer_pattern = self.kwargs.get('bayer_pattern', 'RGGB')
        self.white_level = int(self.kwargs.get('white_level', 1023))
        self.black_level_r = int(self.kwargs.get('black_level_r', 0))
        self.black_level_gr = int(self.kwargs.get('black_level_g', 0))
        self.black_level_gb = int(self.kwargs.get('black_level_gb', 0))
        self.black_level_b = int(self.kwargs.get('black_level_b', 0))

    def __rggb_black_level_correction(self) -> np.ndarray: 
        """
        RGGB black level correction
        """
        inputs = self.inputs.astype(np.float32)
        blc_output = inputs.copy()

        blc_output[0::2, 0::2] -= self.black_level_r
        blc_output[0::2, 1::2] -= self.black_level_gr
        blc_output[1::2, 0::2] -= self.black_level_gb
        blc_output[1::2, 1::2] -= self.black_level_b

        return blc_output.astype(np.uint16)
    
    def __grbg_black_level_correction(self) -> np.ndarray: 
        """
        GRBG black level correction
        """
        inputs = self.inputs.astype(np.float32)
        blc_output = inputs.copy()

        blc_output[0::2, 0::2] -= self.black_level_gr
        blc_output[0::2, 1::2] -= self.black_level_b
        blc_output[1::2, 0::2] -= self.black_level_gb
        blc_output[1::2, 1::2] -= self.black_level_r

        return blc_output.astype(np.uint16)
    
    def __bggr_black_level_correction(self) -> np.ndarray: 
        """
        BGGR black level correction
        """
        inputs = self.inputs.astype(np.float32)
        blc_output = inputs.copy()

        blc_output[0::2, 0::2] -= self.black_level_b
        blc_output[0::2, 1::2] -= self.black_level_gb
        blc_output[1::2, 0::2] -= self.black_level_gr
        blc_output[1::2, 1::2] -= self.black_level_r

        return blc_output.astype(np.uint16)
    
    def __gbrg_black_level_correction(self) -> np.ndarray: 
        """
        GBRG black level correction
        """
        inputs = self.inputs.astype(np.float32)
        blc_output = inputs.copy()

        blc_output[0::2, 0::2] -= self.black_level_gb
        blc_output[0::2, 1::2] -= self.black_level_b
        blc_output[1::2, 0::2] -= self.black_level_gr
        blc_output[1::2, 1::2] -= self.black_level_r

        return blc_output.astype(np.uint16)
    
    def run(self) -> np.ndarray: 
        """
        run the BLC
        """
        if self.bayer_pattern == 'RGGB': 
            return self.__rggb_black_level_correction()
        elif self.bayer_pattern == 'GRBG': 
            return self.__grbg_black_level_correction()
        elif self.bayer_pattern == 'BGGR': 
            return self.__bggr_black_level_correction()
        elif self.bayer_pattern == 'GBRG': 
            return self.__gbrg_black_level_correction()
        else: 
            raise ValueError('Invalid bayer pattern')
    
if __name__ == '__main__': 
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    white_level = 1023
    black_level_r = 64
    black_level_gr = 64
    black_level_gb = 64
    black_level_b = 64

    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116185548.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
    blc_output = BLC(inputs = img, white_level = white_level, black_level_r = black_level_r, black_level_gr = black_level_gr, black_level_gb = black_level_gb, black_level_b = black_level_b).run()
    blc_output = np.clip(blc_output, 0, white_level).astype(np.uint16)
    blc_output = cv2.demosaicing(blc_output, cv2.COLOR_BayerRG2BGR)
    blc_output_8 = (blc_output.astype(np.uint32) * 255 // white_level).astype(np.uint8)
    cv2.imwrite(root_path / 'outputs' / '02_blc.png', blc_output_8)