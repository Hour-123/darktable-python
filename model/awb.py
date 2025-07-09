# @Author: Jiahao Huang
# @Date: 2025-07-09 
# @Description: auto white balance model

import numpy as np
from typing import Dict, Any, Union

class AWB: 
    """
    AWB model

    description: this model is used to correct the white balance of the image

    steps: 
        1. get the mean value of the R, G, B channel
        2. get the gain of the R, G, B channel
        3. get the white balance matrix
        4. output AWB image

    usage: 
        awb = AWB(inputs = img, white_level, r_gain, b_gain).run()
    """
    def __init__(self, inputs: np.ndarray, **kwargs: Any) -> None: 
        self.inputs = inputs
        self.kwargs = kwargs
        self.bayer_pattern = self.kwargs.get('bayer_pattern', 'RGGB')
        self.white_level = int(self.kwargs.get('white_level', 1023))
        self.r_gain = float(self.kwargs.get('r_gain', 1.0))
        self.b_gain = float(self.kwargs.get('b_gain', 1.0))

    def __rggb_auto_white_balance(self) -> np.ndarray: 
        """
        RGGB auto white balance
        """
        inputs = self.inputs.astype(np.float32)
        awb_output = inputs.copy()

        awb_output[0::2, 0::2] *= self.r_gain
        awb_output[1::2, 1::2] *= self.b_gain

        return np.clip(awb_output, 0, self.white_level).astype(np.uint16)

    def __bggr_auto_white_balance(self) -> np.ndarray: 
        """
        BGGR auto white balance
        """
        inputs = self.inputs.astype(np.float32)
        awb_output = inputs.copy()
        
        awb_output[0::2, 0::2] *= self.b_gain
        awb_output[1::2, 1::2] *= self.r_gain

        return np.clip(awb_output, 0, self.white_level).astype(np.uint16)
    
    def __grbg_auto_white_balance(self) -> np.ndarray: 
        """
        GRBG auto white balance
        """
        inputs = self.inputs.astype(np.float32)
        awb_output = inputs.copy()

        awb_output[0::2, 1::2] *= self.r_gain
        awb_output[1::2, 0::2] *= self.b_gain

        return np.clip(awb_output, 0, self.white_level).astype(np.uint16)
    
    def __gbrg_auto_white_balance(self) -> np.ndarray: 
        """
        GBRG auto white balance
        """
        inputs = self.inputs.astype(np.float32)
        awb_output = inputs.copy()

        awb_output[0::2, 1::2] *= self.b_gain
        awb_output[1::2, 0::2] *= self.r_gain

        return np.clip(awb_output, 0, self.white_level).astype(np.uint16)
    
    def run(self) -> np.ndarray: 
        """
        run the AWB
        """
        if self.bayer_pattern == 'RGGB': 
            return self.__rggb_auto_white_balance()
        elif self.bayer_pattern == 'BGGR': 
            return self.__bggr_auto_white_balance()
        elif self.bayer_pattern == 'GRBG': 
            return self.__grbg_auto_white_balance()
        elif self.bayer_pattern == 'GBRG': 
            return self.__gbrg_auto_white_balance()
        else: 
            raise ValueError('Invalid bayer pattern')

if __name__ == '__main__': 
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    white_level = 1023
    r_gain = 2.0
    b_gain = 0.5

    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116185548.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
    awb = AWB(inputs=img)
    awb_output = awb.run()
    awb_output  = cv2.demosaicing(img, cv2.COLOR_BayerRG2RGB)
    awb_output_8 = (awb_output.astype(np.uint32) * 255 // white_level).astype(np.uint8)
    cv2.imwrite(root_path / 'outputs' / '03_awb.png', awb_output_8)