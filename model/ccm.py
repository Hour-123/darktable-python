# @Author: Jiahao Huang
# @Date: 2025-07-09 
# @Description: Color Correction Matrix model

import numpy as np
from typing import Dict, Any, Union

class CCM: 
    """
    CCM model

    description: this model is used to correct the color of the image

    steps: 
        1. get the CC matrix
        2. matmul the CCM matrix with the image
        3. return the corrected image
    """
    def __init__(self, inputs: np.ndarray, **kwargs: Any) -> None:
        self.inputs = inputs
        self.kwargs = kwargs
        self.ccm_matrix = self.kwargs.pop('ccm_matrix', 
                            [[ 1.631906, -0.381807, -0.250099], 
                            [-0.298296, 1.614734, -0.316438], 
                            [0.023770, -0.538501, 1.514732 ]]
                            )
        self.white_level = self.kwargs.pop('white_level', 1023)
    
    def run(self) -> np.ndarray:
        return self.__color_correction()
    

    def __color_correction(self) -> np.ndarray:
        """
        CCM Matrix
        """
        self.inputs = self.inputs.astype(np.float32)
        ccm_matrix = np.array(self.ccm_matrix).T
        ccm_output = np.matmul(self.inputs, ccm_matrix)
        white_level = self.white_level
        return np.clip(ccm_output, 0, white_level).astype(np.uint16)
    
if __name__ == '__main__': 
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    white_level = 1023
    ccm_matrix = [[ 1.631906, -0.381807, -0.250099], 
                [-0.298296, 1.614734, -0.316438], 
                [0.023770, -0.538501, 1.514732 ]]
    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116185548.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
    ccm = CCM(inputs = img, white_level = white_level, ccm_matrix = ccm_matrix)
    ccm_output = ccm.run()
    ccm_output = cv2.demosaicing(ccm_output, cv2.COLOR_BayerRG2RGB)
    ccm_output_8 = (ccm_output.astype(np.uint32) * 255 // white_level).astype(np.uint8)
    cv2.imwrite(root_path / 'outputs' / '05_ccm.png', ccm_output_8)