# @Author: Jiahao Huang
# @Date: 2025-07-09 
# @Description: Color filter array model

import numpy as np
from typing import Dict, Any, Union
import cv2

class CFA: 
    """
    CFA model

    description: this model only accomplish malvar method, opencv method. 
    """
    def __init__(self, inputs: np.ndarray, **kwargs: Any) -> None: 
        self.inputs = inputs
        self.kwargs = kwargs
        self.bayer_pattern = self.kwargs.get('bayer_pattern', 'RGGB')
        self.white_level = int(self.kwargs.get('white_level', 1023))
        self.cfa_method = self.kwargs.get('cfa_method', 'opencv')

    def __padding_inputs(self, inputs: np.ndarray, padding: int) -> np.ndarray: 
        """
        padding the inputs
        """
        inputs = np.pad(inputs, ((padding, padding), (padding, padding)), 'reflect')
        return inputs
    
    def __opencv_demosaic(self) -> np.ndarray: 
        """
        use opencv demosaic alogrithm
        """
        if self.bayer_pattern == 'RGGB':
            demosaic_output = cv2.cvtColor(self.inputs, cv2.COLOR_BayerRGGB2RGB)
        elif self.bayer_pattern == 'BGGR': 
            demosaic_output = cv2.cvtColor(self.inputs, cv2.COLOR_BayerBGGR2RGB)
        elif self.bayer_pattern == 'GRBG':
            demosaic_output = cv2.cvtColor(self.inputs, cv2.COLOR_BayerGR2RGB)
        elif self.bayer_pattern == 'GBRG':
            demosaic_output = cv2.cvtColor(self.inputs, cv2.COLOR_BayerGB2RGB)
        return np.clip(demosaic_output, 0, self.white_level).astype(np.uint16)

    def __malvar_demosaic(self) -> np.ndarray: 
        """
        use malvar demosaic alogrithm
        """
        H, W = self.inputs.shape
        padding = 2

        self.inputs = self.inputs.astype(np.float32)
        demosaic_output = np.zeros((H, W, 3), dtype = np.float32)
        
        self.inputs = self.__padding_inputs(self.inputs, padding)

        for i in range(padding, H + padding, 2): 
            for j in range(padding, W + padding, 2): 
                if self.bayer_pattern == 'RGGB':
                    demosaic_output[i - padding, j - padding] = self.__get_malvar_pixel(color_type = 'r', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_malvar_pixel(color_type = 'gr', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_malvar_pixel(color_type = 'gb', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_malvar_pixel(color_type = 'b', center_y = i + 1, center_x = j + 1)
                elif self.bayer_pattern == 'BGGR':  
                    demosaic_output[i - padding, j - padding] = self.__get_malvar_pixel(color_type = 'b', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_malvar_pixel(color_type = 'gb', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_malvar_pixel(color_type = 'gr', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_malvar_pixel(color_type = 'r', center_y = i + 1, center_x = j + 1)
                elif self.bayer_pattern == 'GRBG':
                    demosaic_output[i - padding, j - padding] = self.__get_malvar_pixel(color_type = 'gr', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_malvar_pixel(color_type = 'r', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_malvar_pixel(color_type = 'b', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_malvar_pixel(color_type = 'gb', center_y = i + 1, center_x = j + 1)
                elif self.bayer_pattern == 'GBRG':
                    demosaic_output[i - padding, j - padding] = self.__get_malvar_pixel(color_type = 'gb', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_malvar_pixel(color_type = 'b', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_malvar_pixel(color_type = 'r', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_malvar_pixel(color_type = 'r', center_y = i + 1, center_x = j + 1)
                else:
                    raise ValueError('Invalid bayer pattern')
        return np.clip(demosaic_output, 0, self.white_level).astype(np.uint16)
    
    def __get_malvar_pixel(self, color_type: str, center_y: int, center_x: int) -> np.ndarray:
        """
        get the malvar pixel
        """
        if color_type == 'r':
            r_pixel = self.inputs[center_y, center_x]
            g_pixel = 4 * self.inputs[center_y, center_x] - self.inputs[center_y - 2, center_x] - self.inputs[center_y + 2, center_x] - self.inputs[center_y, center_x - 2] - self.inputs[center_y, center_x + 2] \
                + 2 * (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x] + self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1])
            b_pixel = 6 * self.inputs[center_y, center_x] - 2 * (self.inputs[center_y - 2, center_x] + self.inputs[center_y + 2, center_x] + self.inputs[center_y, center_x - 2] + self.inputs[center_y, center_x + 2]) \
                + 2 * (self.inputs[center_y - 1, center_x - 1] + self.inputs[center_y - 1, center_x + 1] + self.inputs[center_y + 1, center_x - 1] + self.inputs[center_y + 1, center_x + 1]) 
            g_pixel /= 8
            b_pixel /= 8
            return np.array([r_pixel, g_pixel, b_pixel])
        elif color_type == 'gr':
            g_pixel = self.inputs[center_y, center_x]
            r_pixel = 5 * self.inputs[center_y, center_x] - self.inputs[center_y, center_x - 2] - self.inputs[center_y, center_x + 2] - self.inputs[center_y - 1, center_x - 1] - self.inputs[center_y - 1, center_x + 1] \
                - self.inputs[center_y + 1, center_x - 1] - self.inputs[center_y + 1, center_x + 1] + 0.5 * (self.inputs[center_y - 2, center_x] + self.inputs[center_y + 2, center_x] ) \
                + 4 * (self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1])
            b_pixel = 5 * self.inputs[center_y, center_x] - self.inputs[center_y - 2, center_x] - self.inputs[center_y + 2, center_x] - self.inputs[center_y - 1, center_x - 1] - self.inputs[center_y -1, center_x + 1] \
                - self.inputs[center_y + 1, center_x + 1] - self.inputs[center_y + 1, center_x - 1] + 0.5 * (self.inputs[center_y, center_x - 2] + self.inputs[center_y, center_x + 2]) \
                + 4 * (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x])
            r_pixel /= 8
            b_pixel /= 8
            return np.array([r_pixel, g_pixel, b_pixel])
        elif color_type == 'gb':
            g_pixel = self.inputs[center_y, center_x]
            r_pixel = 5 * self.inputs[center_y, center_x] - self.inputs[center_y - 2, center_x] - self.inputs[center_y + 2, center_x] - self.inputs[center_y - 1, center_x - 1] - self.inputs[center_y + 1, center_x - 1] \
                - self.inputs[center_y - 1, center_x + 1] - self.inputs[center_y + 1, center_x + 1] + 0.5 * (self.inputs[center_y, center_x - 2] + self.inputs[center_y, center_x + 2]) \
                + 4 * (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x])
            b_pixel = 5 * self.inputs[center_y, center_x] - self.inputs[center_y, center_x - 2] - self.inputs[center_y, center_x + 2] - self.inputs[center_y - 1, center_x - 1] - self.inputs[center_y - 1, center_x + 1] \
                - self.inputs[center_y + 1, center_x - 1] - self.inputs[center_y + 1, center_x + 1] + 0.5 * (self.inputs[center_y - 2, center_x] + self.inputs[center_y + 2, center_x]) \
                + 4 * (self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1])
            r_pixel /= 8
            b_pixel /= 8
            return np.array([r_pixel, g_pixel, b_pixel])
        elif color_type == 'b':
            b_pixel = self.inputs[center_y, center_x]
            g_pixel = 4 * self.inputs[center_y, center_x] - self.inputs[center_y - 2, center_x] - self.inputs[center_y + 2, center_x] - self.inputs[center_y, center_x - 2] - self.inputs[center_y, center_x + 2] \
                + 2 * (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x] + self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1])
            r_pixel = 6 * self.inputs[center_y, center_x] - 2 * (self.inputs[center_y - 2, center_x] + self.inputs[center_y + 2, center_x] + self.inputs[center_y, center_x - 2] + self.inputs[center_y, center_x + 2]) \
                + 2 * (self.inputs[center_y - 1, center_x - 1] + self.inputs[center_y - 1, center_x + 1] + self.inputs[center_y + 1, center_x - 1] + self.inputs[center_y + 1, center_x + 1]) 
            g_pixel /= 8
            r_pixel /= 8
            return np.array([r_pixel, g_pixel, b_pixel])
        else:
            raise ValueError('Invalid color type')
        
    def run(self) -> np.ndarray: 
        """
        run the CFA
        """
        if self.cfa_method == 'opencv':
            return self.__opencv_demosaic()
        elif self.cfa_method == 'malvar':
            return self.__malvar_demosaic()
        else:
            raise ValueError('Invalid cfa method')
    
if __name__ == '__main__': 
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    white_level = 1023
    cfa_method = 'malvar'
    bayer_pattern = 'RGGB'

    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116185548.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
    cfa = CFA(inputs = img, white_level = white_level, cfa_method = cfa_method, bayer_pattern = bayer_pattern)
    cfa_output = cfa.run()
    # cfa_output: H×W×3, uint16 / float32 / float64 均可
    cfa_vis = (np.clip(cfa_output, 0, white_level) * 255 // white_level).astype(np.uint8)
    
    # 增强图像亮度和对比度以便更好地可视化
    cfa_vis = cv2.normalize(cfa_vis, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite(root_path / 'outputs' / '04_cfa.png', cfa_vis)
