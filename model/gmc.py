# @Author: Jiahao Huang
# @Date: 2025-07-09
# @Description: Gamma Correction Module


import numpy as np  
from typing import Any, Dict, List, Optional, Tuple, Union  


class GMC:
    """
    Gamma Correction
    
    description:
        this is a class for Gamma Correction, the gamma value is 0.6, you can change it
    
    step:
        1. get the gamma table
        2. get the gamma output
        
    usage:
        gma = GMC(inputs, gamma=0.6)
    """
    def __init__(self, inputs: np.ndarray, **kwargs: Any) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.gamma = float(self.kwargs.get('gamma', 2.2))  # 默认伽玛 2.2
        # 传感器白电平，用于把输入缩放到 0-255 区间
        self.white_level = int(self.kwargs.get('white_level', 1023))
        self.gamma_table = self.__gamma_table()
        
    def run(self) -> np.ndarray:
        return self.__gamma_correction()
    
    def __gamma_correction(self) -> np.ndarray:
        """
        根据 white_level 把输入映射到 0-255 作为查表索引，再把 8-bit 结果拉回原位深。
        这样无需为高达 16000 或 65535 的位深单独构造超大 LUT。
        """
        # 1) 把输入缩放到 0-255 区间并四舍五入
        indices = (self.inputs.astype(np.float32) * 255.0 / self.white_level + 0.5).astype(np.uint16)
        indices = np.clip(indices, 0, 255)

        # 2) 查 8-bit gamma LUT
        gamma_8bit = self.gamma_table[indices]

        # 3) 将 8-bit 结果拉伸回原位深，保持数据类型一致
        gamma_output = (gamma_8bit.astype(np.uint32) * self.white_level // 255).astype(self.inputs.dtype)

        return gamma_output
    
    
    def __gamma_table(self) -> np.ndarray:
        """
        Gamma Table
        """
        curve = lambda x : x ** (1.0 / self.gamma)
        gamma_table = np.zeros(256, dtype=np.uint8)
        for i in range(0, 256):
            gamma_table[i] = np.clip(curve(float(i) / 255) * 255 , 0, 255)
        return gamma_table
    

