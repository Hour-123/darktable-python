"""
White Balance IOP Module - 白平衡处理模块

这个模块实现了白平衡校正功能，用于纠正不同光源条件下的色彩偏移。
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

try:
    from ..core.datatypes import ImageBuffer, DataType, ColorSpace
except ImportError:
    from core.datatypes import ImageBuffer, DataType, ColorSpace


class WhiteBalanceMethod(Enum):
    """白平衡方法枚举"""
    CAMERA = "camera"           # 使用相机元数据
    MANUAL = "manual"           # 手动设置倍数
    AUTO = "auto"               # 自动白平衡（未实现）
    DAYLIGHT = "daylight"       # 使用日光倍数
    GRAY_WORLD = "gray_world"   # 灰度世界算法
    WHITE_PATCH = "white_patch" # 白色区域检测
    ADAPTIVE = "adaptive"       # 自适应白平衡


@dataclass
class WhiteBalanceParams:
    """白平衡参数"""
    method: WhiteBalanceMethod = WhiteBalanceMethod.CAMERA
    
    # 手动倍数 (R, G, B)
    manual_multipliers: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # 相机倍数 (从 RAW 元数据获取)
    camera_multipliers: Optional[List[float]] = None
    
    # 日光倍数 (从 RAW 元数据获取)
    daylight_multipliers: Optional[List[float]] = None
    
    # 灰度世界算法参数
    gray_world_percentile: float = 50.0  # 使用的百分位数
    
    # 白色区域检测参数
    white_patch_percentile: float = 99.0  # 白色区域的百分位数阈值
    white_patch_min_size: int = 100      # 白色区域的最小像素数
    
    # 自适应算法参数
    adaptive_sensitivity: float = 0.5    # 自适应敏感度 0.6表示60%依赖白色区域检测，40%依赖灰度世界
    adaptive_smooth_factor: float = 0.3  # 平滑因子 0.4表示40%校正强度，60%保持原状
    
    # 是否启用
    enabled: bool = True


class WhiteBalance:
    """白平衡处理器"""
    
    def __init__(self):
        self.params = WhiteBalanceParams()
        
    def set_params(self, params: WhiteBalanceParams):
        """设置处理参数"""
        self.params = params
        
    def process(self, image_buffer: ImageBuffer) -> ImageBuffer:
        """
        处理图像缓冲区
        
        Args:
            image_buffer: 输入图像缓冲区 (必须是 RGB 格式)
            
        Returns:
            处理后的图像缓冲区
        """
        if not self.params.enabled:
            return image_buffer
            
        if image_buffer.colorspace != ColorSpace.RGB:
            raise ValueError(f"白平衡只能处理 RGB 图像，当前色彩空间: {image_buffer.colorspace}")
            
        if image_buffer.channels != 3:
            raise ValueError(f"白平衡需要 3 通道图像，当前通道数: {image_buffer.channels}")
        
        # 获取白平衡倍数
        multipliers = self._get_multipliers()
        
        # 如果是基于图像数据的算法，需要从图像中计算倍数
        if multipliers is None:
            if self.params.method == WhiteBalanceMethod.GRAY_WORLD:
                multipliers = self._calculate_gray_world_multipliers(image_buffer.data)
            elif self.params.method == WhiteBalanceMethod.WHITE_PATCH:
                multipliers = self._calculate_white_patch_multipliers(image_buffer.data)
            elif self.params.method == WhiteBalanceMethod.ADAPTIVE:
                multipliers = self._calculate_adaptive_multipliers(image_buffer.data)
            else:
                print("⚠️  无法计算白平衡倍数，使用默认值")
                multipliers = (1.0, 1.0, 1.0)
        
        if multipliers is None:
            print("⚠️  无法获取白平衡倍数，跳过处理")
            return image_buffer
            
        print(f"🎨 应用白平衡倍数: R={multipliers[0]:.3f}, G={multipliers[1]:.3f}, B={multipliers[2]:.3f}")
        
        # 应用白平衡
        corrected_data = self._apply_white_balance(image_buffer.data, multipliers)
        
        # 创建新的图像缓冲区
        return ImageBuffer(
            data=corrected_data,
            width=image_buffer.width,
            height=image_buffer.height,
            channels=image_buffer.channels,
            datatype=image_buffer.datatype,
            colorspace=image_buffer.colorspace,
            bayer_pattern=image_buffer.bayer_pattern,
            black_level=image_buffer.black_level,
            white_point=image_buffer.white_point
        )
    
    def _get_multipliers(self) -> Optional[Tuple[float, float, float]]:
        """获取白平衡倍数"""
        if self.params.method == WhiteBalanceMethod.MANUAL:
            return self.params.manual_multipliers
            
        elif self.params.method == WhiteBalanceMethod.CAMERA:
            if self.params.camera_multipliers and len(self.params.camera_multipliers) >= 3:
                return self._normalize_multipliers(
                    self.params.camera_multipliers[0],  # R
                    self.params.camera_multipliers[1],  # G
                    self.params.camera_multipliers[2]   # B
                )
            else:
                print("⚠️  相机倍数不可用，回退到手动倍数")
                return self.params.manual_multipliers
                
        elif self.params.method == WhiteBalanceMethod.DAYLIGHT:
            if self.params.daylight_multipliers and len(self.params.daylight_multipliers) >= 3:
                return self._normalize_multipliers(
                    self.params.daylight_multipliers[0],  # R
                    self.params.daylight_multipliers[1],  # G
                    self.params.daylight_multipliers[2]   # B
                )
            else:
                print("⚠️  日光倍数不可用，回退到手动倍数")
                return self.params.manual_multipliers
                
        elif self.params.method == WhiteBalanceMethod.GRAY_WORLD:
            # 灰度世界算法需要在process方法中获取图像数据
            return None  # 在process方法中处理
            
        elif self.params.method == WhiteBalanceMethod.WHITE_PATCH:
            # 白色区域检测算法需要在process方法中获取图像数据
            return None  # 在process方法中处理
            
        elif self.params.method == WhiteBalanceMethod.ADAPTIVE:
            # 自适应算法需要在process方法中获取图像数据
            return None  # 在process方法中处理
                
        elif self.params.method == WhiteBalanceMethod.AUTO:
            # TODO: 实现自动白平衡算法
            print("⚠️  自动白平衡尚未实现，使用手动倍数")
            return self.params.manual_multipliers
            
        return None
    
    def _normalize_multipliers(self, r: float, g: float, b: float) -> Tuple[float, float, float]:
        """
        归一化白平衡倍数，以绿色通道为基准
        
        Args:
            r, g, b: 原始倍数
            
        Returns:
            归一化后的倍数 (R, G, B)
        """
        if g == 0:
            print("⚠️  绿色通道倍数为0，无法归一化")
            return (1.0, 1.0, 1.0)
            
        return (r / g, 1.0, b / g)
    
    def _apply_white_balance(self, data: np.ndarray, multipliers: Tuple[float, float, float]) -> np.ndarray:
        """
        应用白平衡倍数到图像数据
        
        Args:
            data: 图像数据 (H, W, 3)
            multipliers: RGB 倍数
            
        Returns:
            处理后的图像数据
        """
        # 复制数据
        result = data.copy()
        
        # 应用倍数
        result[:, :, 0] *= multipliers[0]  # R 通道
        result[:, :, 1] *= multipliers[1]  # G 通道
        result[:, :, 2] *= multipliers[2]  # B 通道
        
        # 确保数据在合理范围内
        if data.dtype == np.float32 or data.dtype == np.float64:
            # 浮点数据，限制在 [0, 1] 范围
            result = np.clip(result, 0.0, 1.0)
        else:
            # 整数数据，限制在原始数据类型范围内
            dtype_info = np.iinfo(data.dtype)
            result = np.clip(result, dtype_info.min, dtype_info.max)
            
        return result
    
    def _calculate_gray_world_multipliers(self, data: np.ndarray) -> Tuple[float, float, float]:
        """
        使用灰度世界算法计算白平衡倍数
        
        Args:
            data: RGB 图像数据 (H, W, 3)
            
        Returns:
            白平衡倍数 (R, G, B)
        """
        # 计算每个通道的统计值
        percentile = self.params.gray_world_percentile
        
        r_val = float(np.percentile(data[:, :, 0], percentile))
        g_val = float(np.percentile(data[:, :, 1], percentile))
        b_val = float(np.percentile(data[:, :, 2], percentile))
        
        # 避免除零错误
        if r_val == 0 or g_val == 0 or b_val == 0:
            print("⚠️  灰度世界算法：检测到零值，使用默认倍数")
            return (1.0, 1.0, 1.0)
        
        # 以绿色通道为基准归一化
        r_mult = g_val / r_val
        g_mult = 1.0
        b_mult = g_val / b_val
        
        print(f"🌍 灰度世界算法计算结果: R={r_mult:.3f}, G={g_mult:.3f}, B={b_mult:.3f}")
        
        return (r_mult, g_mult, b_mult)
    
    def _calculate_white_patch_multipliers(self, data: np.ndarray) -> Tuple[float, float, float]:
        """
        使用白色区域检测算法计算白平衡倍数
        
        Args:
            data: RGB 图像数据 (H, W, 3)
            
        Returns:
            白平衡倍数 (R, G, B)
        """
        # 计算总亮度
        brightness = np.mean(data, axis=2)
        
        # 找到最亮的像素
        threshold = np.percentile(brightness, self.params.white_patch_percentile)
        white_mask = brightness >= threshold
        
        # 检查白色区域大小
        white_pixel_count = np.sum(white_mask)
        if white_pixel_count < self.params.white_patch_min_size:
            print(f"⚠️  白色区域检测：白色区域太小({white_pixel_count}像素)，使用默认倍数")
            return (1.0, 1.0, 1.0)
        
        # 计算白色区域的平均RGB值
        white_r = np.mean(data[white_mask, 0])
        white_g = np.mean(data[white_mask, 1])
        white_b = np.mean(data[white_mask, 2])
        
        # 避免除零错误
        if white_r == 0 or white_g == 0 or white_b == 0:
            print("⚠️  白色区域检测：检测到零值，使用默认倍数")
            return (1.0, 1.0, 1.0)
        
        # 以绿色通道为基准归一化
        r_mult = white_g / white_r
        g_mult = 1.0
        b_mult = white_g / white_b
        
        print(f"🔍 白色区域检测结果: R={r_mult:.3f}, G={g_mult:.3f}, B={b_mult:.3f} ({white_pixel_count}像素)")
        
        return (r_mult, g_mult, b_mult)
    
    def _calculate_adaptive_multipliers(self, data: np.ndarray) -> Tuple[float, float, float]:
        """
        使用自适应算法计算白平衡倍数
        
        结合灰度世界和白色区域检测的结果
        
        Args:
            data: RGB 图像数据 (H, W, 3)
            
        Returns:
            白平衡倍数 (R, G, B)
        """
        # 获取灰度世界结果
        gray_world_mults = self._calculate_gray_world_multipliers(data)
        
        # 获取白色区域检测结果
        white_patch_mults = self._calculate_white_patch_multipliers(data)
        
        # 使用加权平均融合结果
        sensitivity = self.params.adaptive_sensitivity
        smooth_factor = self.params.adaptive_smooth_factor
        
        # 计算加权平均
        r_mult = gray_world_mults[0] * (1 - sensitivity) + white_patch_mults[0] * sensitivity
        g_mult = gray_world_mults[1] * (1 - sensitivity) + white_patch_mults[1] * sensitivity
        b_mult = gray_world_mults[2] * (1 - sensitivity) + white_patch_mults[2] * sensitivity
        
        # 应用平滑因子
        r_mult = r_mult * smooth_factor + 1.0 * (1 - smooth_factor)
        g_mult = g_mult * smooth_factor + 1.0 * (1 - smooth_factor)
        b_mult = b_mult * smooth_factor + 1.0 * (1 - smooth_factor)
        
        print(f"🎯 自适应算法结果: R={r_mult:.3f}, G={g_mult:.3f}, B={b_mult:.3f}")
        
        return (r_mult, g_mult, b_mult)
    
    def __str__(self) -> str:
        return f"WhiteBalance(method={self.params.method.value}, enabled={self.params.enabled})" 