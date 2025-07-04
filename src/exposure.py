"""
Exposure IOP Module - 曝光校正模块

这个模块实现了曝光和亮度调整功能，允许调整图像的整体亮度。
支持手动曝光调整和自适应曝光调整两种模式。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

try:
    from ..core.datatypes import ImageBuffer, DataType, ColorSpace
except ImportError:
    from core.datatypes import ImageBuffer, DataType, ColorSpace


class ExposureMode(Enum):
    """曝光模式枚举"""
    MANUAL = "manual"           # 手动曝光调整
    ADAPTIVE = "adaptive"       # 自适应曝光调整
    HYBRID = "hybrid"           # 混合模式：基于自适应计算的手动调整


@dataclass
class ExposureParams:
    """曝光参数"""
    # 曝光模式
    mode: ExposureMode = ExposureMode.MANUAL
    
    # 手动曝光调整 (EV 值，范围通常 -3.0 到 +3.0)
    exposure: float = 0.0
    
    # 黑电平调整 (范围 -1.0 到 +1.0)
    black_level_correction: float = 0.0
    
    # 自适应曝光参数
    adaptive_target_brightness: float = 0.4  # 目标亮度 [0.0-1.0]
    adaptive_sensitivity: float = 1.0        # 调整敏感度 [0.1-2.0]
    adaptive_max_adjustment: float = 2.0     # 最大调整幅度 (EV)
    
    # 高ISO补偿（自适应模式下使用）
    iso_compensation_enabled: bool = True
    
    # 是否启用
    enabled: bool = True


class Exposure:
    """曝光处理器"""
    
    def __init__(self):
        self.params = ExposureParams()
        
    def set_params(self, params: ExposureParams):
        """设置处理参数"""
        self.params = params
        
    def process(self, image_buffer: ImageBuffer, raw_info: Optional[Dict[str, Any]] = None) -> ImageBuffer:
        """
        处理图像缓冲区
        
        Args:
            image_buffer: 输入图像缓冲区 (RGB 格式)
            raw_info: RAW文件信息（自适应模式下需要）
            
        Returns:
            处理后的图像缓冲区
        """
        if not self.params.enabled:
            return image_buffer
            
        if image_buffer.colorspace != ColorSpace.RGB:
            raise ValueError(f"曝光校正只能处理 RGB 图像，当前色彩空间: {image_buffer.colorspace}")
        
        # 根据模式计算最终的曝光值
        final_exposure = self._calculate_final_exposure(image_buffer, raw_info)
        
        print(f"📸 曝光校正 ({self.params.mode.value}): EV={final_exposure:.2f}, 黑电平={self.params.black_level_correction:.3f}")
        
        # 应用曝光调整
        corrected_data = self._apply_exposure(image_buffer.data, final_exposure)
        
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
    
    def _calculate_final_exposure(self, image_buffer: ImageBuffer, raw_info: Optional[Dict[str, Any]]) -> float:
        """
        根据模式计算最终的曝光值
        
        Args:
            image_buffer: 图像缓冲区
            raw_info: RAW文件信息
            
        Returns:
            最终的曝光调整值（EV）
        """
        if self.params.mode == ExposureMode.MANUAL:
            return self.params.exposure
        
        elif self.params.mode == ExposureMode.ADAPTIVE:
            return self._calculate_adaptive_exposure(image_buffer, raw_info)
        
        elif self.params.mode == ExposureMode.HYBRID:
            # 混合模式：自适应计算 + 手动调整
            adaptive_exposure = self._calculate_adaptive_exposure(image_buffer, raw_info)
            return adaptive_exposure + self.params.exposure
        
        else:
            return self.params.exposure
    
    def _calculate_adaptive_exposure(self, image_buffer: ImageBuffer, raw_info: Optional[Dict[str, Any]]) -> float:
        """
        计算自适应曝光补偿值
        
        Args:
            image_buffer: 图像缓冲区
            raw_info: RAW文件信息
            
        Returns:
            建议的曝光补偿值（EV）
        """
        data = image_buffer.data
        
        # 1. 计算亮度统计
        mean_brightness = float(data.mean())
        median_brightness = float(np.median(data))
        
        # 2. 检测过曝和欠曝像素比例
        overexposed_ratio = np.sum(data > 0.95) / data.size
        underexposed_ratio = np.sum(data < 0.05) / data.size
        
        # 3. 计算建议的曝光补偿
        exposure_compensation = 0.0
        
        # 基于目标亮度的调整
        brightness_diff = self.params.adaptive_target_brightness - mean_brightness
        if abs(brightness_diff) > 0.05:  # 只有当差异明显时才调整
            exposure_compensation += brightness_diff * 2.0 * self.params.adaptive_sensitivity
        
        # 基于欠曝比例的调整
        if underexposed_ratio > 0.1:  # 超过10%的像素欠曝
            exposure_compensation += underexposed_ratio * 2.0 * self.params.adaptive_sensitivity
        
        # 基于过曝比例的调整
        if overexposed_ratio > 0.05:  # 超过5%的像素过曝
            exposure_compensation -= overexposed_ratio * 3.0 * self.params.adaptive_sensitivity
        
        # 基于ISO值的调整（如果启用且有相关信息）
        if self.params.iso_compensation_enabled and raw_info:
            iso_speed = raw_info.get('iso_speed')
            if isinstance(iso_speed, int):
                if iso_speed > 1600:
                    exposure_compensation -= 0.3 * self.params.adaptive_sensitivity
                elif iso_speed > 800:
                    exposure_compensation -= 0.1 * self.params.adaptive_sensitivity
        
        # 限制调整范围
        exposure_compensation = np.clip(exposure_compensation, 
                                      -self.params.adaptive_max_adjustment, 
                                      self.params.adaptive_max_adjustment)
        
        # 输出分析信息（可选）
        if hasattr(self, '_verbose') and self._verbose:
            self._print_adaptive_analysis(mean_brightness, median_brightness, 
                                        overexposed_ratio, underexposed_ratio, 
                                        raw_info, exposure_compensation)
        
        return exposure_compensation
    
    def _print_adaptive_analysis(self, mean_brightness: float, median_brightness: float,
                               overexposed_ratio: float, underexposed_ratio: float,
                               raw_info: Optional[Dict[str, Any]], exposure_compensation: float):
        """打印自适应曝光分析信息"""
        print(f"📊 自适应曝光分析:")
        print(f"   目标亮度: {self.params.adaptive_target_brightness:.3f}")
        print(f"   当前平均亮度: {mean_brightness:.3f}")
        print(f"   中位数亮度: {median_brightness:.3f}")
        print(f"   过曝像素比例: {overexposed_ratio:.3f}")
        print(f"   欠曝像素比例: {underexposed_ratio:.3f}")
        if raw_info:
            iso_speed = raw_info.get('iso_speed', 'N/A')
            print(f"   ISO值: {iso_speed}")
        print(f"   计算的曝光补偿: {exposure_compensation:.2f} EV")
        print()
    
    def enable_verbose_analysis(self, enabled: bool = True):
        """启用或禁用详细分析输出"""
        self._verbose = enabled
    
    def get_adaptive_exposure_suggestion(self, image_buffer: ImageBuffer, 
                                       raw_info: Optional[Dict[str, Any]] = None) -> float:
        """
        获取自适应曝光建议值（不实际应用）
        
        Args:
            image_buffer: 图像缓冲区
            raw_info: RAW文件信息
            
        Returns:
            建议的曝光补偿值（EV）
        """
        return self._calculate_adaptive_exposure(image_buffer, raw_info)
    
    def _apply_exposure(self, data: np.ndarray, exposure_ev: float) -> np.ndarray:
        """
        应用曝光调整
        
        Args:
            data: 图像数据 (H, W, C)
            exposure_ev: 曝光调整值（EV）
            
        Returns:
            处理后的图像数据
        """
        # 复制数据
        result = data.copy().astype(np.float32)
        
        # 计算曝光倍数 (EV = log2(倍数))
        exposure_multiplier = 2.0 ** exposure_ev
        
        # 应用黑电平调整 (先减去黑电平，调整后再加回)
        if self.params.black_level_correction != 0:
            result = result + self.params.black_level_correction
            # 确保不会低于0
            result = np.maximum(result, 0.0)
        
        # 应用曝光倍数
        result = result * exposure_multiplier
        
        # 限制到合理范围
        if data.dtype == np.float32 or data.dtype == np.float64:
            # 浮点数据，限制在 [0, 1] 范围
            result = np.clip(result, 0.0, 1.0)
        else:
            # 整数数据，限制在原始数据类型范围内
            dtype_info = np.iinfo(data.dtype)
            result = np.clip(result, dtype_info.min, dtype_info.max)
            result = result.astype(data.dtype)
            
        return result
    
    def __str__(self) -> str:
        if self.params.mode == ExposureMode.MANUAL:
            return f"Exposure(manual={self.params.exposure:.2f}EV, enabled={self.params.enabled})"
        elif self.params.mode == ExposureMode.ADAPTIVE:
            return f"Exposure(adaptive, target={self.params.adaptive_target_brightness:.2f}, enabled={self.params.enabled})"
        else:
            return f"Exposure(hybrid, manual={self.params.exposure:.2f}EV, target={self.params.adaptive_target_brightness:.2f}, enabled={self.params.enabled})" 