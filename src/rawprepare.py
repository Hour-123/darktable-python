"""
RAW 预处理模块
对应 darktable 的 rawprepare IOP

主要功能：
1. 黑电平校正和白点归一化
2. 图像裁剪
3. GainMap/平场校正 (DNG规范)
4. X-Trans滤镜支持
5. 多种数据流处理
"""

import numpy as np
from typing import Tuple, Optional, List
from enum import Enum
try:
    from ..core.datatypes import (
        ImageBuffer, ROI, RawPrepareParams, DataType, ColorSpace, 
        BayerPattern, InvalidDataError
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from core.datatypes import (
        ImageBuffer, ROI, RawPrepareParams, DataType, ColorSpace, 
        BayerPattern, InvalidDataError
    )


class FlatFieldType(Enum):
    """平场校正类型"""
    OFF = 0          # 禁用
    EMBEDDED = 1     # 嵌入式 GainMap


class GainMap:
    """DNG GainMap 数据结构"""
    def __init__(self, 
                 map_gain: np.ndarray,
                 map_points_h: int,
                 map_points_v: int,
                 map_spacing_h: float,
                 map_spacing_v: float,
                 map_origin_h: float = 0.0,
                 map_origin_v: float = 0.0,
                 filter_id: int = 0):
        self.map_gain = map_gain # 增益值   
        self.map_points_h = map_points_h # 水平方向的点数
        self.map_points_v = map_points_v # 垂直方向的点数
        self.map_spacing_h = map_spacing_h # 水平方向的间距
        self.map_spacing_v = map_spacing_v # 垂直方向的间距
        self.map_origin_h = map_origin_h # 水平方向的偏移
        self.map_origin_v = map_origin_v # 垂直方向的偏移
        self.filter_id = filter_id # 滤镜ID


class RawPrepare:
    """
    RAW 预处理模块
    
    主要功能：
    1. 黑电平校正和白点归一化
    2. 图像裁剪
    3. GainMap/平场校正 (DNG规范) 
    4. X-Trans滤镜支持
    5. 数据类型转换 (uint16 -> float32)
    6. 多通道数据处理
    """
    
    def __init__(self):
        self.name = "rawprepare"
        self.gainmaps: Optional[List[GainMap]] = None
        self.flat_field_type = FlatFieldType.OFF # 选择是否应用平场校正 OFF: 不应用平场校正 EMBEDDED: 应用平场校正
        
    def process(self, 
                input_buffer: ImageBuffer, 
                params: RawPrepareParams,
                roi_in: ROI,
                roi_out: ROI) -> ImageBuffer:
        """
        处理 RAW 数据! 
        
        Args:
            input_buffer: 输入图像缓冲区
            params: rawprepare 参数
            roi_in: 输入感兴趣区域
            roi_out: 输出感兴趣区域
            
        Returns:
            处理后的图像缓冲区
        """
        # 验证输入数据类型
        if input_buffer.channels == 1 and not input_buffer.is_raw:
            raise InvalidDataError("Single-channel data should be RAW format")
            
        # 计算有效裁剪区域
        crop_x = self._compute_proper_crop(roi_in, params.crop_left)
        crop_y = self._compute_proper_crop(roi_in, params.crop_top)
        
        # 处理不同的输入数据类型和通道数
        if input_buffer.channels == 1:
            # RAW 马赛克数据
            if input_buffer.datatype == DataType.UINT16:
                output_buffer = self._process_uint16_mosaic(input_buffer, params, roi_in, roi_out, crop_x, crop_y)
            elif input_buffer.datatype == DataType.FLOAT32:
                output_buffer = self._process_float32_mosaic(input_buffer, params, roi_in, roi_out, crop_x, crop_y)
            else:
                raise InvalidDataError(f"Unsupported input data type: {input_buffer.datatype}")
            
            # 应用 GainMap（如果启用）
            if self.flat_field_type == FlatFieldType.EMBEDDED and self.gainmaps:
                output_buffer = self._apply_gainmap(output_buffer, roi_out, crop_x, crop_y)
                
            return output_buffer
        else:
            # 预下采样的多通道数据
            return self._process_multichannel(input_buffer, params, roi_in, roi_out, crop_x, crop_y)
    
    def _process_uint16_mosaic(self, 
                       input_buffer: ImageBuffer,
                       params: RawPrepareParams,
                       roi_in: ROI,
                       roi_out: ROI,
                       crop_x: int,
                       crop_y: int) -> ImageBuffer:
        """处理 uint16 RAW 马赛克数据"""
        
        input_data = input_buffer.data # 输入数据是uint16类型
        output_data = np.zeros((roi_out.height, roi_out.width), dtype=np.float32) # 最终输出数据是float32类型
        
        # 计算每个 Bayer 位置的黑电平和归一化参数
        sub, div = self._compute_normalization_params(params)
        
        # 处理每个像素
        # 裁剪偏移 ➜ 找到 Bayer 位置 ➜ 扣黑 ➜ 归一化
        for j in range(roi_out.height):
            for i in range(roi_out.width):
                # 输入和输出像素位置
                pin_y = j + crop_y
                pin_x = i + crop_x
                
                if (pin_y < input_data.shape[0] and pin_x < input_data.shape[1]):
                    # 确定 Bayer 位置 (R=0, G1=1, G2=2, B=3)
                    bayer_id = self._get_bayer_id(roi_out, params, j, i)
                    
                    # 应用黑电平校正和归一化
                    pixel_value = float(input_data[pin_y, pin_x])
                    output_data[j, i] = (pixel_value - sub[bayer_id]) / div[bayer_id]
        
        # 确保输出在有效范围内
        output_data = np.clip(output_data, 0.0, 1.0)
        
        return ImageBuffer(
            data=output_data,
            width=roi_out.width,
            height=roi_out.height,
            channels=1,
            datatype=DataType.FLOAT32,
            colorspace=ColorSpace.RAW,
            filters=input_buffer.filters,
            bayer_pattern=input_buffer.bayer_pattern,
            xtrans=input_buffer.xtrans,
            black_level=0.0,
            white_point=1.0
        )
    
    def _process_float32_mosaic(self, 
                        input_buffer: ImageBuffer,
                        params: RawPrepareParams,
                        roi_in: ROI,
                        roi_out: ROI,
                        crop_x: int,
                        crop_y: int) -> ImageBuffer:
        """处理 float32 RAW 马赛克数据（未归一化）"""
        
        input_data = input_buffer.data
        output_data = np.zeros((roi_out.height, roi_out.width), dtype=np.float32)
        
        # 计算归一化参数
        sub, div = self._compute_normalization_params(params)
        
        # 处理每个像素
        for j in range(roi_out.height):
            for i in range(roi_out.width):
                pin_y = j + crop_y
                pin_x = i + crop_x
                
                if (pin_y < input_data.shape[0] and pin_x < input_data.shape[1]):
                    bayer_id = self._get_bayer_id(roi_out, params, j, i)
                    pixel_value = input_data[pin_y, pin_x]
                    output_data[j, i] = (pixel_value - sub[bayer_id]) / div[bayer_id]
        
        output_data = np.clip(output_data, 0.0, 1.0)
        
        return ImageBuffer(
            data=output_data,
            width=roi_out.width,
            height=roi_out.height,
            channels=1,
            datatype=DataType.FLOAT32,
            colorspace=ColorSpace.RAW,
            filters=input_buffer.filters,
            bayer_pattern=input_buffer.bayer_pattern,
            xtrans=input_buffer.xtrans,
            black_level=0.0,
            white_point=1.0
        )
    
    def _compute_normalization_params(self, params: RawPrepareParams) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算归一化参数
        
        Returns:
            sub: 每个 Bayer 位置的黑电平
            div: 每个 Bayer 位置的归一化因子
        """
        sub = np.array(params.black_level_separate, dtype=np.float32)
        white_point = float(params.white_point)
        
        # 对于uint16数据，直接使用原始值计算
        div = np.array([white_point - bl for bl in params.black_level_separate], dtype=np.float32)
        
        # 避免除零
        div = np.maximum(div, 0.001)
        
        return sub, div
    
    def _get_bayer_id(self, roi: ROI, params: RawPrepareParams, row: int, col: int) -> int:
        """
        获取 Bayer 位置 ID
        
        Args:
            roi: 输出区域
            params: 参数
            row: 行索引
            col: 列索引
            
        Returns:
            Bayer ID (0=R, 1=G1, 2=G2, 3=B)
        """
        # 考虑裁剪偏移
        abs_row = row + roi.y + params.crop_top
        abs_col = col + roi.x + params.crop_left
        
        # 计算 Bayer 位置 (RGGB 模式)
        return ((abs_row & 1) << 1) + (abs_col & 1)
    
    def _compute_proper_crop(self, roi: ROI, crop_value: int) -> int:
        """
        计算有效的裁剪值
        
        Args:
            roi: 感兴趣区域
            crop_value: 裁剪值
            
        Returns:
            有效的裁剪值
        """
        # 根据缩放比例调整裁剪值
        scale = roi.scale
        return max(0, int(crop_value * scale))
    
    def modify_roi_in(self, roi_out: ROI, params: RawPrepareParams) -> ROI:
        """
        根据输出 ROI 计算输入 ROI
        
        Args:
            roi_out: 输出感兴趣区域
            params: rawprepare 参数
            
        Returns:
            输入感兴趣区域
        """
        # 计算总的裁剪量
        total_crop_x = params.crop_left + params.crop_right
        total_crop_y = params.crop_top + params.crop_bottom
        
        # 根据缩放调整裁剪量
        scale = roi_out.scale
        adjusted_crop_x = int(total_crop_x * scale)
        adjusted_crop_y = int(total_crop_y * scale)
        
        return ROI(
            x=roi_out.x,
            y=roi_out.y,
            width=roi_out.width + adjusted_crop_x,
            height=roi_out.height + adjusted_crop_y,
            scale=roi_out.scale
        )
    
    def modify_roi_out(self, roi_in: ROI, params: RawPrepareParams) -> ROI:
        """
        根据输入 ROI 计算输出 ROI
        
        Args:
            roi_in: 输入感兴趣区域
            params: rawprepare 参数
            
        Returns:
            输出感兴趣区域
        """
        # 应用裁剪
        total_crop_x = params.crop_left + params.crop_right
        total_crop_y = params.crop_top + params.crop_bottom
        
        scale = roi_in.scale
        adjusted_crop_x = int(total_crop_x * scale)
        adjusted_crop_y = int(total_crop_y * scale)
        
        new_width = max(1, roi_in.width - adjusted_crop_x)
        new_height = max(1, roi_in.height - adjusted_crop_y)
        
        return ROI(
            x=roi_in.x + int(params.crop_left * scale),
            y=roi_in.y + int(params.crop_top * scale),
            width=new_width,
            height=new_height,
            scale=roi_in.scale
        )
    
    def _process_multichannel(self, 
                             input_buffer: ImageBuffer,
                             params: RawPrepareParams,
                             roi_in: ROI,
                             roi_out: ROI,
                             crop_x: int,
                             crop_y: int) -> ImageBuffer:
        """处理预下采样的多通道数据，待完善"""
        
        input_data = input_buffer.data
        channels = input_buffer.channels
        output_data = np.zeros((roi_out.height, roi_out.width, channels), dtype=np.float32)
        
        # 计算归一化参数
        sub, div = self._compute_normalization_params(params)
        
        # 处理每个像素和通道
        for j in range(roi_out.height):
            for i in range(roi_out.width):
                pin_y = j + crop_y
                pin_x = i + crop_x
                
                if (pin_y < input_data.shape[0] and pin_x < input_data.shape[1]):
                    for c in range(channels):
                        channel_id = min(c, 3)  # 限制到4个通道
                        pixel_value = input_data[pin_y, pin_x, c] if len(input_data.shape) == 3 else input_data[pin_y, pin_x]
                        output_data[j, i, c] = (pixel_value - sub[channel_id]) / div[channel_id]
        
        output_data = np.clip(output_data, 0.0, 1.0)
        
        return ImageBuffer(
            data=output_data,
            width=roi_out.width,
            height=roi_out.height,
            channels=channels,
            datatype=DataType.FLOAT32,
            colorspace=ColorSpace.RGB if channels > 1 else ColorSpace.RAW,
            filters=input_buffer.filters,
            bayer_pattern=input_buffer.bayer_pattern,
            xtrans=input_buffer.xtrans,
            black_level=0.0,
            white_point=1.0
        )
    
    def _apply_gainmap(self, 
                      input_buffer: ImageBuffer, 
                      roi_out: ROI,
                      crop_x: int,
                      crop_y: int) -> ImageBuffer:
        """应用 GainMap 进行平场校正"""
        
        if not self.gainmaps or len(self.gainmaps) != 4:
            return input_buffer
            
        gainmap = self.gainmaps[0]  # 假设所有 GainMap 具有相同的形状
        
        input_data = input_buffer.data
        output_data = input_data.copy()
        
        # 预计算映射参数，这是DNG规范中的定义，是常数
        map_w = gainmap.map_points_h
        map_h = gainmap.map_points_v
        im_to_rel_x = 1.0 / input_buffer.width # 相对坐标
        im_to_rel_y = 1.0 / input_buffer.height
        rel_to_map_x = 1.0 / gainmap.map_spacing_h # GainMap的网格坐标
        rel_to_map_y = 1.0 / gainmap.map_spacing_v
        
        # 应用 GainMap
        for j in range(roi_out.height):
            # 计算 Y 方向的映射
            y_map = np.clip(((roi_out.y + crop_y + j) * im_to_rel_y - gainmap.map_origin_v) * rel_to_map_y, 0, map_h)
            y_i0 = min(int(y_map), map_h - 1)
            y_i1 = min(y_i0 + 1, map_h - 1)
            y_frac = y_map - y_i0
            
            for i in range(roi_out.width):
                # 计算 Bayer 位置
                bayer_id = self._get_bayer_id_for_gainmap(roi_out, crop_x, crop_y, j, i)
                
                # 计算 X 方向的映射
                x_map = np.clip(((roi_out.x + crop_x + i) * im_to_rel_x - gainmap.map_origin_h) * rel_to_map_x, 0, map_w)
                x_i0 = min(int(x_map), map_w - 1)
                x_i1 = min(x_i0 + 1, map_w - 1)
                x_frac = x_map - x_i0
                
                # 双线性插值获取增益值
                map_data = self.gainmaps[bayer_id].map_gain
                gain_top = (1.0 - x_frac) * map_data[y_i0, x_i0] + x_frac * map_data[y_i0, x_i1]
                gain_bottom = (1.0 - x_frac) * map_data[y_i1, x_i0] + x_frac * map_data[y_i1, x_i1]
                gain = (1.0 - y_frac) * gain_top + y_frac * gain_bottom
                
                # 应用增益
                output_data[j, i] *= gain
        
        input_buffer.data = output_data
        return input_buffer
    
    def _get_bayer_id_for_gainmap(self, roi: ROI, crop_x: int, crop_y: int, row: int, col: int) -> int:
        """为 GainMap 获取 Bayer 位置 ID"""
        abs_row = row + roi.y + crop_y
        abs_col = col + roi.x + crop_x
        return ((abs_row & 1) << 1) + (abs_col & 1)
    
    def set_gainmaps(self, gainmaps: List[GainMap]):
        """设置 GainMap 数据"""
        self.gainmaps = gainmaps
        self.flat_field_type = FlatFieldType.EMBEDDED if gainmaps else FlatFieldType.OFF
    
    def set_flat_field_type(self, flat_field_type: FlatFieldType):
        """设置平场校正类型"""
        self.flat_field_type = flat_field_type
    
    def _adjust_xtrans_filters(self, filters: Optional[np.ndarray], crop_x: int, crop_y: int) -> Optional[np.ndarray]:
        """调整 X-Trans 滤镜模式（Fujifilm 传感器）"""
        if filters is None:
            return None
            
        # X-Trans 是 6x6 模式
        if filters.shape == (6, 6):
            adjusted_filters = np.zeros((6, 6), dtype=filters.dtype)
            for i in range(6):
                for j in range(6):
                    adjusted_filters[j, i] = filters[(j + crop_y) % 6, (i + crop_x) % 6]
            return adjusted_filters
        
        return filters 