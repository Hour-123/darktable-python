"""
核心数据类型定义
对应 darktable 的关键数据结构
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Any, Dict
from enum import Enum
import numpy as np


class DataType(Enum):
    """数据类型枚举，对应 dt_iop_buffer_type_t"""
    UNKNOWN = "unknown"
    UINT16 = "uint16"
    FLOAT32 = "float32"


class ColorSpace(Enum):
    """色彩空间枚举"""
    RAW = "raw"
    RGB = "rgb"
    LAB = "lab"
    XYZ = "xyz"
    SRGB = "srgb"


class BayerPattern(Enum):
    """Bayer 模式枚举"""
    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"
    XTRANS = "XTRANS"
    MONOCHROME = "MONOCHROME"


@dataclass
class ROI:
    """
    感兴趣区域
    对应 darktable 的 dt_iop_roi_t
    """
    x: int
    y: int
    width: int
    height: int
    scale: float = 1.0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")
        if self.scale <= 0:
            raise ValueError("ROI scale must be positive")


@dataclass
class ImageBuffer:
    """
    图像缓冲区
    对应 darktable 的 dt_iop_buffer_dsc_t
    """
    data: np.ndarray
    width: int
    height: int
    channels: int
    datatype: DataType
    colorspace: ColorSpace = ColorSpace.RAW
    
    # RAW 相关属性
    filters: Optional[int] = None  # Bayer 模式掩码
    bayer_pattern: Optional[BayerPattern] = None
    xtrans: Optional[np.ndarray] = None  # X-Trans 模式 (6x6)
    
    # 色彩处理相关
    processed_maximum: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0]))
    black_level: float = 0.0
    white_point: float = 1.0

    def __post_init__(self):
        if self.data.shape[:2] != (self.height, self.width):
            raise ValueError(f"Data shape {self.data.shape} doesn't match dimensions {(self.height, self.width)}")
        
        if len(self.data.shape) == 2:
            actual_channels = 1
        elif len(self.data.shape) == 3:
            actual_channels = self.data.shape[2]
        else:
            raise ValueError(f"Invalid data shape: {self.data.shape}")
            
        if actual_channels != self.channels:
            raise ValueError(f"Channels mismatch: expected {self.channels}, got {actual_channels}")

    @property
    def size(self) -> Tuple[int, int]:
        """图像尺寸 (width, height)"""
        return (self.width, self.height)

    @property
    def is_raw(self) -> bool:
        """是否为 RAW 格式"""
        return self.colorspace == ColorSpace.RAW and self.channels == 1

    @property
    def is_color(self) -> bool:
        """是否为彩色图像"""
        return self.channels >= 3

    def to_float32(self) -> 'ImageBuffer':
        """转换为 float32 格式"""
        if self.datatype == DataType.FLOAT32:
            return self
        
        new_data = self.data.astype(np.float32)
        if self.datatype == DataType.UINT16:
            # 归一化到 [0, 1] 范围
            new_data = (new_data - self.black_level) / (self.white_point - self.black_level)
            new_data = np.clip(new_data, 0.0, 1.0)
        
        return ImageBuffer(
            data=new_data,
            width=self.width,
            height=self.height,
            channels=self.channels,
            datatype=DataType.FLOAT32,
            colorspace=self.colorspace,
            filters=self.filters,
            bayer_pattern=self.bayer_pattern,
            xtrans=self.xtrans,
            processed_maximum=self.processed_maximum.copy(),
            black_level=0.0,
            white_point=1.0
        )


@dataclass
class IOPParams:
    """IOP 模块参数基类"""
    enabled: bool = True
    module_name: str = ""
    priority: int = 0

    def copy(self) -> 'IOPParams':
        """创建参数副本"""
        return type(self)(**self.__dict__)


@dataclass
class RawPrepareParams(IOPParams):
    """rawprepare 模块参数"""
    # 裁剪参数
    crop_left: int = 0
    crop_top: int = 0
    crop_right: int = 0
    crop_bottom: int = 0
    
    # 黑白电平
    black_level_separate: Tuple[int, int, int, int] = (0, 0, 0, 0)
    white_point: int = 65535
    
    # 增益图
    apply_gain_map: bool = False


@dataclass
class DemosaicParams(IOPParams):
    """demosaic 模块参数"""
    method: str = "rcd"  # ppg, amaze, vng4, rcd, lmmse
    green_equilibration: str = "disabled"  # disabled, local, full, both
    color_smoothing: int = 0  # 0-5
    median_threshold: float = 0.0
    dual_threshold: float = 0.2


@dataclass
class ExposureParams(IOPParams):
    """exposure 模块参数"""
    exposure: float = 0.0  # EV
    black: float = 0.0
    highlights: float = 0.0


@dataclass
class ProcessingMetadata:
    """处理元数据"""
    # 原始图像信息
    camera_make: str = ""
    camera_model: str = ""
    iso: int = 100
    exposure_time: float = 1.0
    aperture: float = 1.0
    focal_length: float = 50.0
    
    # 色彩信息
    color_matrix: Optional[np.ndarray] = None
    white_balance: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # 处理历史
    processing_history: list = field(default_factory=list)

    def add_processing_step(self, module_name: str, params: IOPParams):
        """添加处理步骤到历史"""
        self.processing_history.append({
            'module': module_name,
            'params': params.copy(),
            'timestamp': np.datetime64('now')
        })


class PipelineError(Exception):
    """管线处理异常"""
    pass


class InvalidDataError(PipelineError):
    """无效数据异常"""
    pass


class UnsupportedFormatError(PipelineError):
    """不支持的格式异常"""
    pass 