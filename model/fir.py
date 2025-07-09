# @Author: Jiahao Huang
# @Date: 2025-07-09
# @Description: Feed In RAW Image


import numpy as np
import rawpy
import cv2
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import sys
import exifread
import yaml
from yaml.representer import Representer
import re
from pathlib import Path

# 自定义矩阵格式表示
class LiteralStr(str):
    pass

def literal_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

yaml.add_representer(LiteralStr, literal_presenter)

# 将numpy数组/列表转换为格式化的字符串矩阵
def format_matrix(matrix, name=None):
    if matrix is None:
        return None
    
    # 转换numpy数组为Python列表
    if hasattr(matrix, 'tolist'):
        matrix = matrix.tolist()
    
    # 格式化矩阵为多行字符串
    rows = []
    if name:
        rows.append(f"{name} = [")
    else:
        rows.append("[")
        
    for row in matrix:
        if isinstance(row, (list, tuple)):
            # 对行内的每个元素进行格式化，使用固定宽度
            formatted_row = "  [" + ", ".join(f"{x:10.6f}" for x in row) + "]"
            rows.append(formatted_row)
        else:
            # 处理一维数组的每个元素
            rows.append(f"  {row:10.6f}" if isinstance(row, float) else f"  {row}")
    rows.append("]")
    
    return LiteralStr('\n'.join(rows))

class FIR:
    """
    Feed In RAW Image
    
    description:
        this is a class for Execute ISP algorithm
    
    step:
        1. get the raw image
        2. preprocess the raw image
        3. get the raw image with metadata or without metadata
        
    usage:
        raw = FIR(raw_img_path, Height=4032, Width=3024)
    """
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.raw_img_path = self.kwargs.get('RAW_img_path', None)
        if not self.raw_img_path:
            raise ValueError("RAW_img_path must be provided")
        self.raw_height = self.kwargs.get('RAW_Height', None)
        self.raw_width = self.kwargs.get('RAW_Width', None)
        # 配置文件路径
        self.isp_config_path = self.kwargs.get('isp_config_path', 'config/isp_config.yaml')
        # 是否更新配置文件
        self.update_config = self.kwargs.get('update_config', True)
    
    def run(self) -> np.ndarray:
        """
        get the raw image
        """
        # 检查文件扩展名，确定是否有元数据
        if self.raw_img_path and '.' in self.raw_img_path:
            extension = self.raw_img_path.split('.')[-1]
            raw_img_dtype = 'Metadata' if extension.lower() in ('dng', 'nef', 'cr2') else 'NoMetadata'
        else:
            raw_img_dtype = 'NoMetadata'
        __dict__ = {
            'Metadata': self.__get_raw_with_metadata,
            'NoMetadata': self.__get_raw_without_metadata
        }
        print('''
        ====================================================================================================
        |                                                                                                  |
        |                                python - ISP is runing                                            |
        |                                                                                                  |
        ====================================================================================================
        ''')
        return __dict__.pop(raw_img_dtype)()
    
    def __get_raw_with_metadata(self) -> np.ndarray:
        """
        get the raw image with metadata, such as .dng, .DNG, .nef, .NEF, .cr2, .CR2
        """
        raw = rawpy.imread(self.raw_img_path)
        raw_img = raw.raw_image_visible.astype(np.uint16)
        
        # 根据 orientation 纠正方向 (sample5 旋转 90° 的问题在此修复)
        raw_img = self._apply_orientation(raw_img, getattr(raw.sizes, 'flip', 0))
        
        # 提取元数据
        metadata = self.__extract_metadata(raw)
        
        # 如果需要，更新ISP配置文件
        if self.update_config:
            self.__update_isp_config(metadata)
        
        del raw
        return raw_img
        
    def _derive_bayer_pattern(self, raw) -> str:
        """Return Bayer pattern string (e.g., 'RGGB') derived from LibRaw raw_pattern and color_desc."""
        desc_bytes = raw.color_desc
        # raw.color_desc may be bytes (b'RGBG') or str
        if isinstance(desc_bytes, bytes):
            desc = desc_bytes.decode('ascii', errors='ignore')
        else:
            desc = str(desc_bytes)
        pattern_indices = raw.raw_pattern  # 2x2 ndarray
        # Flatten in row-major
        pattern_str = ''.join(desc[int(pattern_indices[i, j])] for i in range(2) for j in range(2))
        pattern_str = pattern_str.upper()
        # Map uncommon patterns (e.g. RGBG) to a supported default RGGB if necessary
        if pattern_str not in ['RGGB', 'GRBG', 'BGGR', 'GBRG']:
            print(f"警告: 检测到不支持的拜尔排列 '{pattern_str}'，将按 RGGB 处理")
            pattern_str = 'RGGB'
        return pattern_str

    def __extract_metadata(self, raw) -> Dict:
        """
        提取 RAW 文件的元数据信息并返回字典
        """
        bayer_pattern_str = self._derive_bayer_pattern(raw)

        # 根据 orientation 纠正宽高信息（flip: 4/5/6/7 表示需要旋转 90/270°）
        flip_code = getattr(raw.sizes, 'flip', 0)
        if flip_code in (4, 5, 6, 7):
            vis_height = raw.sizes.width
            vis_width = raw.sizes.height
        else:
            vis_height = raw.sizes.height
            vis_width = raw.sizes.width
        
        # 创建元数据字典
        metadata = {
            "image": {
                "sizes": {
                    "raw_height": raw.sizes.raw_height,
                    "raw_width": raw.sizes.raw_width,
                    "height": vis_height,
                    "width": vis_width
                }
            },
            "isp_params": {
                "black_level": raw.black_level_per_channel,
                "white_level": raw.white_level,
                "bayer_pattern": bayer_pattern_str,  # 使用字符串形式
                "white_balance": raw.camera_whitebalance,
                "color_matrix": raw.color_matrix if hasattr(raw, 'color_matrix') else None,
                "rgb_xyz_matrix": raw.rgb_xyz_matrix if hasattr(raw, 'rgb_xyz_matrix') else None
            }
        }
        
        # # 打印提取的元数据信息
        # print("\n========== 已提取 RAW 文件元数据信息 ==========")
        # print(f"尺寸信息: {raw.sizes}")
        # print(f"黑电平: {raw.black_level_per_channel}")
        # print(f"白电平: {raw.white_level}")
        # print(f"拜尔模式: {raw.raw_pattern}")
        # print(f"颜色描述: {raw.color_desc}")
        # print(f"相机白平衡: {raw.camera_whitebalance}")
        # print("==========================================\n")
        
        return metadata
    
    def __update_isp_config(self, metadata: Dict) -> None:
        """
        更新 ISP 配置文件，保留原始格式
        """
        if not os.path.exists(self.isp_config_path):
            print(f"ISP 配置文件 {self.isp_config_path} 不存在，跳过更新")
            return
        
        try:
            # 读取现有配置文件的所有行
            with open(self.isp_config_path, 'r', encoding='utf-8') as f:
                config_lines = f.readlines()
            
            # 提取需要更新的参数值
            updates = {}
            
            # 图像尺寸
            if 'image' in metadata and 'sizes' in metadata['image']:
                updates['RAW_Height'] = metadata['image']['sizes'].get('height')
                updates['RAW_Width'] = metadata['image']['sizes'].get('width')
            
            # ISP 参数
            if 'isp_params' in metadata:
                # 白电平
                updates['white_level'] = metadata['isp_params'].get('white_level')
                
                # 拜尔模式
                if 'bayer_pattern' in metadata['isp_params']:
                    updates['bayer_pattern'] = metadata['isp_params']['bayer_pattern']
                
                # 黑电平
                if 'black_level' in metadata['isp_params'] and metadata['isp_params']['black_level'] is not None:
                    black_levels = metadata['isp_params']['black_level']
                    if len(black_levels) >= 4:
                        updates['black_level_r'] = float(black_levels[0])
                        updates['black_level_gr'] = float(black_levels[1])
                        updates['black_level_gb'] = float(black_levels[2])
                        updates['black_level_b'] = float(black_levels[3])
                
                # 白平衡增益
                if 'white_balance' in metadata['isp_params'] and metadata['isp_params']['white_balance'] is not None:
                    wb = metadata['isp_params']['white_balance']
                    if len(wb) >= 3:
                        updates['r_gain'] = float(wb[0])
                        updates['b_gain'] = float(wb[2])
                
                # 色彩矩阵
                if 'color_matrix' in metadata['isp_params'] and metadata['isp_params']['color_matrix'] is not None:
                    ccm = metadata['isp_params']['color_matrix']
                    if ccm.shape[0] >= 3 and ccm.shape[1] >= 3:
                        # 提取 3x3 矩阵
                        ccm_matrix = []
                        for i in range(3):
                            ccm_matrix.append([float(ccm[i][0]), float(ccm[i][1]), float(ccm[i][2])])
                        updates['ccm_matrix'] = ccm_matrix
            
            # 更新配置文件中的参数
            new_lines = []
            i = 0
            ccm_section = False  # 标记是否在处理 CCM 矩阵部分
            ccm_indent = ""      # CCM 矩阵的缩进
            
            while i < len(config_lines):
                line = config_lines[i]
                
                # 检查是否进入 CCM 矩阵部分
                if 'ccm_matrix:' in line and 'ccm_matrix' in updates:
                    new_lines.append(line)
                    ccm_section = True
                    # 获取缩进
                    indent_match = re.match(r'^(\s*)', line)
                    if indent_match:
                        ccm_indent = indent_match.group(1) + "    "
                    else:
                        ccm_indent = "    "
                    
                    # 跳过原有的 CCM 矩阵行
                    i += 1
                    while i < len(config_lines) and (config_lines[i].strip().startswith('-') or config_lines[i].strip() == ""):
                        i += 1
                    
                    # 添加新的 CCM 矩阵
                    for row in updates['ccm_matrix']:
                        new_lines.append(f"{ccm_indent}- [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]\n")
                    
                    continue
                
                # 如果不在 CCM 部分，检查普通参数
                if not ccm_section:
                    # 检查是否是参数行
                    param_match = re.match(r'^(\s*)([a-zA-Z0-9_]+):\s*(.*?)(\s*#.*)?$', line)
                    if param_match:
                        indent, param_name, current_value, comment = param_match.groups()
                        comment = comment or ""  # 如果 comment 为 None，则设为空字符串
                        
                        # 如果是需要更新的参数
                        if param_name in updates and updates[param_name] is not None:
                            # 保持原有格式，更新值
                            if param_name in ['RAW_Height', 'RAW_Width']:
                                # 对于可能是 ~ 的参数，直接替换
                                new_lines.append(f"{indent}{param_name}: {updates[param_name]}{comment}\n")
                            elif isinstance(updates[param_name], (int, float)):
                                # 数值型参数，保持原有格式
                                if current_value.strip() == '~':
                                    new_lines.append(f"{indent}{param_name}: {updates[param_name]}{comment}\n")
                                else:
                                    # 尝试保持原有的格式（空格对齐等）
                                    spaces_match = re.match(r'^([0-9.]+)(\s*)$', current_value.strip())
                                    if spaces_match and spaces_match.group(2):
                                        new_lines.append(f"{indent}{param_name}: {updates[param_name]}{spaces_match.group(2)}{comment}\n")
                                    else:
                                        new_lines.append(f"{indent}{param_name}: {updates[param_name]}{comment}\n")
                            else:
                                # 字符串型参数，加引号
                                new_lines.append(f"{indent}{param_name}: '{updates[param_name]}'{comment}\n")
                            i += 1
                            continue
                
                # 保留原行
                new_lines.append(line)
                i += 1
            
            # 写回文件
            with open(self.isp_config_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            print(f"ISP 配置文件已更新: {self.isp_config_path}")
            
        except Exception as e:
            print(f"更新 ISP 配置文件时出错: {e}")

    def __get_raw_without_metadata(self) -> np.ndarray: 
        """
        get the raw image without metadata, such as .raw, .RAW
        """
        print('get the raw image without metadata...')
        
        if not self.raw_height or not self.raw_width:
            raise ValueError("For RAW files without metadata, RAW_Height and RAW_Width must be provided")
            
        # 直接从文件读取二进制数据
        if self.raw_img_path:  # 确保文件路径不为 None
            raw_data = np.fromfile(self.raw_img_path, dtype=np.uint16)
            raw_img = raw_data.reshape(self.raw_height, self.raw_width)
        else:
            raise ValueError("RAW_img_path cannot be None")
        
        # 创建基本元数据
        metadata = {
            "image": {
                "sizes": {
                    "raw_height": self.raw_height,
                    "raw_width": self.raw_width,
                    "height": self.raw_height,
                    "width": self.raw_width
                }
            },
            "isp_params": {
                "black_level": [64, 64, 64, 64],  # 默认值
                "white_level": 1023,              # 默认 10-bit
                "bayer_pattern": [[0, 1], [3, 2]], # 默认 RGGB
                "color_desc": "RGGB"
            }
        }
        
        # 如果需要，更新ISP配置文件
        if self.update_config:
            self.__update_isp_config(metadata)
        
        return raw_img

    @staticmethod
    def _apply_orientation(img: np.ndarray, flip_code: int) -> np.ndarray:
        """Apply LibRaw/EXIF orientation based on sizes.flip (0-7 = EXIF 1-8 minus 1).

        Mapping (after subtracting 1 from EXIF Orientation):
            0 – Normal                   ➜ identity
            1 – Rotate 180°             ➜ rot90(k=2)
            2 – Mirror horizontal       ➜ fliplr
            3 – Mirror vertical         ➜ flipud
            4 – Mirror horizontal + 90° CW (transpose)  ➜ rot90(fliplr(img), k=-1)
            5 – Rotate 90° CW           ➜ rot90(k=-1)
            6 – Mirror horizontal + 90° CCW (transverse)➜ rot90(fliplr(img), k=1)
            7 – Rotate 90° CCW          ➜ rot90(k=1)
        """
        if flip_code == 0:
            return img  # Identity
        elif flip_code == 1:
            return np.rot90(img, 2)  # 180° rotation
        elif flip_code == 2:
            return np.fliplr(img)  # Horizontal mirror
        elif flip_code == 3:
            return np.flipud(img)  # Vertical mirror
        elif flip_code == 4:
            # Mirror horizontal then rotate 90° CW  (transpose)
            return np.rot90(np.fliplr(img), -1)
        elif flip_code == 5:
            return np.fliplr(np.flipud(np.rot90(img, -1)))
        elif flip_code == 6:
            # Mirror horizontal then rotate 90° CCW (transverse)
            return np.rot90(np.fliplr(img), 1)
        elif flip_code == 7:
            return np.rot90(img, 1)  # 90° CCW rotation
        else:
            return img

if __name__ == '__main__':
    raw_img_path = 'test_images/sample4.dng'
    raw_img = FIR(RAW_img_path = raw_img_path).run()

