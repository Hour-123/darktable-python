# Darktable IOP 流程 Python 复现指南

## 项目概述

本项目旨在用 Python 复现 darktable 的图像处理操作（IOP）流程，特别关注从 RAW 文件处理到最终输出的完整管线。

## 核心数据流

基于对 darktable 源码的分析，主要的数据处理流程如下：

```
RAW 文件 (uint12/14) → rawprepare → demosaic → 其他 IOP 模块 → 最终输出
     ↓                   ↓          ↓              ↓
  uint16 数组        float32 单通道  float32 多通道   sRGB/其他格式
```

## 一些技术要点

### 1. 数据类型转换
- **输入**: RAW 文件通常是 12-14 bit，读取后转为 `uint16`
- **rawprepare 输出**: `float32` 单通道，数值范围 [0.0, 1.0]
- **demosaic 输出**: `float32` 3/4 通道 RGB
- **后续处理**: 均在 `float32` 线性色域内进行

### 2. 关键处理步骤

#### rawprepare 模块

#### demosaic 模块

#### ...

## 项目结构

```
python_iop/
├── README.md                  # 本文档
├── requirements.txt           # Python 依赖
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── datatypes.py       # 数据类型定义
│   │   ├── pipeline.py        # 主要的处理管线
│   │   └── utils.py           # 工具函数
│   ├── iop/
│   │   ├── __init__.py
│   │   ├── rawprepare.py      # RAW 预处理
│   │   ├── demosaic.py        # 去马赛克
│   │   ├── exposure.py        # 曝光调整
│   │   ├── colorin.py         # 色彩输入配置
│   │   ├── colorout.py        # 色彩输出配置
│   │   └── ...                # 其他 IOP 模块
│   └── io/
│       ├── __init__.py
│       ├── raw_reader.py      # RAW 文件读取
│       └── image_writer.py    # 图像输出
├── tests/                     # 测试每一个模块的运行情况
│   ├── __init__.py
│   ├── test_rawprepare.py
│   ├── test_demosaic.py
│   └── ...
├── examples/
│   └── basic_pipeline.py      # 基础使用示例
└── data/
    ├── test_images/           # 测试图像
    └── profiles/              # ICC 配置文件
```

## 实现步骤

### 第一阶段：基础框架
1. **设置项目结构**：创建上述目录结构
2. **安装依赖**：
   ```bash
   pip install numpy rawpy opencv-python pillow colorspacious
   ```
3. **实现基础数据类型**：完成 `datatypes.py`
4. **实现 RAW 文件读取**：使用 `rawpy` 库

### 第二阶段：核心 IOP 模块
1. **rawprepare 模块**：
`Ref: /iop/rawprepare.c`
   - 黑电平校正
   - 白点归一化
   - 裁剪功能
   
2. **demosaic 模块**：
`Ref: /iop/demosaic.c`
   - 针对不同的传感器 CFA 实现不同的适配重建算法（双线性、VNG、AMaZE 等），生成高质量多通道 RGB
   
3. **基础校正模块**：
    ```
    Ref: 
    src/iop/temperature.c         # 白平衡调整
    src/iop/exposure.c           # 曝光/亮度调整  
    src/iop/highlights.c         # 高光恢复
    src/iop/basecurve.c         # 基础色调曲线
    src/iop/lens.c              # 镜头校正(畸变/暗角)
    ```

4. **更多 IOP 模块**：噪点降低、锐化、色调映射等
    ```
    Ref: 
    src/iop/colorcorrection.c    # 色彩校正
    src/iop/colorzones.c        # 色彩区域调整
    src/iop/tonecurve.c         # 色调曲线
    src/iop/shadows_highlights.c # 阴影/高光
    src/iop/sharpen.c           # 锐化
    src/iop/noise_reduction.c   # 降噪
    src/iop/vignette.c          # 暗角效果
    ```

## 测试和验证

### 1. 单元测试
为每个 IOP 模块编写单元测试，确保功能正确性。

### 2. 对比测试
将 Python 实现的结果与 darktable 的输出进行对比。
确保性能可接受，与源码功能对齐。


## 参考资源

1. **darktable 源码**：
   - `src/iop/rawprepare.c` - RAW 预处理
   - `src/iop/demosaic.c` - 去马赛克算法
   - `src/develop/format.c` - 数据格式处理
   ...

2. **关键数据结构**：
   - `dt_iop_buffer_dsc_t` - 图像缓冲区描述
   - `dt_dev_pixelpipe_iop_t` - IOP 处理单元
   - `dt_iop_roi_t` - 感兴趣区域
   ...

3. **算法参考**：
   ...

## 注意事项（持续补充）

1. **内存管理**：处理大图像时注意内存使用
2. **数值精度**：保持 float32 精度，避免精度损失
3. **色彩空间**：正确处理线性色彩空间和 sRGB 转换
---
*本文档基于 darktable 5.3.0+24 版本的源码分析编写。*