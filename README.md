# RAW图像处理管线 - Python实现

## 🎯 项目概述

本项目是一个用Python实现的完整RAW图像处理管线，灵感来自于darktable，旨在提供专业级的RAW图像处理能力。项目支持多种去马赛克算法、白平衡方法和曝光控制模式，提供了命令行工具和图形用户界面。

## ✨ 主要特性

### 🔧 核心算法
- **多种去马赛克算法**: 双线性插值、优化版PPG(Pattern Pixel Grouping)算法
- **智能白平衡**: 相机白平衡、日光白平衡、灰度世界、白色区域检测、自适应白平衡
- **灵活曝光控制**: 自适应、手动、混合模式
- **完整处理管线**: RAW → 预处理 → 去马赛克 → 白平衡 → 曝光校正


### 📁 文件支持
- **多格式支持**: 通过DCRaw支持数百种RAW格式 (DNG, CR2, NEF, ARW, ORF等)
- **自动参数提取**: 从RAW文件自动提取黑电平、白点、相机白平衡等参数
- **批量处理**: 支持批量处理多个RAW文件

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 命令行使用
```bash
# 基础用法（使用默认参数）
python examples/test_pipeline.py

# 指定算法
python examples/test_pipeline.py --demosaic ppg --white-balance adaptive

# 手动曝光控制
python examples/test_pipeline.py --exposure-mode manual --ev 1.5

# 处理自定义文件
python examples/test_pipeline.py --file path/to/your/raw/file.dng

# 查看完整参数列表
python examples/test_pipeline.py --help
```


## 📊 算法性能对比

| 算法 | 处理时间 | 色彩平衡 | 视觉效果 |
|------|----------|----------|----------|
| 双线性插值 | ~21秒 | 较好 | 一般 |
| PPG算法 | ~20秒 | 优秀 | 显著提升 |

**PPG算法优势**：
- 智能分类不同图像区域（平滑/边缘/纹理）
- 针对性插值策略，保持边缘锐度
- 显著减少色彩偏向，提升整体视觉效果

## 🔄 处理流程

```
RAW文件 → DCRaw转换 → RAW预处理 → 去马赛克 → 白平衡 → 曝光校正 → 最终输出
   ↓          ↓          ↓         ↓       ↓        ↓
12-16bit   float32    单通道    RGB三通道  色彩平衡  亮度调整
```

### 详细步骤说明

1. **RAW文件加载**: 使用DCRaw处理各种RAW格式
2. **RAW预处理**: 黑电平校正、白点归一化、数据类型转换
3. **去马赛克**: 从Bayer阵列重建RGB图像
4. **白平衡**: 色彩平衡校正，消除色温偏差
5. **曝光校正**: 亮度和对比度调整

## 📁 项目结构

```
python_iop/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖清单
├── requirements_build.txt       # 构建依赖
├── dcraw/                       # DCRaw支持文件
│   ├── dcraw                    # DCRaw可执行文件
│   └── dcraw.c                  # DCRaw源代码
├── src/                         # 核心源代码
│   ├── core/                    # 核心框架
│   │   ├── datatypes.py         # 数据类型定义
│   │   └── pipeline.py          # 处理管线
│   ├── iop/                     # 图像处理模块
│   │   ├── rawprepare.py        # RAW预处理
│   │   ├── demosaic.py          # 去马赛克算法
│   │   ├── white_balance.py     # 白平衡算法
│   │   └── exposure.py          # 曝光控制
│   └── io/                      # 输入输出模块
│       └── dcraw_wrapper.py     # DCRaw接口
├── examples/                    # 示例和测试
│   ├── test_pipeline.py         # 命令行处理工具
│   ├── sample/                  # 示例RAW文件
│   │   ├── sample.DNG           # 示例文件1
│   │   ├── sample2.dng          # 示例文件2
│   │   └── sample3.dng          # 示例文件3
│   └── output/                  # 处理结果输出
├── ui/                          # 图形用户界面
│   ├── run_tkinter_app.py       # GUI启动脚本
│   ├── tkinter_app.py           # 主GUI应用
│   ├── image_viewer.py          # 图像查看器
│   └── README_TKINTER.md        # GUI使用说明
└── venv/                        # 虚拟环境
```

## 🎛️ 参数说明

### 去马赛克算法
- **bilinear**: 双线性插值（快速，适合预览）
- **ppg**: Pattern Pixel Grouping（高质量，推荐）
- **vng4**: Variable Number of Gradients（规划中）

### 白平衡方法
- **camera**: 使用相机元数据白平衡
- **daylight**: 日光白平衡（5500K）
- **gray_world**: 灰度世界假设
- **white_patch**: 白色区域检测
- **adaptive**: 自适应算法（推荐）

### 曝光模式
- **adaptive**: 自适应曝光（根据图像内容自动调整）
- **manual**: 手动曝光（指定EV值）
- **hybrid**: 混合模式（结合自适应和手动）

## 💡 使用示例

### 命令行高级用法
```bash
# 高质量处理推荐设置
python examples/test_pipeline.py --demosaic ppg --white-balance adaptive --exposure-mode adaptive

# 批量处理
for file in *.dng; do
    python examples/test_pipeline.py --file "$file" --demosaic ppg --output-dir "processed_$(basename "$file" .dng)"
done

# 参数验证
python examples/test_pipeline.py --ev 10.0  # 自动提示参数错误
```


## 📊 技术亮点

### PPG算法优化
- **向量化处理**: 使用NumPy广播机制，性能提升10倍以上
- **智能分类**: 根据图像内容自动选择最佳插值策略
- **边缘保持**: 避免边缘模糊，保持图像锐度

### 自适应白平衡
- **多算法融合**: 结合灰度世界和白色区域检测
- **鲁棒性增强**: 处理各种光照条件和色温
- **参数自适应**: 根据图像特征自动调整

### 智能曝光控制
- **内容感知**: 分析图像直方图和统计信息
- **动态范围优化**: 最大化利用传感器动态范围
- **过曝保护**: 避免高光溢出

## 🔧 开发信息

### 核心依赖
- **NumPy**: 数值计算和数组操作
- **SciPy**: 科学计算和图像处理
- **Pillow**: 图像I/O和基础操作
- **rawpy**: RAW文件解析
- **Tkinter**: GUI界面框架


### 支持平台
- **Windows**: 完全支持
- **macOS**: 完全支持
- **Linux**: 完全支持



## 📈 未来规划

### 算法扩展
- [ ] 实现VNG4高级去马赛克算法
- [ ] 增加噪声降低模块


### 性能优化
- [ ] GPU加速支持
- [ ] 多线程并行处理
- [ ] 内存优化


---

