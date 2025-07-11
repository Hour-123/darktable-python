# **项目功能进度报告 (v2)**

# python_IOP_Hour
---

## **核心功能**

当前项目实现了一个基于场景参考（Scene-Referred）工作流的RAW图像处理管线。其核心是 **`filmicrgb` 色调映射模块**，取代了传统的色调曲线（Tone Curve）方法。

### **1. 处理管线 (Pipeline)**
*   **配置文件驱动**: 处理流程由 `pipeline_config.yaml` 定义，按顺序执行指定的图像操作 (IOP) 模块。
*   **当前默认流程**:
    1.  `exposure`: 对输入的线性RAW数据进行曝光补偿，为 `filmicrgb` 准备合适的亮度范围。
    2.  `filmicrgb`: 作为核心模块，执行从场景参考到显示参考（Scene to Display）的色调映射，将高动态范围的图像数据压缩以适应标准显示器。

### **2. 图像操作模块 (IOPs)**
*   **`filmicrgb.py`**:
    *   实现了基于样条曲线（Cubic Spline）的复杂色调映射算法，这是现代数字暗房软件（如darktable）的核心技术。
    *   通过参数（白点、黑点、对比度、纬度）精确控制动态范围压缩和最终图像的"影调感"。
    *   支持色彩保持模式 (`luminance`, `max_rgb`)，在剧烈的亮度调整下尽可能维持色彩的自然观感。
*   **`exposure.py`**:
    *   提供基础的曝光补偿功能（可调整 EV 值）。
*   **`tonecurve.py`**:
    *   实现了传统的基于色调曲线的调整方法（目前在默认管线中被 `filmicrgb` 取代）。

## **总结与下一步**

### **当前状态**
项目已经超越了基础RAW处理的范畴，搭建起了以高级色调映射算法 `filmicrgb` 为中心处理框架。这表明项目的重点在于复现和探索现代化的、更符合人眼视觉感知的图像处理技术。

### **下一步计划**
*   **完善管线**: 将去马赛克、白平衡、降噪等模块重新整合到新的 `filmicrgb` 工作流中。
*   **算法优化**: 对现有模块（特别是 `filmicrgb` 中的样条曲线计算）进行性能分析和优化。
*   **扩展模块**: 增加色彩平衡（Color Balance）、局部对比度（Local Contrast）等更多高级图像处理模块。 

# python-ISP
---

本仓库实现了一条**简化的 RAW→sRGB 图像信号处理（ISP）流水线**，可直接运行并在 `outputs/` 下生成各阶段结果及总览图。

## 目录结构
```
python_ISP/
├── run.py                 # 演示入口，串起整条 ISP 并保存结果
├── config/
│   └── isp_config.yaml    # 传感器 / ISP 参数配置，会在运行时被 FIR 更新
├── model/                 # 各功能模块（FIR、DPC、BLC…）
│   ├── fir.py             # Feed-In RAW：读取 RAW、提取元数据、方向校正
│   ├── dpc.py             # Dead-Pixel Correction
│   ├── blc.py             # Black-Level Correction
│   ├── awb.py             # Auto-White-Balance
│   ├── cfa.py             # Demosaic（Bayer → RGB）
│   ├── ccm.py             # Color-Correction-Matrix
│   └── gmc.py             # Gamma Mapping / Curve（16-bit → 8-bit）
├── test_images/           # 若干示例 RAW（*.dng）
└── outputs/               # 运行后自动生成各阶段 PNG
```

> 说明：本 ISP 模块为主项目分支，不使用 `src/` 目录，运行、调试均不依赖外部源码树。

## 运行方式
```bash
# 1. 建议先创建虚拟环境并安装依赖
pip install -r requirements.txt  # （若已准备好 rawpy、opencv-python、numpy、PyYAML 等可跳过）

# 2. 直接运行
python run.py
```
运行后将在 `outputs/` 下看到：
- `01_raw.png` … `07_gmc.png`：每个处理模块的 8-bit 显示结果
- `00_summary.png`：自动拼接的总览图；若本地装有 matplotlib 会弹窗显示

若需切换测试图，可修改 `config/isp_config.yaml` 中的 `RAW_img_path`，或直接在 shell 中：
```bash
sed -i '' 's#RAW_img_path: .*#RAW_img_path: "test_images/sample5.dng"#' config/isp_config.yaml
python run.py
```

## ISP 处理流水
下表对应 `run.py > get_module_output()` 中的顺序：
| 步骤 | 模块 | 主要功能 |
| ---- | ---- | -------- |
| 1 | **FIR** | 读取 RAW → `np.uint16`，根据 LibRaw 元数据修正旋转 / 镜像；同步更新 `isp_config.yaml`（尺寸、白/黑电平、拜尔模式等） |
| 2 | **DPC** | 死像素检测（阈值+邻域平均）并替换 |
| 3 | **BLC** | 四通道独立减去黑电平|
| 4 | **AWB** | 使用相机白平衡或灰世界增益校正 R/B 通道 |
| 5 | **CFA** | OpenCV Bayer demosaic，将单通道 RAW 还原为 3 通道 RGB；也可选择手写的 demosaic 算法|
| 6 | **CCM** | 3×3 颜色矩阵乘法，粗略映射至 sRGB 空间 |
| 7 | **GMC** | 16-bit 线性 → 8-bit Gamma/Curve，便于显示与存储 |

---
如需进一步实验/扩展，可在 `model/` 中新增模块后，在 `run.py` 对应位置调用即可。

_Author: Jiahao Huang_
_Date: 2025/07/09_