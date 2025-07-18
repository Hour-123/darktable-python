# Darktable 源代码中白平衡模块分析

## 概述
Darktable 中的白平衡模块主要通过 `src/iop/temperature.c` 文件实现。该模块的核心功能是调整图像的颜色平衡，使其适应不同的光照条件。它通过调整色温（Temperature）和色调（Tint）来实现，或者直接设置红、绿、蓝通道的系数。在 darktable 的管线中，此模块在去马赛克（demosaicing）**之前**运行，直接对原始传感器数据进行操作。**然而，为了简化我们的 Python 实现，我们计划在去马赛克之后应用白平衡。**

### 关键参数
- **色温 (Temperature)**: 以开尔文（K）为单位，范围从 `1901K` 到 `25000K`。用于校正光源的颜色，从暖色（低K值）到冷色（高K值）。
- **色调 (Tint)**: 范围从 `0.135` 到 `2.326`。用于微调绿色和品红色之间的平衡，以补偿光源在普朗克轨迹（黑体辐射轨迹）上的偏移。
- **通道系数 (Channel Coefficients)**: 直接控制红、绿、蓝（以及第四个通道，用于某些非标准 Bayer 传感器）通道的乘数。这些系数通常以绿色通道为基准（归一化为 1.0）。
- **预设 (Preset)**: 提供多种一键式白平衡设置。

### 预设
模块提供了多种预设模式，定义在 `dt_iop_temperature_params_t` 结构中的 `preset` 字段。
- **As Shot (as shot)**: 使用相机在拍摄时记录在 EXIF 元数据中的白平衡设置。这是默认和最常用的设置。
- **Spot (spot white balance)**: 使用颜色选择器工具在图像的特定区域上取样，计算该区域的平均颜色并将其作为中性灰来设置白平衡。
- **User (manual)**: 允许用户手动调整色温、色调或通道系数。
- **Camera Reference (D65)**: 使用标准的 D65 日光光源作为参考白点。这通常用于色彩校准场景。
- **As Shot to Reference (from camera reference)**: 这是一个校正模式，它首先应用“As Shot”设置，然后应用一个额外的变换，将光源校正到 D65 参考。
- **相机特定预设**: 从 `common/wb_presets.h` 文件中加载，包含了大量针对特定相机型号的预设值，如日光、阴天、钨丝灯等。

### 核心算法流程
白平衡调整的核心是将用户设定的色温和色调转换为一组针对相机特定 RGB 传感器的通道乘数。这个过程可以分解为以下几个步骤：

1.  **色温/色调到 CIE XYZ 转换**:
    -   `_temperature_tint_to_XYZ(TempK, tint)` 函数是起点。它首先调用 `_temperature_to_XYZ(TempK)`。
    -   在 `_temperature_to_XYZ` 中，代码根据色温 `TempK` 的值选择不同的物理模型：
        -   **T < 4000K**: 使用 `_spd_blackbody` 函数，该函数基于普朗克定律计算黑体在不同波长下的谱功率密度（SPD），模拟白炽灯等光源。
        -   **T >= 4000K**: 使用 `_spd_daylight` 函数，该函数基于 CIE D 光源标准计算日光在不同波长下的 SPD。
    -   `_spectrum_to_XYZ` 函数接收 SPD 函数作为输入，并使用 `external/cie_colorimetric_tables.c` 中定义的 CIE 1931 标准观察者颜色匹配函数，将 SPD 积分计算出 CIE XYZ 三刺激值。
    -   最后，`_temperature_tint_to_XYZ` 应用 `tint` 值。这是一个近似调整，通过直接缩放 XYZ 中的 `Y` 分量实现：`xyz.Y /= tint`。这在垂直于普朗克轨迹的方向上移动白点，以校正绿色/品红偏移。

2.  **CIE XYZ 到相机 RGB 系数**:
    -   `_xyz2mul(self, xyz, mul)` 函数负责将 CIE XYZ 值转换为相机空间的 RGB 乘数。
    -   此转换需要一个从 XYZ 空间到相机原生 RGB 空间的 `3x3` 转换矩阵。这个矩阵存储在 `self->data->XYZ_to_CAM` 中，并在 `_prepare_matrices` 函数中从图像的元数据或 darktable 的相机数据库中获取。
    -   将 XYZ 值与此矩阵相乘，得到相机空间的 R、G、B 值。
    -   `for_four_channels(c) mul[c] = (coeffs[c] > 0) ? (1.0 / coeffs[c]) : 1.0;` 这行代码计算出最终的乘数，即系数的倒数。
    -   最后，所有乘数都相对于绿色通道进行归一化，即 `mul[c] /= mul[GREEN]`，确保绿色通道的乘数始终为 `1.0`。

3.  **反向转换 (系数到色温/色调)**:
    -   `_mul2temp(self, p, &TempK, &tint)` 执行反向操作。
    -   它首先调用 `_mul2xyz` 将 RGB 乘数转换回 CIE XYZ 值（使用 `CAM_to_XYZ` 逆矩阵）。
    -   然后 `_XYZ_to_temperature` 使用二分搜索算法在色温范围内查找与计算出的 `XYZ.Z / XYZ.X` 比率最匹配的色温 `TempK`。
    -   `tint` 值则通过 `XYZ.Y / XYZ.X` 的比率计算得出。

4.  **图像处理 (darktable 原生流程)**:
    -   `process(self, ...)` 函数是最终应用白平衡校正的地方，它在去马赛克**之前**执行。
    -   它获取计算出的 `d_coeffs`（即红、绿、蓝及备用通道的最终系数）。
    -   函数内通过检查 `piece->src_filter` 判断输入是 Bayer、X-Trans 还是已处理的 RGB 数据。
    -   它直接在CFA（Color Filter Array）数据上操作，为每个像素的 R, G, B（或 G1, G2）分量应用对应的系数：`out[k+c] = in[k+c] * d_coeffs[c];`。
    -   该模块还提供了一个 OpenCL 内核 `kernel_whitebalance` 用于 GPU 加速，其逻辑与 CPU 版本相同。

### GUI 实现要点
- GUI 元素（如色温滑块）通过 `_color_temptint_sliders` 等函数被赋予了颜色渐变，为用户提供了直观的视觉反馈。
- 当用户与滑块或预设交互时，会触发 `gui_changed` 和其他回调函数，这些函数会调用核心算法来重新计算系数，并请求管线（pipeline）刷新。

### Python 复刻计划
要在 Python 中复刻此模块，需要以下步骤：

1.  **实现核心转换**:
    -   从 `external/cie_colorimetric_tables.c` 中提取 CIE 1931 颜色匹配函数数据表。
    -   使用 NumPy 实现 `_spd_blackbody` 和 `_spd_daylight` 函数来计算光谱功率分布。
    -   实现 `_spectrum_to_XYZ`，使用 NumPy 的数值积分功能（如 `np.trapz`）来替代 C 代码中的循环。
    -   实现 `_temperature_tint_to_XYZ` 和 `_xyz2mul` 的逻辑。

2.  **获取相机矩阵**:
    -   最关键的挑战是获取准确的 `XYZ_to_CAM` 转换矩阵。此矩阵对于每个相机型号都是唯一的。
    -   可以尝试从 darktable 的 `wb_presets.h` 和 `cameras.xml` 文件中解析这些矩阵，或者从 Adobe DNG Converter 等工具生成的 DNG 文件元数据中提取。

3.  **应用到图像 (调整后流程)**:
    -   根据我们的简化流程，白平衡将在去马赛克**之后**应用。
    -   使用 `rawpy` 库读取 RAW 文件并直接进行去马赛克，输出线性的 RGB 图像。
    -   将计算出的 R, G, B 系数（绿色通道系数已归一化为 1.0）分别应用到 RGB 图像的三个通道上。
    -   这个方法简化了处理，因为我们无需处理复杂的 Bayer 模式，但其结果可能与 darktable 的原生流程有细微差异。

4.  **构建用户接口 (可选)**:
    -   使用如 `PyQt` 或 `Tkinter` 的库创建一个简单的 GUI，包含色温和色调滑块，以交互方式控制参数。

### 挑战与总结
- **颜色科学的精确性**: 转换的准确性高度依赖于物理常数和数据表的精确度。
- **相机矩阵**: 获取和管理不同相机的色彩矩阵是最大的工程挑战。
- **Tint 的近似处理**: darktable 中 `tint` 的实现是一个简化处理。更精确的模型会沿着垂直于普朗克轨迹的方向进行调整，但这需要更复杂的数学计算。

通过上述分析，我们对 darktable 的白平衡模块有了深入的理解，为 Python 复刻奠定了坚实的基础。

*分析日期: 2025-07-17* 