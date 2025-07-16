# darktable_python 

本项目是一个基于 darktable 的图像处理管线，使用 python 复刻其 iop 模块。它采用模块化方法，通过配置文件将各种图像操作（IOP）模块链接在一起。

## 功能特性

目前已实现的图像处理模块包括：

-   **Bloom (辉光)**: 在高光周围产生柔和的发光效果。
-   **Blurs (模糊)**: 应用不同类型的模糊（高斯、镜头、运动）。
-   **Borders (边框)**: 为图像添加照片式边框。
-   **Color Balance RGB (色彩平衡)**: 基于提亮、伽马和增益调整图像的色彩平衡。
-   **Diffuse (柔光)**: 模拟光线扩散或光晕，常用于制作光环效果。
-   **Dither (抖动)**: 通过增加噪点来减少色彩断层。
-   **Exposure (曝光)**: 调整图像的整体曝光。
-   **Filmic RGB (Filmic 色调映射)**: 一个现代化的色调映射模块，用于将高动态范围场景压缩为适合在标准显示器上观看的图像。
-   **Grain (胶片颗粒)**: 模拟摄影胶片的颗粒感。
-   **Highpass (高反差保留)**: 一种可用于锐化的滤波器。
-   **Invert (反相)**: 反转图像颜色。
-   **Lowlight (低光)**: 增强阴影区域的细节。
-   **Sharpen (锐化)**: 对图像进行锐化。
-   **Soften (柔化)**: 对图像应用柔化效果。
-   **Tone Curve (色调曲线)**: 一个经典的模块，使用曲线调整图像的色调。
-   **Vignette (暗角)**: 添加暗角效果。
-   **Watermark (水印)**: 为图像添加水印。

## 目录结构

```
darktable_python/
├── core/
│   └── pipeline.py         # 主处理管线
├── iop/
│   ├── bloom.py
│   ├── blurs.py
│   ├── borders.py
│   ├── ...                 # 各个图像操作 (IOP) 模块
│   └── watermark.py
├── assets/
│   └── logo.png            # 模块使用的素材 (例如：水印)
├── output/                 # 处理后图像的默认输出目录
├── test_suite/
│   ├── verify_exposure.py  # `exposure` 模块的验证脚本
│   └── test_assets/        # 用于验证的图像素材
├── utils/
│   └── watermark_generator.py # 水印生成工具
├── gui.py                  # 图形用户界面 (GUI)
├── pipeline_config.yaml    # 管线配置文件
├── requirements.txt        # 项目依赖
└── README.md
```

## 运行方式

1.  **安装依赖**

    建议使用虚拟环境。

    ```bash
    pip install -r requirements.txt
    ```

2.  **配置处理管线**

    修改 `pipeline_config.yaml` 文件来定义处理步骤。您可以指定输入文件、要应用的 IOP 模块序列以及它们的参数。

    `pipeline_config.yaml` 示例:
    ```yaml
    input_file: 'test_images/your_image.arw'
    output_file: 'pipeline_output.jpg'
    pipeline:
      - module: exposure
        enabled: true
        params:
          exposure: 0.5
      - module: filmicrgb
        enabled: true
        params:
          contrast: 1.2
          latitude: 0.1
          balance: 0.0
      - module: sharpen
        enabled: true
        params:
          amount: 1.0
          radius: 1.0
    ```

3.  **运行管线**

    执行主处理管线脚本:
    ```bash
    python core/pipeline.py
    ```
    处理完成的图像将保存在 `output/` 目录中。

4.  **运行图形界面 (GUI)**

    除了通过命令行，您还可以使用图形界面来操作处理管线。

    ```bash
    python gui.py
    ```
    这将启动一个交互式窗口，您可以在其中：
    -   动态选择输入图像。
    -   实时调整所有模块的参数。
    -   查看处理日志。
    -   处理完成后直接预览输出图像。

5.  **运行验证脚本**

    项目包含一个简单的验证套件，用于将核心模块的输出与 darktable 生成的基准进行比较，以确保结果的一致性。

    要运行 `exposure` 模块的验证，请执行：
    ```bash
    python test_suite/verify_exposure.py
    ```
    脚本将处理测试图像，并输出一个量化的比较报告 (PSNR 和 SSIM)，以显示 Python 实现与基准之间的匹配程度。

## 测试架构

**目标：** 在 Python 中精确复刻 Darktable 的图像处理模块，实现效果对齐。以 `exposure` 模块为例。

---

#### 1. 最初的尝试：使用 `darktable-cli` (失败)

- **代码分析：** 我们首先分析了 C 源码 `darktable/src/iop/exposure.c`，确定了核心处理公式为 `output = (input - black) * scale`。
- **测试框架：** 建立了一个基于 `darktable-cli` 的测试流程，用它生成基准图像，再与 Python 的输出进行比对。
- **遇到的问题：** 此方法遭遇了持续的失败。我们曾怀疑过：
    - **输入数据不匹配**：以为是“场景相关”与“显示相关”工作流的差异。
    - **输出数据不匹配**：以为是 `darktable-cli` 默认应用了伽马校正。
    - **数据裁剪错误**：修复了 Python 测试脚本中的一个浮点数裁剪 Bug。
- **结论：** 尽管修正了多个问题，但测试始终无法通过。`darktable-cli` 作为一个“黑盒”，其内部复杂的处理流程（如色彩空间转换、伽马校正等）给验证带来了太多不确定性。

---

#### 2. 使用 Darktable GUI 的初始化数据作为 baseline

我们放弃了 `darktable-cli`，并制定了新的、更可靠的验证策略。

- **核心思想：** Darktable GUI 是最终效果的决定者，因此它应该成为我们验证的“唯一真理来源”。

- **新的验证流程：**
    1.  **准备输入图像 (`sample-origin.tif`)**:
        - 在 Darktable GUI 中打开一个 RAW 文件。
        - **关闭所有**图像处理模块（包括曝光、胶片、色调曲线等）。
        - 导出为 **32位浮点线性 TIFF**。这张图片是我们 Python 脚本的纯净输入。
    2.  **准备基准图像 (`sample-baseline.tif`)**:
        - 基于上一步的状态，**仅开启** `exposure` 模块（例如，设置曝光为 `+1.0 EV`）。
        - 再次导出为 **32位浮点线性 TIFF**。这张图片是我们的“正确答案”。
    3.  **运行 Python 验证脚本**:
        - 脚本读取 `sample-origin.tif`。
        - 应用我们自己实现的 Python `exposure` 模块的逻辑得到 `sample-python-output.tif`。
        - 将结果与 `sample-baseline.tif` 进行像素级比较。

---

#### 3. 效果对齐

- **验证成功：** 新流程取得了立竿见影的效果，Python 输出与 GUI 基准图像的 **PSNR 值达到了 102.97 dB，SSIM 接近 1.0000**。
- **项目结论：**
    - 成功验证了 Python `exposure` 模块的正确性。
    - 确立了一套**基于 Darktable GUI、可信赖、可重复**的模块验证标准。
    - 将此方法记录在项目的 `README.md` 中，作为后续开发其他模块的可供参考标准流程。

_update date: 2025-07-15_
