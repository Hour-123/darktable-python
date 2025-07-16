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

_update date: 2025-07-15_
