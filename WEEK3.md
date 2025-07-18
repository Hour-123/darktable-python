# 阶段性进度汇报  
_更新时间：2025-07-18_
---
## 1. 处理管线（Pipeline）与交互框架
### 1.1 引言与设计目标
在 **darktable** 的完整链路中，“处理管线”承担着“装配线”角色：  
它接收未经烘焙的 RAW 数据，按照既定工序顺序调用若干图像操作模块（IOP），最后输出可直接观赏或进一步分发的图片。
本项目的管线设计坚持三条原则：
1. **单一真源** — 所有参数都写在同一个 YAML 文件，命令行与 GUI 共同读取。  
2. **即插即用** — 新增 IOP 无需修改核心脚本，只要放进 `iop/` 目录并遵循命名约定。  
3. **双入口一致** — CLI 与 GUI 调用同一套底层逻辑；用户既可批量脚本，也可所见即所得。
---
### 1.2 系统架构概述
1. **外层控制**  
   * `gui.py`：负责人机交互，将界面变动实时写回 YAML，并以子进程启动同一管线脚本。  
   * 命令行：直接执行 `python core/pipeline.py`，读取同一 YAML。
2. **核心处理**  
   * `core/pipeline.py`：读取配置 → RAW 解码 → 逐模块调用 → 输出。  
   * RAW 解码：调用 `rawpy`，并保持线性（γ=1.0）、关闭自动亮度；确保后续 IOP 获得“干净”输入。
3. **模块层**  
   * 每个 IOP 为独立 Python 类，文件名 `snake_case`、类名 `CamelCase`。  
   * 动态加载机制通过 `importlib` 完成，不依赖硬编码的 if-else。
---
### 1.3配置驱动的流水线
#### 1.3.1 YAML 片段示例
```yaml
input_file : test_images/scene.arw
output_file: demo.jpg
pipeline:
  - module: exposure
    enabled: true
    params:
      exposure_ev : 0.5
  - module: filmicrgb
    enabled: true
    params:
      contrast : 1.2
      latitude : 0.1
  - module: sharpen
    enabled: false      # 可随时禁用
```
要点说明：
* **顺序即执行顺序** — 文件中的排列决定加工次序。  
* **enabled 开关** — 便于做 A/B Test，不必删改条目即可跳过。  
* **params 字典** — 与 IOP 构造函数形参一一对应，可被 GUI 解析为滑块/输入框。
#### 1.3.2 核心函数解读
```python
# 1. 反射加载类
module = importlib.import_module(f"iop.{name}")
cls    = getattr(module, "".join(w.capitalize() for w in name.split('_')))
# 2. 执行
instance   = cls(**params)
image_data = instance.process(image_data)
```
只要新 IOP 文件名为 `local_contrast.py`，且类名 `LocalContrast`，管线便可无感识别。

---
### 1.4 运行时流程
1. **读取配置** —— `yaml.safe_load` 获取字典结构。  
2. **RAW 解码** ——  
   ```python
   raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
   ```  
   输出 16-bit 线性 RGB，范围 0-65535，随后归一化为 0-1 浮点。  
3. **循环调用 IOP** —— 逐条记录进度与耗时；如遇异常即刻中止并打印堆栈。  
4. **γ 校正 & 存盘** —— `x^(1/2.2)` 后量化 `uint8`，写入 `output/`.  
---
### 1.5 图形界面亮点
#### 1.5.1 组件自动生成  
GUI 通过遍历 YAML，自动为每个模块绘制：
* 复选框：控制 enabled  
* 数值滑块 + 输入框：同步双向绑定到 `DoubleVar`  
* 文本输入：处理字符串参数（如水印路径）
因此，添加新 IOP 后无需修改 UI 代码。
#### 1.5.2 运行机制  
点击「Run Pipeline」时，GUI 会：
1. 调 `save_config()` 把当前 UI 状态写回 YAML；  
2. 以子进程启动 `core/pipeline.py`；  
3. 实时抓取 stdout/stderr 显示在日志窗；  
4. 处理成功后调用 `display_image()` 读取最新 JPEG 并自适应缩放预览。
#### 1.5.3 体验细节  
* **拖动分隔条**  调整控件区宽度；  
* **Trackpad 支持**  绑定 `<MouseWheel>`，保证 macOS 上平滑滚动；  
* **文件对话框**  允许拖选任意 RAW/JPEG/TIFF 文件并自动写入相对路径。
---
### 1.6 性能与可维护性评估
| 维度         | 现状                                         | 后续改进           |
|--------------|----------------------------------------------|--------------------|
| CPU 性能     | 纯 NumPy，单线程；24MP <10 s                | Numba / OpenCL 加速|
| 内存占用     | 浮点全图常驻，约 350 MB/24MP                 | 分块处理 / fp16    |
| 错误诊断     | 全链条异常捕获并打印堆栈                    | 日志文件 + 级别    |
| 可扩展性     | 反射发现 IOP，GUI 自动渲染                  | 插件注册表（热加载）|
---
_至此，处理管线与交互框架的阶段性目标已全部达成，为后续 IOP 扩展及颜色科学研究奠定了统一、可靠、易用的基座。_

---
## 2. 常规图像处理功能的 Python 复刻与一致性验证  
### 2.1 目标与范围  
darktable 是一款深受摄影师喜爱的开源数码暗房软件。它之所以灵活，原因在于内部包含二十多种“IOP（Image Operation）”模块，例如 _曝光补偿、锐化、暗角_。  
本模块的目标就是——
1. **功能复刻**：用 Python 1:1 还原这些 IOP 的视觉效果；  
2. **统一调用**：把所有模块封装成“乐高积木”，可由外层流水线随意组合；  
3. **结果对齐**：输出必须与 darktable 原版在数学意义上高度一致（PSNR、SSIM 等指标几乎满分）。  
当前版本已覆盖的 IOP 一览（17 项）  
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
---
### 2.2 实现概览  
#### 2.2.1 模块封装思想  
* **单一职责**：每个 `.py` 文件只实现一项图像操作，类名 CamelCase 与文件名 snake_case 对应。  
* **统一接口**  
  ```python
  class Exposure:
      def __init__(self, exposure_ev: float = 0.0, black_level: float = 0.0): …
      def process(self, rgb_linear: np.ndarray) -> np.ndarray: …
  ```  
  这样流水线可以用反射（`importlib`）动态加载，不需要写 `if…elif`。  
#### 2.2.2 目录与调用链  
```
darktable_python/
├─ core/pipeline.py      # 流水线主脚本
├─ iop/                  # 17 个 IOP 均位于此
│   ├─ exposure.py
│   ├─ filmicrgb.py
│   └─ …
├─ utils/                # 公共小工具
├─ test_suite/           # 每个 IOP 对应一个验证脚本
└─ gui.py                # 图形界面
```
`core/pipeline.py` 读取 `pipeline_config.yaml`，逐步执行：  
1. 解码 RAW（`rawpy`）→ 线性 RGB；  
2. 依序调用 IOP，对 NumPy 数组就地加工；  
3. 最后执行 γ 校正并存盘。  
---
### 2.3 关键代码片段解读  
1. **模块动态加载**  
   ```python
   def load_iop_module(name):
       module = importlib.import_module(f"iop.{name}")
       cls = getattr(module, "".join(w.capitalize() for w in name.split('_')))
       return cls
   ```
   只要保证文件名与类名规则一致，新增模块零配置接入。  
2. **GUI 与 YAML**  
   图形界面基于 `customtkinter`，所有滑块/开关直接回写 `pipeline_config.yaml`，再调用同一份流水线脚本。这样命令行与 GUI 共享同一真源，避免“双份逻辑”。  
---
### 2.4 一致性验证体系  
#### 2.4.1 为什么不用 `darktable-cli`？  
初版测试用 CLI 生成基准，发现暗中触发了 gamut 压缩、Gamma 等隐含步骤，导致对比失败。于是改用 **darktable GUI 导出 32-bit 线性 TIFF**，确保“所见即所得”。  
#### 2.4.2 三步走流程（以 Exposure 为例）  
| 步骤 | 动作 | 目的 |
|------|------|------|
| ① 生成 *origin* | GUI 关闭全部模块 → 导出 | 得到“干净输入” |
| ② 生成 *baseline* | GUI 仅开目标模块（+1 EV）| 得到“答案” |
| ③ Python 复刻 | `verify_exposure.py` 读 *origin* → 调用 `Exposure.process()` → 写 *python-output*，与 *baseline* 做 PSNR/SSIM | 量化差异 |
### 2.4.3 结果  
```
PSNR : 102.97 dB
SSIM : 1.0000
Verdict : SUCCESS
```
> 对 32-bit 浮点图而言，PSNR > 60 dB 就已肉眼无差；百余 dB 基本可视为比特级一致。  
---
### 2.5 GUI 模块亮点  
* **所见即所得**：拖动滑块实时刷新预览；  
* **混合输入控件**：每个数值参数既有滑条也可键入精确值；  
* **可重入**：点击「Run Pipeline」会在后台启动同一 Python 进程，GUI 流水日志实时滚动。  
---
### 2.6 下一步计划  
1. 将 **验证脚本** 扩展到其他 IOP；  
2. 在 `test_suite` 建立 **自动回归**：GitHub Actions 上跑多机型 RAW + 多参数网格；  
3. GUI 增加 **对比视图**（原图 / 处理后滑动条对比），提升用户体验。  
---
##S## 附：Exposure 验证脚本核心逻辑（节选）
```python
origin   = imageio.imread("sample-origin.tif")
exposed  = Exposure(exposure_ev=1.0).process(origin)
baseline = imageio.imread("sample-baseline.tif")

psnr_val = psnr(baseline, exposed, data_range=baseline.ptp())
ssim_val = ssim(baseline, exposed, data_range=baseline.ptp(), channel_axis=-1)
```
_脚本中对 PSNR 和 SSIM 两个指标均使用 `skimage.metrics` 官方实现，避免自写公式带来误差。_

---
## 3 白平衡模块——研究与复刻路线报告  
### 3.1 背景与目标  
在整条 RAW → sRGB 流水线中，白平衡（White Balance, WB）承担着重要的作用。  
Darktable 的 C 实现（`src/iop/temperature.c`）以严谨的物理光谱模型著称，但其运行位置在去马赛克之前且与相机专用矩阵深度耦合，直接移植难度较高。

本项目对白平衡的诉求分三层：  
1. **科学准确** - 遵循物理光谱→CIE XYZ→相机 RGB 的完整链路；  
2. **工程可落地** - 模块化，易插入现有 Python 流水线；  
3. **用户友好** - 提供色温 / 色调滑块与取色器，摄影师无需关心底层矩阵。  
---
## 3.2 Darktable 原理回顾（C语言 侧）  
#### 3.2.1 处理时机  
• Darktable 在 Bayer → RGB 解码之前直接对 CFA 数据逐像素乘以通道系数；  
• CPU 与 OpenCL 路径共存，保证桌面与 GPU 结果一致。  
#### 3.2.2 算法链条  
```
色温K / Tint
      │
      ▼                      (物理模型)
┌──────────────┐  SPD_BlackBody / SPD_Daylight
│  生成光谱SPD │ ───────────────► λ ↦ P(λ)
└──────────────┘
      │  积分                  (CIE 1931 CMF)
      ▼
┌──────────────┐  _spectrum_to_XYZ
│   得到 XYZ   │ ──────────────► (X,Y,Z)
└──────────────┘
      │ Tint 校正 (Y /= tint)【待考证】
      ▼
┌──────────────┐  3×3  XYZ→CAM【待考证】
│  投射到相机  │ ──────────────► (R,G,B)
└──────────────┘
      │  取倒数 & 以 G 归一化
      ▼
通道增益 (mul_R, mul_G=1, mul_B)
```

• 色温分界：T<4000 K 走黑体辐射；T≥4000 K 走 CIE D 系列日光模型  
• Tint 采用经验缩放 `Y /= tint`，相当于在 uv 色度图垂直于普朗克轨迹方向做一次近似位移  
• 相机矩阵 `XYZ_to_CAM` 从元数据或预设库中获取，不同机型互不通用  

---
### 3.3 关键参数  
- **色温 (Temperature)**: 以开尔文（K）为单位，范围从 `1901K` 到 `25000K`。用于校正光源的颜色，从暖色（低K值）到冷色（高K值）。
- **色调 (Tint)**: 范围从 `0.135` 到 `2.326`。用于微调绿色和品红色之间的平衡，以补偿光源在普朗克轨迹（黑体辐射轨迹）上的偏移。
- **通道系数 (Channel Coefficients)**: 直接控制红、绿、蓝（以及第四个通道，用于某些非标准 Bayer 传感器）通道的乘数。这些系数通常以绿色通道为基准（归一化为 1.0）。
- **预设 (Preset)**: 针对自动监测到的不同相机模式，提供查找表进行参数查找。

---
### 3.4 与整体管线的衔接

1. **前置于 Exposure** — 白平衡后图像动态范围更统一，便于后续曝光补偿。  
2. **后接 FilmicRGB** — 保证进入色调映射时白点已归于 D65，可减少高光色漂。  
3. **ISP 分支兼容** — 若走 `python-ISP`，WB 保持在 Bayer 解码前；主管线则沿计划放在去马赛克后，两者共存不冲突。  

---