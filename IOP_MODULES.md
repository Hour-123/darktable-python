# Darktable 图像操作模块 (IOP) 清单

这是从 `darktable/src/iop/` 源码目录整理的图像操作模块列表。它们是 darktable 图像处理管线的核心。

---

## 1. 曝光与色调 (Exposure & Tone)

这些模块主要处理图像的亮度、对比度和动态范围。

- `basecurve.c`: 模拟相机JPEG效果的基础曲线。
- `exposure.c`: 曝光补偿。✅
- `filmic.c`: (已废弃) 旧版的 Filmic 模块。
- `filmicrgb.c`: **核心模块** - 现代化的 Filmic RGB，用于将高动态范围的场景参考图像映射到显示参考。✅
- `globaltonemap.c`: 全局色调映射。
- `levels.c`: 色阶调整。
- `negadoctor.c`: 用于处理扫描的胶片负片。
- `profile_gamma.c`: Gamma 曲线调整。
- `rgblevels.c`: RGB 色阶。
- `rgbcurve.c`: RGB 曲线。
- `sigmoid.c`: 一个替代 Filmic RGB 的色调映射模块。
- `shadhi.c`: 阴影和高光调整。
- `tonecurve.c`: 色调曲线（can be replaced by filmicrgb）。✅
- `toneequal.c`: 色调均衡器。
- `zonesystem.c`: 区域系统，类似 Ansel Adams 的分区曝光法。

## 2. 色彩 (Color)

这些模块专注于图像的色彩校正和创意性色彩调整。

- `channelmixer.c`: (已废弃) 旧版通道混合器。
- `channelmixerrgb.c`: 新版通道混合器，在RGB色彩空间工作。
- `colorbalance.c`: (已废弃) 旧版色彩平衡。
- `colorbalancergb.c`: **核心模块** - 强大的色彩平衡工具，用于色彩分级。
- `colorchecker.c`: 使用标准色卡进行色彩校正。
- `colorcontrast.c`: 色彩对比度。
- `colorcorrection.c`: 色彩校正。
- `colorequal.c`: 色彩均衡器。
- `colorize.c`: 单色上色。
- `colormapping.c`: 色彩映射/色彩查找表（CLUT）。
- `colorreconstruction.c`: 修复过曝区域的色彩。
- `colorzones.c`: 色彩区域，可选择性地调整不同色彩的饱和度、亮度等。
- `lut3d.c`: 3D LUT (Look-Up Table) 应用。
- `monochrome.c`: 黑白转换。
- `primaries.c`: (已废-弃)调整三原色。
- `splittoning.c`: 分离色调。
- `temperature.c`: 色温和色调调整。
- `velvia.c`: 模拟富士 Velvia 胶片，增加饱和度。
- `vibrance.c`: 自然饱和度。

## 3. 校正 (Correction)

这些模块用于修复图像中的技术性问题。

- `atrous.c`: 基于 a trous 小波变换的工具，用于锐化或降噪。
- `cacorrect.c`: (已废弃) 旧版色差校正。
- `cacorrectrgb.c`: 新版色差校正。
- `defringe.c`: 去除边缘的色边。
- `denoiseprofile.c`: **核心模块** - 基于相机噪点配置文件的专业降噪。
- `hazeremoval.c`: 去雾。
- `highlights.c`: 高光重建。
- `hotpixels.c`: 去除热像素/坏点。
- `lens.cc`: 镜头畸变、暗角校正。
- `lowpass.c`: 低通滤波器，用于模糊。
- `nlmeans.c`: 非局部均值降噪。
- `rawdenoise.c`: 基础的 RAW 降噪。
- `rawoverexposed.c`: RAW 格式过曝指示。
- `retouch.c`: 修复、克隆、仿制图章工具。
- `sharpen.c`: 锐化。
- `spots.c`: 污点去除。

## 4. 效果与滤镜 (Effects & Filters)

用于添加艺术效果。

- `bloom.c`: 泛光效果。__in pro__
- `blurs.c`: 多种模糊效果。
- `borders.c`: 添加边框。✅
- `diffuse.c`: 柔光/扩散效果。
- `dither.c`: 抖动，用于减少色带。
- `grain.c`: 模拟胶片颗粒。✅
- `highpass.c`: 高通滤波器，常用于锐化。
- `lowlight.c`: 低光照效果。
- `soften.c`: 柔化。
- `vignette.c`: 暗角。✅
- `watermark.c`: 添加水印。✅

## 5. 几何 (Geometry)

用于图像的变换和裁剪。

- `ashift.c`: 透视校正。
- `crop.c`: 裁剪。
- `enlargecanvas.c`: 扩展画布。
- `flip.c`: 翻转。
- `liquify.c`: 液化。
- `rotatepixels.c`: 旋转。
- `scalepixels.c`: 缩放。

## 6. 内部与管线 (Internal & Pipeline)

这些是 darktable 处理流程中的基础或辅助模块，通常用户不直接操作。

- `colorin.c`: 输入色彩配置文件转换。
- `colorout.c`: 输出色彩配置文件转换。
- `demosaic.c`: **核心模块** - 去马赛克，将 RAW 数据转换为彩色图像。
- `finalscale.c`: 管线末端的最终缩放。
- `gamma.c`: Gamma 调整。
- `invert.c`: 反相。✅
- `rawprepare.c`: RAW 数据准备。
- `useless.c`: (调试或模板) 

_Author: Jiahao Huang_