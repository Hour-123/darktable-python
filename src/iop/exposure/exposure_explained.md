# `exposure.py` 模块说明

> 复刻了 **Darktable** “曝光（exposure）”模块的核心算法（不含 GUI）。支持手动曝光、序列去闪（deflicker）、Exif 曝光补偿以及 ROI 基准测光。依赖极少，可直接作为命令行工具或被其他脚本导入。

---

## 目录
1. 快速概览  
2. 依赖与安装  
3. 数据流与工作流程  
4. 代码结构拆解  
   1. 色彩辅助函数（sRGB → Lab）  
   2. 参数数据类 `ExposureParams`  
   3. 曝光核心 `apply_exposure`  
   4. 序列去闪算法 `_compute_deflicker_correction`  
   5. 区域测光 / 曝光映射  
   6. I/O 与 EXIF 工具  
   7. 命令行接口 `main()`  
5. 示例  

---

## 1 快速概览

```
out = (in - black) * 2 ** EV
```

脚本围绕上式展开：

* **手动模式**：用户指定 EV（曝光值）与黑电平。  
* **去闪模式**：自动分析序列帧的直方图，再调整 EV 达到给定“目标亮度级”。  
* **可选**：减去相机的 `ExposureBiasValue`，把机内补偿和后期补偿合并。  
* **区域映射**：选定 ROI，在手动模式下一键把该区域的 Lab L 调整到目标值。  

最终输出 0–1 浮点数组，保存为 8bit PNG 方便预览。

---

## 2 依赖与安装

```bash
pip install pillow numpy
```

仅依赖：

* Pillow：读取 / 写入常见图像格式、解析 EXIF  
* NumPy：向量化数值运算、直方图与颜色空间变换  

---

## 3 数据流与工作流程

1. 读取文件 → `PIL.Image` → 转 `float32[0-1]` NumPy 数组  
2. 读取 EXIF 曝光补偿（若有）  
3. 依据模式计算 **EV 修正值**  
   * 手动：直接使用 `params.exposure`  
   * 去闪：调用 `_compute_deflicker_correction()`  
   * ROI 映射：首帧调用 `correct_exposure_for_roi()` 追加 EV  
4. 调用 `apply_exposure()` 执行线性缩放 + 黑电平校正  
5. 截断到 `[0,1]`，写回磁盘（默认 `out/adj_XXXX.png`）  

---

## 4 代码结构拆解

### 4.1 色彩辅助函数（sRGB → Lab）

| 函数 | 作用 | 备注 |
|------|------|------|
| `_srgb_to_linear(x)` | Gamma 解码 sRGB → 线性色 | 分段 2.4 γ 曲线 |
| `_rgb_to_lab(rgb)`   | 批量把 RGB → CIE Lab    | 便于用 L 通道做测光 |
| `_lab_to_lch(lab)`   | Lab → LCh               | 内部暂未用到 |

> 这些函数只在 ROI 测光时调用，性能影响极小。

---

### 4.2 参数数据类 `ExposureParams`

| 字段 | 含义 | 默认 |
|------|------|------|
| `mode` | `'manual'` / `'deflicker'` | `'manual'` |
| `exposure` | 手动 EV（正值变亮，负值变暗） | `0.0` |
| `black` | 黑电平偏移（0-1） | `0.0` |
| `deflicker_percentile` | 去闪直方图百分位 | `50.0` |
| `deflicker_target_level` | 去闪目标亮度（以 EV 表示，相对原始白点） | `-4.0` |
| `compensate_exposure_bias` | 是否扣除 EXIF 曝光补偿 | `False` |
| `raw_black_level / raw_white_point` | 原始传感器黑 / 白电平 | `0 / 65535` |

---

### 4.3 曝光核心 `apply_exposure(img, params, exif_bias)`

1. 计算总 EV：  
   `ev = params.exposure - exif_bias (+ deflicker)`  
2. EV → 线性白点系数：`white = 2 ** (-ev)`  
3. 线性缩放 & 加黑：`(img - black) / (white - black)`  
4. 最终裁剪到 `[0,1]`。

---

### 4.4 序列去闪 `_compute_deflicker_correction`

1. 将帧转为 16-bit 灰度（或直接使用 raw）。  
2. 取 `deflicker_percentile` 百分位的像素值 `raw_val`。  
3. 计算该像素距白点的 EV：  

   ```
   ev = log2(raw_val - raw_black) - log2(raw_white - raw_black)
   ```  

4. 将其拉到目标级 `deflicker_target_level`：  

   ```
   correction = target - ev
   ```  

5. 返回 `correction`（以 EV 表示），在 `apply_exposure` 中加到手动 EV 上。

---

### 4.5 区域测光 / 曝光映射

* `measure_roi_lightness(img, roi)`  
  返回 ROI 内平均 Lab L（0-100）。
* `correct_exposure_for_roi(img, params, roi, target_L)`  
  1. 估算当前 L → 目标 L 的比例 `ratio`  
  2. 转为 EV：`delta_ev = log2(ratio)`  
  3. 叠加到 `params.exposure`（只在序列首帧调用一次）。  

适合固定机位的 timelapse：先手动模式对第一张定好 EV，然后片头片尾都能保持同一目标亮度。

---

### 4.6 I/O 与 EXIF 工具

| 函数 | 说明 |
|------|------|
| `_pil_to_float(img)` | Pillow 图像 → 0-1 float32 NumPy |
| `_float_to_pil(arr)` | 0-1 float → 8-bit Pillow 图像 |
| `_read_exif_bias(img)` | 解析 `ExposureBiasValue`，失败返回 0 |

---

### 4.7 命令行接口 `main()` & `argparse`

```bash
python exposure.py frames/*.jpg \
    --mode deflicker \
    --percentile 50 --target -4 \
    --bias
```

支持通配符、ROI、黑电平等所有参数；执行后自动在 `out/` 目录生成调整后的帧。  

---

## 5 示例

### 5.1 手动调一张 JPG +1.5 EV

```bash
python exposure.py img.jpg --ev 1.5
```

### 5.2 去闪 timelapse（取 70% 直方图，目标 −3 EV）

```bash
python exposure.py timelapse/*.tif \
    --mode deflicker \
    --percentile 70 --target -3
```

### 5.3 使用 ROI 把天空压到 L = 60

```bash
python exposure.py sky.jpg \
    --ev 0.0 --roi 100,50,500,300 --L 60
```

---
