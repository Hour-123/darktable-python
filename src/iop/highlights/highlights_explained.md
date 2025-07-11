# Highlight Reconstruction (纯 Python 版)

本项目提供了一个纯 Python/CPU 实现的高光重建（Highlight Reconstruction）工具，改编自 Darktable 的高光重建模块。针对过曝（clipped）区域，提供多种色度重建算法，可在普通 RGB(JPEG) 图像上使用。

---

## 功能概览

- 自动检测图像中过曝（像素值超过阈值）的区域
- 在 CIE-Lab 或 LCh 颜色空间中对过曝区域进行色度重建
- 提供 6 种重建模式：
  1. **CLIP**   直接裁剪到阈值  
  2. **LCH**   LCh 空间去饱和（过曝处转灰）  
  3. **INPAINT** 使用 OpenCV Telea 算法在 Lab 色度通道插值  
  4. **LAPLACIAN** 多尺度金字塔 + 高斯迭代扩散色度  
  5. **SEGMENTS** 连通域分割 + 边界平均色度填充  
  6. **OPPOSED** Lab 空间高斯模糊，仅在过曝区域替换色度  

---

## 依赖

- Python 3.7+
- numpy
- opencv-python
- scipy
- scikit-image
---

## 核心模块：highlight_reconstruct.py

```text
highlight_reconstruct.py
├─ Enum/IntEnum 定义
│   ├─ Mode         # 6 种重建策略
│   └─ RecoveryMode # （暂未使用，可扩展）
│
├─ @dataclass HLParams      # 参数容器
│   ├─ mode          : Mode        # 0~5
│   ├─ clip          : float       # 过曝阈值
│   ├─ noise_level   : float       # segments 模式噪声强度
│   ├─ iterations    : int         # laplacian 迭代次数
│   ├─ scales        : int         # laplacian 金字塔层数
│   ├─ candidating   : float       # （保留）
│   ├─ combine       : float       # segments 合并核大小
│   ├─ recovery      : RecoveryMode# （保留）
│   └─ solid_color   : float       # （保留）
│
├─ 工具函数
│   ├─ _to_float32(img)        # 归一化到 [0,1] float32
│   ├─ _from_float32(img,…)    # 反归一化到原始 dtype
│   └─ _calc_mask(img,thr)     # RGB 任一通道 ≥ thr → mask
│
├─ 各模式实现
│   ├─ _clip
│   ├─ _lch_desaturate
│   ├─ _inpaint_magic
│   ├─ _opposed_lab
│   ├─ _laplacian_multi
│   └─ _segments
│
├─ reconstruct_highlights(img, params)
│   # 入口，根据 params.mode 自动调用对应子函数
│
└─ main() & CLI
    python highlight_reconstruct.py input.jpg -o out.png -m 5 -c 1.0
```

### CLI 用法

```bash
python highlight_reconstruct.py <input> [-o OUTPUT] [-m MODE] [-c CLIP]
```

- `<input>`：输入图像路径（RGB 格式，uint8/uint16/float）
- `-o, --output`：输出文件（默认 `out.png`）
- `-m, --mode`：模式编号 0~5
- `-c, --clip`：过曝阈值（0~2）

---

## 测试脚本：test_highlight_reconstruct.py

- 在同一目录下读取 `input.jpg`（或通过命令行指定）
- 自动处理 BGR↔RGB 通道互换
- 对 JPEG 做 **sRGB → 线性空间**（gamma 解码）
- 调用 `reconstruct_highlights`
- **线性 → sRGB**（gamma 编码）→ 保存 JPEG

```bash
# 默认行为（脚本目录下有 input.jpg）
python test_highlight_reconstruct.py

# 或指定输入、输出目录、参数
python test_highlight_reconstruct.py my.jpg -o results --clip 1.2 --noise 0.01 --scales 6
```

处理后你会在 `outputs/` 目录下看到：
```
input_clip.jpg
input_lch.jpg
input_inpaint.jpg
input_laplacian.jpg
input_segments.jpg
input_opposed.jpg
```

---

## 使用示例

1. 准备一张过曝（clipped）且偏洋红的 JPG 图 `input.jpg`  
2. 直接运行测试脚本：

   ```bash
   python test_highlight_reconstruct.py
   ```

3. 查看 `outputs/` 下的六张重建对比图  

---

## 常见问题

- **为什么输出图像整体偏绿？**  
  - 忘记做 `cv2.imread` 后的 BGR→RGB，或者写出前未做 RGB→BGR。  
  - 在非线性 sRGB 空间直接做 Lab 运算，需先 gamma 解码→算子→gamma 编码。

- **如何调整过曝判定阈值？**  
  使用 `--clip <float>` 参数（默认 1.0，对应 255/255）。

- **segments 模式噪声太明显？**  
  调小 `--noise`（默认为 0），或直接设为 0 关闭噪声。

- **想改 laplacian 金字塔层数？**  
  用 `--scales <int>`（最大 8）。

---
