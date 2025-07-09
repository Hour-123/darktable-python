# python_ISP

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

### 关于 "16-bit→8-bit" 可视化
传感器输出往往是 10/12/14/16 bit，若直接写入 PNG，普通查看器通常**只取高 8 bit**，导致图像发暗。`run.py` 内部使用
```python
img8 = clip(img16 * 255 / white_level)
```
将整幅数据线性缩放到 `[0,255]` 后转 `uint8`，既不影响前面高精度计算，又保证任何查看器都能正常预览。

---
如需进一步实验/扩展，可在 `model/` 中新增模块后，在 `run.py` 对应位置调用即可。

_Author: Jiahao Huang_
_Date: 2025/07/09_

