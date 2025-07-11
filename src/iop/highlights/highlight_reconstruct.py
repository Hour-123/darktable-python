#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
highlight_reconstruct.py
纯 Python / CPU 版 darktable 高光重建（简化实现）
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Tuple

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import flood_fill
from skimage.measure import label, regionprops


# ---------- 参数与枚举 ---------- #
class Mode(IntEnum):
    CLIP = 0
    LCH = 1
    INPAINT = 2
    LAPLACIAN = 3
    SEGMENTS = 4
    OPPOSED = 5


class RecoveryMode(IntEnum):
    OFF = 0
    SMALL = 1
    LARGE = 2
    SMALLF = 3
    LARGEF = 4
    ADAPT = 5
    ADAPTF = 6


@dataclass
class HLParams:
    mode: Mode = Mode.OPPOSED
    clip: float = 1.0
    noise_level: float = 0.0
    iterations: int = 30
    scales: int = 7           # 2**scales 像素半径
    candidating: float = 0.4
    combine: float = 2.0
    recovery: RecoveryMode = RecoveryMode.OFF
    solid_color: float = 0.0


# ---------- 工具函数 ---------- #
def _to_float32(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """把输入 uint8/uint16/float 转为 0-1 范围 float32，并返回是否需要再转回整数"""
    need_back = img.dtype != np.float32
    img_f = img.astype(np.float32)
    if img_f.max() > 1.1:
        img_f /= 255.0 if img_f.max() <= 255 else img_f.max()
    img_f = np.clip(img_f, 0.0, 1.0)
    return img_f, need_back


def _from_float32(img_f: np.ndarray, need_back: bool, shape_like: np.ndarray) -> np.ndarray:
    if not need_back:
        return img_f
    out = np.clip(img_f * 255.0, 0, 255).astype(shape_like.dtype)
    return out


def _calc_mask(img: np.ndarray, thr: float) -> np.ndarray:
    """>=thr 视为裁剪"""
    return np.any(img >= thr, axis=2)


# ---------- 模式实现 ---------- #
def _clip(img: np.ndarray, p: HLParams) -> np.ndarray:
    return np.minimum(img, p.clip)


def _lch_desaturate(img: np.ndarray, p: HLParams) -> np.ndarray:
    mask = _calc_mask(img, p.clip)
    gray = img.mean(axis=2, keepdims=True)
    out = img.copy()
    out[mask] = gray[mask]
    return out


def _inpaint_magic(img: np.ndarray, p: HLParams) -> np.ndarray:
    """Magic-Lantern 方向插值，简单四方向均值"""
    H, W, _ = img.shape
    mask = _calc_mask(img, p.clip)

    # OpenCV 自带的 inpaint 仅支持单通道/3 通道 uint8，我们转换至 Lab 的 a,b
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # 处理 a,b 通道，亮度 L 保持
    for ch in (A, B):
        ch_f = ch.astype(np.float32)
        ch_f[mask] = 0
        # Telea 算法半径 3 px（足够快）
        ch_i = cv2.inpaint(ch_f, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        ch[:] = ch_i.astype(np.uint8)

    lab_f = cv2.merge([L, A, B])
    out = cv2.cvtColor(lab_f, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return out


def _opposed_lab(img: np.ndarray, p: HLParams) -> np.ndarray:
    """Lab 色度扩散：保 L，模糊 a,b"""
    mask = _calc_mask(img, p.clip)
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    # 使用引导滤波或高斯 + 蒸馏：这里只用高斯快速近似
    sigma = 3
    a_blur = cv2.GaussianBlur(a, (0, 0), sigma)
    b_blur = cv2.GaussianBlur(b, (0, 0), sigma)

    # 仅在 mask 区域替换
    a[mask] = a_blur[mask]
    b[mask] = b_blur[mask]

    lab_out = cv2.merge([L, a, b]).astype(np.uint8)
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return out


def _laplacian_multi(img: np.ndarray, p: HLParams) -> np.ndarray:
    """
    多尺度拉普拉斯 + 引导滤波近似。
    使用金字塔迭代扩散色度，保持亮度。
    """
    mask = _calc_mask(img, p.clip)
    if not mask.any():
        return img.copy()

    # 转 Lab 处理 a,b
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    a_out, b_out = a.copy(), b.copy()
    # 金字塔层数
    layers = min(p.scales, 8)
    a_pyr = [a_out]
    b_pyr = [b_out]
    m_pyr = [mask.astype(np.float32)]

    # 构建金字塔
    for _ in range(layers - 1):
        a_pyr.append(cv2.pyrDown(a_pyr[-1]))
        b_pyr.append(cv2.pyrDown(b_pyr[-1]))
        m_pyr.append(cv2.pyrDown(m_pyr[-1]))

    # 自顶向下迭代
    for lvl in reversed(range(layers)):
        a_lvl, b_lvl, m_lvl = a_pyr[lvl], b_pyr[lvl], m_pyr[lvl] > 0.1

        # 对色度做 guided filter / fast bilateral，快速用高斯迭代逼近
        for _ in range(p.iterations // layers):
            a_blur = cv2.GaussianBlur(a_lvl, (0, 0), 1.5)
            b_blur = cv2.GaussianBlur(b_lvl, (0, 0), 1.5)
            a_lvl[m_lvl] = a_blur[m_lvl]
            b_lvl[m_lvl] = b_blur[m_lvl]

        # 若非顶层，向上一层上采样叠加
        if lvl > 0:
            up_shape = (a_pyr[lvl - 1].shape[1], a_pyr[lvl - 1].shape[0])
            a_up = cv2.pyrUp(a_lvl, dstsize=up_shape)
            b_up = cv2.pyrUp(b_lvl, dstsize=up_shape)
            a_pyr[lvl - 1][m_pyr[lvl - 1] > 0.1] = a_up[m_pyr[lvl - 1] > 0.1]
            b_pyr[lvl - 1][m_pyr[lvl - 1] > 0.1] = b_up[m_pyr[lvl - 1] > 0.1]

    lab_out = cv2.merge([L, a_pyr[0], b_pyr[0]]).astype(np.uint8)
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return out


def _segments(img: np.ndarray, p: HLParams) -> np.ndarray:
    """
    极简分割版：找连通域，> combine 阈值再 morphological closing，
    然后用区域边界平均色度填充。
    """
    mask = _calc_mask(img, p.clip)
    if not mask.any():
        return img.copy()

    # 形态学合并
    kernel_size = int(max(1, p.combine))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_comb = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    labeled = label(mask_comb)
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    a_out, b_out = a.copy(), b.copy()

    for region in regionprops(labeled):
        coords = region.coords
        if coords.shape[0] < 10:  # 太小的不处理
            continue
        # 找到边界 3px 膨胀后减原区域 → ring 区域作为候选
        m_region = np.zeros_like(mask_comb, bool)
        m_region[coords[:, 0], coords[:, 1]] = True
        ring = cv2.dilate(m_region.astype(np.uint8), np.ones((5, 5), np.uint8)) \
               .astype(bool) & (~m_region)
        if not ring.any():
            continue
        # 取 ring 区域 a,b 平均，填入区域
        a_val = a[ring].mean()
        b_val = b[ring].mean()
        a_out[m_region] = a_val
        b_out[m_region] = b_val

    lab_out = cv2.merge([L, a_out, b_out]).astype(np.uint8)
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # 可选添加噪声
    if p.noise_level > 1e-3:
        noise = np.random.normal(0, p.noise_level, out.shape).astype(np.float32)
        out = np.clip(out + noise * mask[..., None], 0, 1)
    return out


# ---------- 总调度函数 ---------- #
def reconstruct_highlights(img: np.ndarray, params: HLParams) -> np.ndarray:
    img_f, need_back = _to_float32(img)

    if params.mode == Mode.CLIP:
        out = _clip(img_f, params)
    elif params.mode == Mode.LCH:
        out = _lch_desaturate(img_f, params)
    elif params.mode == Mode.INPAINT:
        out = _inpaint_magic(img_f, params)
    elif params.mode == Mode.LAPLACIAN:
        out = _laplacian_multi(img_f, params)
    elif params.mode == Mode.SEGMENTS:
        out = _segments(img_f, params)
    elif params.mode == Mode.OPPOSED:
        out = _opposed_lab(img_f, params)
    else:
        raise ValueError("Unknown mode")

    return _from_float32(out, need_back, img)


# ---------- 示例主函数 ---------- #
def main():
    import argparse, os

    parser = argparse.ArgumentParser(description="Python highlight reconstruction demo")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("-o", "--output", default="out.png", help="输出文件名")
    parser.add_argument("-m", "--mode", type=int, default=5, choices=range(6),
                        help="算法模式 0 clip |1 lch |2 inpaint |3 laplacian |4 segments |5 opposed")
    parser.add_argument("-c", "--clip", type=float, default=1.0, help="裁剪阈值 (0-2)")
    args = parser.parse_args()

    img = cv2.cvtColor(cv2.imread(args.input, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    params = HLParams(mode=Mode(args.mode), clip=args.clip)

    out = reconstruct_highlights(img, params)

    cv2.imwrite(args.output,
                cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"done. saved -> {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()