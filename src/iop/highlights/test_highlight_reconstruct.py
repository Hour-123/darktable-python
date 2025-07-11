#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_highlight_reconstruct.py

修正版：自动做 BGR<->RGB 通道互换 + sRGB <-> 线性空间 gamma 编/解码，
才能在 Lab 空间里正确做高光重建，不至于一片绿色。
"""

import os
import argparse
import numpy as np
import cv2
from highlight_reconstruct import reconstruct_highlights, HLParams, Mode

def srgb_to_linear(img_srgb: np.ndarray) -> np.ndarray:
    """把 0-1 之间的 sRGB（带 gamma）解码到线性空间"""
    mask = img_srgb <= 0.04045
    img_lin = np.empty_like(img_srgb, dtype=np.float32)
    # 低亮度分量
    img_lin[mask] = img_srgb[mask] / 12.92
    # 高亮度分量
    img_lin[~mask] = ((img_srgb[~mask] + 0.055) / 1.055) ** 2.4
    return img_lin

def linear_to_srgb(img_lin: np.ndarray) -> np.ndarray:
    """把线性空间(0-1)编码回带 gamma 的 sRGB(0-1)"""
    mask = img_lin <= 0.0031308
    out = np.empty_like(img_lin, dtype=np.float32)
    out[mask] = img_lin[mask] * 12.92
    out[~mask] = 1.055 * np.power(img_lin[~mask], 1/2.4) - 0.055
    return np.clip(out, 0.0, 1.0)

def main():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    default_input = os.path.join(script_dir, 'input.jpg')

    parser = argparse.ArgumentParser(
        description="Batch-test highlight reconstruction modes (with correct color pipeline)"
    )
    parser.add_argument(
        'input',
        nargs='?',
        default=default_input,
        help='输入 JPG（默认同目录下的 input.jpg）'
    )
    parser.add_argument('-o', '--output_dir', default=os.path.join(script_dir, 'outputs'),
                        help="输出目录（默认 ./outputs）")
    parser.add_argument('-c', '--clip', type=float, default=1.0,
                        help="裁剪阈值 clip（0~2）")
    parser.add_argument('--noise', type=float, default=0.0,
                        help="segments 模式下的噪声强度")
    parser.add_argument('--scales', type=int, default=7,
                        help="laplacian 模式下的 scales")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 读取 BGR
    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[Error] 无法读取图像: {args.input}")
        return

    # 2) BGR -> RGB 并规范到 0-1
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 3) sRGB -> 线性空间
    img_lin = srgb_to_linear(img_rgb)

    # 4) 对每种模式分别重建
    for mode in Mode:
        params = HLParams(
            mode=mode,
            clip=args.clip,
            noise_level=args.noise,
            scales=args.scales
        )
        print(f"Processing mode {mode.name} ({mode.value}) ...")
        out_lin = reconstruct_highlights(img_lin, params)

        # 5) 线性 -> sRGB
        out_srgb = linear_to_srgb(out_lin)

        # 6) RGB -> BGR 写回磁盘
        out_bgr = cv2.cvtColor(
            (out_srgb * 255.0).clip(0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
        fname = f"{os.path.splitext(os.path.basename(args.input))[0]}_" \
                f"{mode.name.lower()}.jpg"
        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, out_bgr)
        print(f"  saved -> {out_path}")

    print("All done.")

if __name__ == "__main__":
    main()