#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wb.py —— 把 Darktable 的“自动白平衡”搬到 Python
完全兼容 rawpy 0.10 – 0.25+
"""

from __future__ import annotations
import json, os, sys, inspect, textwrap
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import imageio.v3 as iio
import rawpy


# ------------------------------------------------------------
# 0. 打印环境信息，方便排障
# ------------------------------------------------------------
def _banner() -> None:
    from platform import platform
    print(textwrap.dedent(f"""\
        =====  WB SCRIPT  =====
        rawpy  : {rawpy.__version__}
        python : {sys.version.split()[0]}
        system : {platform()}
        ========================
    """))


# ------------------------------------------------------------
# 1. 读取 CameraMultipliers（DNG 内 Tag）
# ------------------------------------------------------------
def read_cam_mul(dng: os.PathLike) -> Tuple[float, float, float] | None:
    with rawpy.imread(os.fspath(dng)) as r:
        cm = r.camera_whitebalance            # (R, G1, B, G2)
        if not cm or all(v == 0 for v in cm):
            return None
        r_, g1, b_, g2 = cm
        g = (g1 + g2) / 2
        return (r_ / g, 1.0, b_ / g)


# ------------------------------------------------------------
# 2. JSON 里找相机白平衡预设
# ------------------------------------------------------------
def read_wb_json(make: str, model: str, json_path: os.PathLike,
                 prefer: Sequence[str] = ("Daylight", "Flash", "Cloudy")
                 ) -> Tuple[float, float, float] | None:
    make, model = make.lower().strip(), model.lower().strip()

    with open(json_path, encoding='utf-8') as fp:
        data = json.load(fp)

    for entry in data.get("wb_presets", []):
        if entry["maker"].lower() != make:
            continue
        for cam in entry["models"]:
            if cam["model"].lower() != model:
                continue
            # 优先选用户想要的场景
            for want in prefer:
                preset = next((p for p in cam["presets"]
                               if p["name"].lower() == want.lower()), None)
                if preset:
                    r, g, b, _ = preset["channels"]
                    return (r / g, 1.0, b / g)
            # 找不到就拿第 1 个
            r, g, b, _ = cam["presets"][0]["channels"]
            return (r / g, 1.0, b / g)
    return None


# ------------------------------------------------------------
# 3. 综合选择 (Tag → JSON → 中性)
# ------------------------------------------------------------
def get_auto_wb(dng: os.PathLike, wb_json: os.PathLike) -> Tuple[float, float, float]:
    wb = read_cam_mul(dng)
    if wb:
        return wb

    with rawpy.imread(os.fspath(dng)) as r:
        wb = read_wb_json(r.metadata.make or "", r.metadata.model or "", wb_json)
    if wb:
        return wb

    print("WARNING  : no WB tag / preset found, fall back to neutral 1,1,1")
    return (1.0, 1.0, 1.0)


# ------------------------------------------------------------
# 4. 与 rawpy 版本无关的后处理封装
# ------------------------------------------------------------
def postprocess_compat(
    raw: rawpy.RawPy,
    *, user_wb: Tuple[float, float, float],
    output_bps: int = 16,
    gamma: Tuple[float, float] | None = (1, 1),
) -> np.ndarray:
    """
    根据 RawPy.postprocess 的函数签名自动裁剪/补充参数，
    同时把 user_wb 转成旧版需要的 4 通道。
    """
    sig = inspect.signature(raw.postprocess).parameters
    kwargs: dict = {"user_wb": user_wb,
                    "no_auto_bright": True,
                    "output_bps": output_bps}

    if "use_camera_matrix" in sig:
        kwargs["use_camera_matrix"] = True
    else:
        # 老版缺该参数，而且要求 4 通道 WB
        r, g, b = user_wb
        kwargs["user_wb"] = (r, g, b, g)      # G2 = G1

    if gamma and "gamma" in sig:
        kwargs["gamma"] = gamma

    # 删除任何旧版不存在的键
    for k in list(kwargs):
        if k not in sig:
            kwargs.pop(k)

    return raw.postprocess(**kwargs)


# ------------------------------------------------------------
# 5. DNG → PNG 主函数
# ------------------------------------------------------------
def dng_to_png(
    dng_path: os.PathLike,
    png_path: os.PathLike,
    wb_json: os.PathLike,
) -> None:
    wb = get_auto_wb(dng_path, wb_json)
    with rawpy.imread(os.fspath(dng_path)) as raw:
        rgb = postprocess_compat(raw, user_wb=wb)
    iio.imwrite(os.fspath(png_path), rgb)
    print("saved →", png_path)


# ------------------------------------------------------------
# 6. CLI 示例（改这里！）
# ------------------------------------------------------------
if __name__ == "__main__":
    _banner()

    # ❶ 你的 DNG / RAW 文件路径
    DNG = Path(r"D:\25Summer\Darktable_Python\wb_manual\test\test_phone.dng")

    # ❷ darktable 的 wb_presets.json 路径
    WB_JSON = Path(r"D:\25Summer\Darktable_Python\wb_manual\wb_presets.json")

    # ❸ 输出 PNG 路径（默认与原文件同名）
    PNG = DNG.with_suffix(".auto_wb.png")

    if not DNG.exists():
        sys.exit(f"[ERROR] DNG not found: {DNG}")
    if not WB_JSON.exists():
        sys.exit(f"[ERROR] WB json not found: {WB_JSON}")

    dng_to_png(DNG, PNG, WB_JSON)