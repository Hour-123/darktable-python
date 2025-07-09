"""
exposure.py – minimal, self-contained re-implementation of Darktable’s
“exposure” module (computation part, no GUI).

author : chatGPT (2025-07-07)
license: MIT
-----------------------------------------------------------------------
Implemented features
1. Manual mode
   out = (in - black) * 2 ** EV
2. Automatic “deflicker” mode (for whole image sequences)
   • pick the N-th percentile of the raw histogram
   • compute how many EV this value is below the sensor white point
   • add/subtract the difference to reach a user chosen target level
3. Optional compensation of the camera exposure bias (Exif tag “ExposureBiasValue”)
4. Area exposure mapping helpers (measure + correct) – work on Lab lightness
   (ROI interface only, no on-screen picker)
-----------------------------------------------------------------------
External packages
  pip install pillow numpy
-----------------------------------------------------------------------
"""

from __future__ import annotations
import math
import pathlib
from dataclasses import dataclass, asdict
from typing import Sequence, Tuple, Iterable

import numpy as np
from PIL import Image, TiffImagePlugin, JpegImagePlugin

#############################
# 1. colour helpers (sRGB)  #
#############################
_D65 = (95.047, 100.0, 108.883)
# sRGB linear transform matrix (IEC 61966-2-1)
_M_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """gamma-decode sRGB [0-1] -> linear"""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB [0-1] -> Lab (float32), vectorised on final axis"""
    rgb_lin = _srgb_to_linear(rgb)
    xyz = rgb_lin @ _M_SRGB_TO_XYZ.T  # [...,3]
    xr, yr, zr = xyz[..., 0] / _D65[0], xyz[..., 1] / _D65[1], xyz[..., 2] / _D65[2]

    def f(t):
        delta = 6 / 29
        return np.where(t > delta ** 3, np.cbrt(t), (t / (3 * delta ** 2) + 4 / 29))

    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack((L, a, b), axis=-1)


def _lab_to_lch(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360
    return np.stack((L, C, h), axis=-1)


#############################
# 2. parameter container    #
#############################
class Mode:
    MANUAL = "manual"
    DEFLICKER = "deflicker"


@dataclass
class ExposureParams:
    mode: str = Mode.MANUAL
    exposure: float = 0.0                  # EV
    black: float = 0.0                     # offset (0-1 float for processed data)
    deflicker_percentile: float = 50.0     # %
    deflicker_target_level: float = -4.0   # EV
    compensate_exposure_bias: bool = False

    # raw meta (used only by deflicker). provide sensor black + white for each image
    raw_black_level: int = 0
    raw_white_point: int = 65535


##################################
# 3. core exposure manipulation  #
##################################
def _ev_to_white(ev: float) -> float:
    """EV -> white level coefficient"""
    return 2.0 ** (-ev)


def _white_to_ev(white: float) -> float:
    return -math.log2(max(white, 1e-20))


def apply_exposure(img: np.ndarray, p: ExposureParams, exif_bias: float = 0.0) -> np.ndarray:
    """
    img : float32 array 0-1 (RGB or gray).
    Returns new float32 array 0-1 (clipped).
    """
    ev = p.exposure
    if p.compensate_exposure_bias:
        ev -= exif_bias

    # deflicker overrides ev
    if p.mode == Mode.DEFLICKER:
        ev += _compute_deflicker_correction(img, p)

    white = _ev_to_white(ev)
    scale = 1.0 / (white - p.black)

    out = (img - p.black) * scale
    return np.clip(out, 0.0, 1.0)


#############################
# 4. deflicker computation  #
#############################
def _raw_to_ev(raw: int, p: ExposureParams) -> float:
    raw_max = p.raw_white_point - p.raw_black_level
    raw_val = max(raw - p.raw_black_level, 1)
    return -math.log2(raw_max) + math.log2(raw_val)


def _compute_deflicker_correction(img: np.ndarray, p: ExposureParams) -> float:
    """
    Works on a luminance version of the image (can be RGB 8/16 or a raw frame already).
    """
    # If already a 2-D raw array (16-bit), use it directly for histogram.
    if img.ndim == 2:
        raw16 = img.astype(np.uint16)
    else:  # RGB – approximate luminance into 16-bit space
        y = (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722)
        raw16 = np.round(y * 65535).astype(np.uint16)

    # percentile
    thr = p.deflicker_percentile
    raw_val = int(np.percentile(raw16, thr))
    ev = _raw_to_ev(raw_val, p)
    correction = p.deflicker_target_level - ev
    return correction


##################################################
# 5. area exposure mapping (measure / correct)   #
##################################################
def measure_roi_lightness(img: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    """
    Return average Lab L (0-100) inside ROI.
    roi = (x0, y0, x1, y1) in pixel coords.
    """
    x0, y0, x1, y1 = roi
    sub = img[y0:y1, x0:x1]
    lab = _rgb_to_lab(sub)
    return float(lab[..., 0].mean())


def correct_exposure_for_roi(img: np.ndarray,
                             p: ExposureParams,
                             roi: Tuple[int, int, int, int],
                             target_L: float,
                             exif_bias: float = 0.0) -> float:
    """
    Computes how many EV must be added to map ROI to the desired Lab L,
    updates p.exposure and returns the delta EV applied.
    """
    current_L = measure_roi_lightness(img, roi)
    if current_L <= 0.1:
        return 0.0

    # Rough ratio between current and target lightness (using Y approximations)
    ratio = target_L / current_L
    delta_ev = math.log2(ratio)

    if p.compensate_exposure_bias:
        delta_ev -= exif_bias

    p.exposure += delta_ev
    return delta_ev


#############################
# 6. I/O utilities          #
#############################
def _pil_to_float(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    else:  # uint8
        return arr.astype(np.float32) / 255.0


def _float_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _read_exif_bias(img: Image.Image) -> float:
    """Try to read ExposureBiasValue (in EV) from EXIF; return 0 if unavailable."""
    try:
        exif = img.getexif()
        tag = next(k for k, v in TiffImagePlugin.TAGS_V2.items() if v == "ExposureBiasValue")
        value = exif.get(tag, 0)
        if isinstance(value, JpegImagePlugin.IFDRational) or isinstance(value, tuple):
            # convert rational (num,den)
            if isinstance(value, tuple):
                num, den = value
            else:
                num, den = value.numerator, value.denominator
            return float(num) / float(den)
        return float(value)
    except Exception:
        return 0.0


#############################
# 7. demo / main            #
#############################
def main(seq: Sequence[str],
         params: ExposureParams,
         out_dir: str = "out",
         roi: Tuple[int, int, int, int] | None = None,
         target_L: float = 50.0):
    """
    seq : list of image paths (all frames of a timelapse, for example)
    params : initial parameters (may be mutated)
    out_dir : directory to write results
    roi / target_L : enable area mapping if supplied
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(exist_ok=True)

    for idx, f in enumerate(seq):
        img_pil = Image.open(f).convert("RGB")
        frame = _pil_to_float(img_pil)
        exif_bias = _read_exif_bias(img_pil)

        if roi and params.mode == Mode.MANUAL:  # area mapping only useful in manual mode
            if idx == 0:  # compute once on the reference frame
                delta = correct_exposure_for_roi(frame, params, roi, target_L, exif_bias)
                print(f"[mapping] adjusted exposure by {delta:+.2f} EV to hit target L={target_L}")

        out = apply_exposure(frame, params, exif_bias)
        Image.fromarray((out * 255).astype(np.uint8)).save(out_path / f"adj_{idx:04d}.png")

        print(f"{f}  ->  EV={params.exposure:+.3f} black={params.black:+.4f}")

    print("\nFinished. Parameters used:")
    print(asdict(params))


if __name__ == "__main__":
    # -------------------- example usage --------------------
    # 1) manual single picture
    #    python exposure.py img.jpg
    #
    # 2) deflicker over a timelapse
    #    python exposure.py frames/*.tif --mode deflicker --percentile 50 --target -4
    #
    import argparse, sys, glob

    ap = argparse.ArgumentParser(description="Simple exposure module re-implementation")
    ap.add_argument("files", nargs="+", help="input image(s)")
    ap.add_argument("--mode", choices=[Mode.MANUAL, Mode.DEFLICKER],
                    default=Mode.MANUAL)
    ap.add_argument("--ev", type=float, default=0.0, help="manual mode EV")
    ap.add_argument("--black", type=float, default=0.0, help="black offset 0-1")
    ap.add_argument("--percentile", type=float, default=50.0, help="deflicker percentile")
    ap.add_argument("--target", type=float, default=-4.0, help="deflicker target EV")
    ap.add_argument("--bias", action="store_true", help="compensate EXIF exposure bias")
    ap.add_argument("--roi", metavar="x0,y0,x1,y1",
                    help="area mapping rectangle (only manual mode)")
    ap.add_argument("--L", type=float, default=50.0, help="target L for ROI")
    args = ap.parse_args()

    # expand wildcards if shell didn’t already
    files: Iterable[str] = []
    for pat in args.files:
        files = (*files, *glob.glob(pat))

    params = ExposureParams(mode=args.mode,
                            exposure=args.ev,
                            black=args.black,
                            deflicker_percentile=args.percentile,
                            deflicker_target_level=args.target,
                            compensate_exposure_bias=args.bias)

    roi_tuple = tuple(map(int, args.roi.split(","))) if args.roi else None
    main(files, params, out_dir="out", roi=roi_tuple, target_L=args.L)