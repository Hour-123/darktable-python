# test_deflicker.py
from exposure import apply_exposure, _compute_deflicker_correction, ExposureParams, Mode, _pil_to_float, _float_to_pil
from PIL import Image
import numpy as np, glob

def main():
    files = sorted(glob.glob("frame*.jpg"))
    # 设定：deflicker 百分位 20%，目标 EV=-1.5
    params = ExposureParams(
        mode                    = Mode.DEFLICKER,
        deflicker_percentile    = 20.0,
        deflicker_target_level  = -1.5,
        raw_black_level         = 0,
        raw_white_point         = 65535
    )
    for f in files:
        img = Image.open(f).convert("RGB")
        arr = _pil_to_float(img)
        corr = _compute_deflicker_correction(arr, params)
        out = apply_exposure(arr, params)
        print(f"{f}: deflicker correction = {corr:.3f} EV, new exposure={params.exposure:.3f}")
        _float_to_pil(out).save("out_"+f)

if __name__ == "__main__":
    main()