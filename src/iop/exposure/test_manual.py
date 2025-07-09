# test_manual.py
from exposure import apply_exposure, ExposureParams, Mode, _pil_to_float, _float_to_pil
from PIL import Image
import numpy as np

def main():
    # 1) 加载原图
    src = Image.open("img.jpg").convert("RGB")
    arr  = _pil_to_float(src)   # float32, [0..1]
    
    # 2) 打印原图的平均亮度（简单手段：RGB 三通道平均）
    print("原图平均亮度 (RGB mean) =", float(arr.mean()))
    
    # 3) 配置参数：manual 模式，加 +1 EV，不做黑电平偏移
    params = ExposureParams(
        mode    = Mode.MANUAL,
        exposure= 0.5,
        black   = 0.0
    )
    
    # 4) 应用曝光调整
    out_arr = apply_exposure(arr, params, exif_bias=0.0)
    
    # 5) 打印调整后平均亮度
    print("调整后平均亮度 =", float(out_arr.mean()))
    
    # 6) 保存结果
    out_img = _float_to_pil(out_arr)
    out_img.save("img_ev1.png")
    print("已保存：img_ev1.png")

if __name__ == "__main__":
    main()