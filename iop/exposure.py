"""
exposure - A Python implementation of the exposure IOP.
"""

import numpy as np

def exposure_v1(image_data: np.ndarray, exposure_ev: float, black_level: float = 0.0) -> np.ndarray:
    """
    Applies exposure and black level correction based on darktable's C implementation.

    This model matches the core principles of the exposure module.
    In a linear color space, exposure compensation is a simple multiplication.

    Args:
        image_data (np.ndarray): The input image data in a linear format,
                                 expected to be float (e.g., 0.0 to 1.0).
        exposure_ev (float): The exposure compensation in EV (Exposure Values).
                             A value of +1.0 doubles the brightness.
        black_level (float): The black level correction. This is an offset
                             applied before the exposure multiplication.

    Returns:
        np.ndarray: The processed image data.
    """
    if image_data.dtype != np.float32 and image_data.dtype != np.float64:
        raise ValueError("Input image data must be of float type for processing.")

    # The C source code's core logic is:
    # out = (in - black) * scale
    # where white = exp2(-exposure)
    # and scale = 1.0 / (white - black)

    white = 2.0 ** (-exposure_ev)

    # Avoid division by zero if white point and black point are the same.
    if np.isclose(white, black_level):
        scale = 1.0
    else:
        scale = 1.0 / (white - black_level)

    # Apply the formula
    processed_image = (image_data - black_level) * scale

    return processed_image


# For future development, we can create a class to hold parameters,
# similar to how darktable's structs work.

class Exposure:
    """
    A class-based representation of the exposure module.
    This will allow us to store parameters and have a more robust `process` method.
    """
    def __init__(self, exposure_ev: float = 0.0, black_level: float = 0.0):
        self.exposure_ev = exposure_ev
        self.black_level = black_level
    
    def process(self, image_data: np.ndarray) -> np.ndarray:
        """
        Applies the stored exposure and black level correction,
        matching the logic from darktable's C source code.
        """
        # The C source code's core logic is:
        # out = (in - black) * scale
        # where white = exp2(-exposure)
        # and scale = 1.0 / (white - black)

        white = 2.0 ** (-self.exposure_ev)

        # Avoid division by zero if white point and black point are the same.
        if np.isclose(white, self.black_level):
            scale = 1.0
        else:
            scale = 1.0 / (white - self.black_level)

        return (image_data - self.black_level) * scale


if __name__ == '__main__':
    import os
    import rawpy
    import imageio

    print("--- Running Exposure Module Test ---")

    # 构建输入和输出文件的路径
    # 我们从 darktable_python/iop/ 目录运行，所以需要向上两级
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_raw_path = os.path.join(base_dir, '..', 'database', 'sample2.dng')
    output_image_path = os.path.join(base_dir, 'output', 'exposure_test_output.jpg')
    
    # 检查输入文件是否存在
    if not os.path.exists(input_raw_path):
        print(f"错误: 输入的 RAW 文件未找到，路径: {input_raw_path}")
        print("请确保测试RAW文件存在于 `database` 目录下。")
    else:
        try:
            print(f"1. 正在读取并解码 RAW 文件: {input_raw_path}")
            with rawpy.imread(input_raw_path) as raw:
                # 关键步骤: 解码为16位线性图像数据
                # gamma=(1, 1) 禁用了伽马校正，no_auto_bright=True 禁用了自动亮度
                raw_image = raw.postprocess(
                    gamma=(1, 1),
                    no_auto_bright=True,
                    output_bps=16,
                    use_camera_wb=True
                )

            # 将16位整型 (0-65535) 转换为浮点型 (0.0-1.0)
            linear_image_float = raw_image.astype(np.float32) / 65535.0
            print(f"   - 解码完成, 图像尺寸: {linear_image_float.shape}")

            # 2. 初始化并应用曝光模块
            exposure_val = -0.5
            black_val = 0.01
            print(f"2. 应用曝光模块 (曝光: {exposure_val} EV, 黑场: {black_val})")
            exposure_module = Exposure(exposure_ev=exposure_val, black_level=black_val)
            processed_image = exposure_module.process(linear_image_float)

            # 3. 准备输出: 进行伽马校正以便查看
            print("3. 应用伽马校正 (gamma=2.2) 以便在屏幕上正确显示")
            # 首先将数值裁剪到 [0, 1] 范围，防止乘法后超出范围
            clipped_image = np.clip(processed_image, 0, 1)
            gamma_corrected_image = clipped_image ** (1 / 2.2)

            # 将浮点型 (0.0-1.0) 转换回8位整型 (0-255) 以便保存为JPG
            output_image_uint8 = (gamma_corrected_image * 255).astype(np.uint8)

            # 4. 保存结果
            print(f"4. 正在保存结果到: {output_image_path}")
            imageio.imwrite(output_image_path, output_image_uint8)
            print("--- 测试成功完成! ---")

        except Exception as e:
            print(f"测试过程中发生错误: {e}") 