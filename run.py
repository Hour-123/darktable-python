# @Author: Jiahao Huang
# @Date: 2025-07-09 
# @Description: run demo

import sys
from isp_pipeline import ISP_Pipeline
from path import Path
import os
import time
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__) + '/model')
sys.path.insert(0, os.path.dirname(__file__) + '/config')
sys.path.insert(0, os.path.dirname(__file__) + '/assets')
sys.path.insert(0, os.path.dirname(__file__) + '/test_images')
sys.path.insert(0, os.path.dirname(__file__))

def run_demo():
    """
    run demo
    
    description:
        this is a demo for ISP Pipeline
        
    step:
        1. get the ISP Pipeline from yaml
        2. run the ISP Pipeline
        3. get the ISP Pipeline output
    
    usage:
        run_demo()
    """
    root_path = Path(os.path.abspath(__file__)).parent
    yaml_path = root_path / 'config' / 'isp_config.yaml'
    ISP_Pipeline(config_path=yaml_path).run()
    

def get_module_output():
    from model.dpc import DPC
    from model.awb import AWB
    from model.blc import BLC
    from model.ccm import CCM
    from model.cfa import CFA
    from model.fir import FIR
    from model.gmc import GMC

    root_path = Path(os.path.abspath(__file__)).parent
    cfg = yaml.safe_load(open(root_path / 'config' / 'isp_config.yaml', 'r'))

    raw_path = cfg['RAW_img_path']
    raw = FIR(RAW_img_path = raw_path).run() # 这个步骤会更新配置文件
    
    # 重新读取配置文件，确保使用 FIR 更新后的配置
    cfg = yaml.safe_load(open(root_path / 'config' / 'isp_config.yaml', 'r'))

    # 把 10/12/14/16 bit 数据映射到 8-bit 便于可视化
    def to_8bit(img: np.ndarray, white_level: int) -> np.ndarray:
        scale = 255.0 / white_level
        out = (img.astype(np.float32) * scale + 0.5).clip(0, 255).astype(np.uint8)
        return out

    wl = int(cfg.get('white_level', 1023))

    cv2.imwrite(root_path / 'outputs' / '01_raw.png', to_8bit(raw, wl))
    dpc = DPC(raw, **cfg).run()
    cv2.imwrite(root_path / 'outputs' / '02_dpc.png', to_8bit(dpc, wl))
    blc = BLC(dpc, **cfg).run()
    cv2.imwrite(root_path / 'outputs' / '03_blc.png', to_8bit(blc, wl))
    awb = AWB(blc, **cfg).run()
    cv2.imwrite(root_path / 'outputs' / '04_awb.png', to_8bit(awb, wl))
    cfa = CFA(awb, **cfg).run()
    cv2.imwrite(root_path / 'outputs' / '05_cfa.png', to_8bit(cfa[..., ::-1], wl))
    ccm = CCM(cfa, **cfg).run()
    cv2.imwrite(root_path / 'outputs' / '06_ccm.png', to_8bit(ccm[..., ::-1], wl))
    gmc = GMC(ccm, **cfg).run()
    cv2.imwrite(root_path / 'outputs' / '07_gmc.png', to_8bit(gmc[..., ::-1], wl))

    # --------------------------------------------------------------
    #  将各阶段结果拼接成一张总览图并弹窗显示
    # --------------------------------------------------------------
    def create_collage(img_paths, cols=3, resize_max=640):
        """将若干 PNG 按列数拼接并返回 collage BGR 图像。"""
        imgs = []
        for p in img_paths:
            im = cv2.imread(str(p))
            if im is None:
                continue
            # 若过大则等比缩小到 resize_max
            h, w = im.shape[:2]
            scale = 1.0
            if max(h, w) > resize_max:
                scale = resize_max / max(h, w)
                im = cv2.resize(im, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            imgs.append(im)
        if not imgs:
            return None

        # 统一高度（以最小高度为准）
        min_h = min(i.shape[0] for i in imgs)
        resized = [cv2.resize(i, (int(i.shape[1]*min_h/i.shape[0]), min_h), interpolation=cv2.INTER_AREA) for i in imgs]

        # 计算行列
        rows = (len(resized) + cols - 1) // cols
        blank = np.zeros_like(resized[0])
        grid = []
        for r in range(rows):
            row_imgs = []
            for c in range(cols):
                idx = r*cols + c
                row_imgs.append(resized[idx] if idx < len(resized) else blank)
            grid.append(cv2.hconcat(row_imgs))
        collage = cv2.vconcat(grid)
        return collage

    step_imgs = [root_path / 'outputs' / f'{i:02d}_{name}.png' for i, name in enumerate([
        'raw','dpc','blc','awb','cfa','ccm','gmc'], start=1)]
    coll = create_collage(step_imgs, cols=3)
    if coll is not None:
        cv2.imwrite(root_path / 'outputs' / '00_summary.png', coll)
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            plt.title('ISP pipeline overview')
            plt.imshow(cv2.cvtColor(coll, cv2.COLOR_BGR2RGB))
            plt.show()
        except Exception as _:
            print('Summary image saved as outputs/00_summary.png')

    print("Successfully finished.")
    
if __name__ == "__main__":
    # run_demo()
    get_module_output()
