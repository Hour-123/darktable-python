#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/9/11
# @Description: ISP Pipeline


import numpy as np
from path import Path
import os
import yaml
import importlib
import sys
from model.fir import FIR

class ISP_Pipeline:
    """
    this is a class for ISP Pipeline

    step:
        1. get the ISP Pipeline from yaml
        2. run the ISP Pipeline
        3. get the ISP Pipeline output

    usage:
        isp = ISP_Pipeline(config_path)
    """
    def __init__(self, config_path: str | None = None) -> None:
        super().__init__()
        self.config_path = config_path
        self.root_path = Path(os.path.abspath(__file__)).parent
        self.cfg = self.__from_yaml(self.config_path)
        self.pipe = self.__get_isp_pipeline()

    def run(self) -> np.ndarray:
        return self.__run_isp_pipeline()
    
    
    def __from_yaml(self, yaml_path):
        """ Instantiation from a yaml file. """
        if not isinstance(yaml_path, str):
            raise TypeError(
                f'expected a path string but given a {type(yaml_path)}'
            )
        with open(yaml_path, 'r', encoding='utf-8') as fp:
            yml = yaml.safe_load(fp)
        return yml


    def __get_isp_pipeline(self) -> list:
        """
        get ISP Pipeline
        """
        enable_pipeline = self.cfg['enable'].items()
        module = [k for k, v in enable_pipeline if v is True]
        pipe = []
        for m in module:
            py = importlib.import_module(f'model.{m.lower()}')
            cla = getattr(py, m)
            pipe.append(cla)
        return pipe
    
    
    def __run_isp_pipeline(self) -> np.ndarray:
        """
        run ISP Pipeline
        """
        inp = FIR(**self.cfg).run()
        for p in self.pipe:
            inp = p(inp, **self.cfg).run()
        self.__save_isp_pipeline_outputs(inp)
        return inp
    
    
    def __save_isp_pipeline_outputs(self, output: np.ndarray) -> None:
        """
        save ISP Pipeline outputs
        """
        import cv2
        image_id = self.cfg['RAW_img_path'].split('/')[-1].split('.')[0]

        output_path = self.root_path / 'outputs' / f'{image_id}.png'
        cv2.imwrite(output_path, output[..., ::-1])
        print('ISP Pipeline outputs saved successfully and the path is: ', output_path)
    



