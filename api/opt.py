#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   api\opt.py 
@Time    :   2021-12-01 17:35:15 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""
import argparse
import os
import numpy as np
import taichi as ti


class Option(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--seed', type=int, default=0, help='random seed')
        self.parser.add_argument('--device', type=str, default='gpu', help='device')
        self.parser.add_argument('--show_gui', action='store_true', help='show gui')
        self.parser.add_argument('--width', type=int, default=256, help='image width')
        self.parser.add_argument('--height', type=int, default=256, help='image height')
        self.parser.add_argument('--fps', type=int, default=24, help='fps')
        self.parser.add_argument('--background_color',
                                 type=int,
                                 default=0x000000,
                                 help='background color')
        self.parser.add_argument('--result_dir', type=str, default='results', help='result dir')
        self.parser.add_argument('--duration', type=float, default=5.0, help='duration')
        self.parser.add_argument('--save_video', action='store_true', help='save video')

    def get_device(self, device_name):
        device_factory = {
            'cpu': ti.cpu,
            'gpu': ti.gpu,
        }
        if device_name in device_factory:
            return device_factory[device_name]
        else:
            raise ValueError('device name error')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.device = self.get_device(opt.device)

        np.random.seed(opt.seed)
        ti.init(arch=opt.device, default_fp=ti.f64, random_seed=opt.seed)

        if os.path.exists(opt.result_dir) is False:
            os.mkdir(opt.result_dir)

        return opt

    def load_from_file(self):
        pass