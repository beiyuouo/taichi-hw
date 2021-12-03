#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fazzy_cloth.py 
@Time    :   2021-12-02 10:29:41 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import os
import time
import taichi as ti
import numpy as np

from api import *


@ti.kernel
def paint_cloth(pixels: ti.template(), time: ti.f32):
    for i, j in pixels:
        color = ti.Vector([45.0 / 255.0, 48.0 / 255.0, 71.0 / 255.0])
        x_width = 0.6
        y_width = 0.4
        for color_i in ti.static(range(3)):
            xi, yi = i / pixels.shape[0], j / pixels.shape[1]
            xtemp, y_temp = xi, yi
            yi += ti.cos(xtemp * 3. * y_temp * 2. + time) * .05 * ti.cos(time * .5) * (
                (xtemp + 1.) / 2.) * (color_i + 1) * .3
            xi += ti.cos(y_temp * 3. * xtemp * 2. + time) * .05 * ti.cos(time * .5) * (
                (y_temp + 1.) / 2.) * (color_i + 1) * .3

            xi += gold_noise(ti.Vector([i, j], ti.f64), xi / 10.) * .003
            yi += gold_noise(ti.Vector([i, j], ti.f64), yi / 10.) * .003

            if (xi > 0.5 - x_width / 2 and xi < 0.5 + x_width / 2 and yi > 0.5 - y_width / 2
                    and yi < 0.5 + y_width / 2):
                if distance(ti.Vector([ti.cos(yi * np.pi * 2. * 15.), 0.]), ti.Vector(
                    [1, 0.])) < .1 or distance(ti.Vector([ti.cos(xi * np.pi * 2. * 15.), 0.]),
                                               ti.Vector([1, 0.])) < .1:
                    color[color_i] = 1.0

        pixels[i, j] = color


def main():
    args = Option().parse(['--name', 'Fazzy Cloth', '--show_gui', '--save_video'])
    pixels = ti.Vector.field(3, ti.f64, shape=(args.width, args.height))

    gui = ti.GUI(name=args.name,
                 res=(args.width, args.height),
                 background_color=args.background_color,
                 show_gui=args.show_gui)

    pause = False
    start_time = time.time()
    video_manager = ti.VideoManager(output_dir=args.result_dir,
                                    framerate=args.fps,
                                    automatic_build=False)

    if args.show_gui:
        while gui.running:
            for e in gui.get_events():
                if e.key == gui.ESCAPE:
                    gui.running = False
                elif e.key == gui.SPACE:
                    pause = not pause

            if not pause:
                cur_time = time.time()
                paint_cloth(pixels, cur_time - start_time)
                gui.set_image(pixels)
                gui.show()

    if args.save_video:
        for i in range(int(args.fps * args.duration)):
            paint_cloth(pixels, i / args.fps)
            video_manager.write_frame(pixels.to_numpy())
            print(f'\rFrame {i:03d} is recorded', end='')

        print()
        print(f'Video is saved to {args.result_dir}')
        video_manager.make_video(gif=True, mp4=True)
        print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
        print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')
        os.rename(video_manager.get_output_filename(".mp4"),
                  os.path.join(args.result_dir, args.name + ".mp4"))

        os.rename(video_manager.get_output_filename(".gif"),
                  os.path.join(args.result_dir, args.name + ".gif"))


if __name__ == '__main__':
    main()