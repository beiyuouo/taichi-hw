#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   meta_balls.py 
@Time    :   2021-12-02 17:17:53 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

# Reference: https://www.shadertoy.com/view/XsXGRS
import os
import time
import taichi as ti
import numpy as np

from api import *

CI = ti.Vector([.3, .5, .6])
CO = ti.Vector([0.0745, 0.0862, 0.1058])
CM = ti.Vector([.0, .0, .0])
CE = ti.Vector([.8, .7, .5])


@ti.func
def ball(p: ti.template(), r: ti.f32) -> ti.f32:
    return r / p.dot(p)


@ti.func
def sample(uv: ti.template(), time: ti.f32):
    t0 = ti.sin(time * 1.9) * .46
    t1 = ti.sin(time * 2.4) * .49
    t2 = ti.cos(time * 1.4) * .57

    r = ball(uv + ti.Vector([t0, t2]), .33) * ball(uv - ti.Vector([t0, t1]), .27) * ball(
        uv + ti.Vector([t1, t2]), .59)

    color = ti.select(
        r > .4 and r < .7,
        ti.Vector([step(.1, r * r * r),
                   step(.1, r * r * r),
                   step(.1, r * r * r)]) * CE, ti.select(r < .9, ti.select(r < .7, CO, CM), CI))

    return color


@ti.kernel
def paint(pixels: ti.template(), time: ti.f32):
    for i, j in pixels:
        # print(i, j)
        color = ti.Vector([0., 0., 0.])
        uv = ti.Vector([i / pixels.shape[0] * 2. - 1, j / pixels.shape[1] * 2. - 1])
        # print(uv)
        color = sample(uv, time)
        # print(color)
        color = clamp(color, 0., 1.)
        pixels[i, j] = color


def main():
    args = Option().parse(['--name', 'Meta Ball', '--show_gui', '--save_video'])
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
                paint(pixels, cur_time - start_time)
                gui.set_image(pixels)
                gui.show()

    if args.save_video:
        for i in range(int(args.fps * args.duration)):
            paint(pixels, i / args.fps)
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