import taichi as ti
import numpy as np

PHI = 1.61803398874989484820459


@ti.func
def fract(x: ti.f32) -> ti.f32:
    return x - ti.floor(x)


@ti.func
def distance(x: ti.template(), y: ti.template()) -> ti.f64:
    z = x - y
    z = z * z
    return ti.sqrt(z.sum())


@ti.func
def gold_noise(xy: ti.template(), seed: ti.f64):
    return fract(ti.tan(distance(xy * PHI, xy) * seed) * xy[0])


@ti.func
def clamp(x: ti.template(), min: ti.f32, max: ti.f32) -> ti.f32:
    for i in ti.static(range(len(x))):
        x[i] = ti.max(min, ti.min(max, x[i]))
    return x


@ti.func
def step(x: ti.template(), y: ti.template()) -> ti.f32:
    return ti.select(x > y, 1.0, 0.0)