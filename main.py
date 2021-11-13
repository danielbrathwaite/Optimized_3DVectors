import math
import random
import pygame as pg
import keyboard
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIN_WIDTH = 800
WIN_HEIGHT = 600

CAMERA_POS = 0, -100, 0
CAMERA_MOVE_SPEED = 0.4
CAMERA_TILT_SPEED = 0.01
CAMERA_FOCAL_DISTANCE = 700

SPHERE_POINTS = 100

BALL_RADIUS = 2

GRAVITY = 0.0008
FRICTION = 0.0005


@ti.data_oriented
class point_system:

    def __init__(self, max_num_rows_and_cols: ti.i32):
        self.max_num_points = max_num_rows_and_cols * max_num_rows_and_cols
        self.points = ti.Matrix.field(4, 2, dtype=ti.f32, shape=(max_num_rows_and_cols, max_num_rows_and_cols))

    @ti.kernel
    def move(self, n_p: ti.i32):
        for i, j in self.points:
            if not i * self.points.m + j + 1 <= n_p:
                self.points[i, j][0, 0] = self.points[i, j][0, 0] + self.points[i, j][1, 0]
                self.points[i, j][0, 1] = self.points[i, j][0, 1] + self.points[i, j][1, 1]

    @ti.kernel
    def add_point(self, n_p: ti.i32):
        if not n_p == self.max_num_points:
            self.points[n_p / self.points.m, n_p % self.points.m] = ti.Matrix([[ti.random(dtype=ti.f32) * WIN_WIDTH, ti.random(dtype=ti.f32) * WIN_HEIGHT], [1, 0, 0], [0, 0, 0], [0, 0, 0]])


if __name__ == '__main__':

    max_num_points = 256
    n_points = 256
    n_r_a_c = round(math.sqrt(max_num_points))
    p_sys = point_system(n_r_a_c)

    print(p_sys.points[0, 0])

    for x in range(n_points):
        p_sys.add_point(x)

    gui = ti.GUI("What is this", (WIN_WIDTH, WIN_HEIGHT), background_color=0x25A6D9)

    running = True
    print(p_sys.points.to_numpy().shape)
    while(running and gui.running):
        if keyboard.is_pressed('Esc'):
            running = False

        p_sys.move(n_points)
        gui.circles(np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]), radius=3)
        gui.show()





