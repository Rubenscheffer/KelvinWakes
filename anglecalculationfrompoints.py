# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:27:44 2021

@author: Ruben
"""

import numpy as np

x_ship = [206, 172]
y_ship = [205, 187]
x_wake = [206, 141]
y_wake = [205, 127]


def angle(x, y):
    delta_x = x[1] - x[0]
    delta_y = y[1] - y[0]
    rc = delta_y / delta_x
    ang = np.arctan(rc) * (180 / np.pi)
    return ang


ship_ang = angle(x_ship, y_ship)
wake_ang = angle(x_wake, y_wake)
ang_diff = np.abs(ship_ang - wake_ang)