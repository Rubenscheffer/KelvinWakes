# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:41:15 2021

@author: Ruben
"""

import numpy as np

fr_l = [0.130978157, 0.161071127,0.069399966,0.13389721,0.09804128,0.028448597,0.154938731,0.116486056,0.089638839,0.129786584,0.10036129]


def calc(fr_l, n=1, l=0.9):
    # print((2*n - 1)**2 * np.pi**2 *fr_l**4/ l**2)
    a = np.sqrt((2*n - 1)**2 * np.pi**2 *fr_l**4/ l**2 - 1)
    b = 2 * (2*n - 1)**2 * np.pi**2 *fr_l**4/ l**2 - 1
    
    return np.arctan(a/b) * 180 / np.pi


lowest_n = np.zeros(11)


for n in range(179):
    for i, f in enumerate(fr_l):
        z = calc(f, n)
        if not lowest_n[i] > 0:
            lowest_n[i] = z
        