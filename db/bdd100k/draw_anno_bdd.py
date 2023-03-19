#!/usr/bin/env python

import sys
import json
import numpy as np
import pickle
import cv2


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

DARK_GREEN = (115, 181, 34)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK = (180, 105, 255)
CYAN = (255, 128, 0)

CHOCOLATE = (30, 105, 210)
PEACHPUFF = (185, 218, 255)
STATEGRAY = (255, 226, 198)


GT_COLOR = [PINK, CYAN, ORANGE, YELLOW, BLUE]




def draw_bdd_annotation(file_path, x1,y1,x2,y2):
    img = cv2.imread(file_path)
    bottom_point = (x1,y1)
    top_point = (x2,y2)

    img = cv2.circle(img, bottom_point, 5, color=GT_COLOR[2], thickness=-1)
    img = cv2.circle(img, top_point, 5, color=GT_COLOR[0], thickness=-1)

    print("shape = ", img.shape )
    
    cv2.imshow("annotation image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



file_path="/home/fan.wu/Downloads/data_set/bdd100k/bdd100k/labels/lane/colormaps/val/b1c9c847-3bda4659.png"
x1 = 1
y1 = 507
x2 = 461
y2 = 388

draw_bdd_annotation(file_path, x1,y1,x2,y2)

