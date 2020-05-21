import sys
import os
import numpy as np
import json
import copy
import cv2
from numpy import load
import pandas as pd
from tqdm import tqdm
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def get_DWF_image(img):
    # DWT algorithm
    #--------------------#
    img = img.astype(np.float)
    height, width = img.shape[:2]
    result = np.zeros((height, width), np.float)

    # Horizontal processing
    width2 = int(width / 2)
    for i in range(height):
        for j in range(0, width - 1, 2):
            j1 = j + 1
            j2 = int(j / 2)
            result[i, j2] = (img[i, j] + img[i, j1]) / 2
            result[i, width2 + j2] = (img[i, j] - img[i, j1]) / 2

    # copy array
    mid_image = np.copy(result)

    # Vertical processing:
    height2 = int(height / 2)
    for i in range(0, height - 1, 2):
        for j in range(0, width):
            i1 = i + 1
            i2 = int(i / 2)
            result[i2, j] = (mid_image[i, j] + mid_image[i1, j]) / 2
            result[height2 + i2, j] = (mid_image[i, j] - mid_image[i1, j]) / 2
    concat_img = result.astype(np.uint8)
    result_img = concat_img[height2:height,0:width2]
    return result_img

def get_segments(img):
    # Split into  3*3 chunks
    #--------------------#
    segments = []
    size = img.shape[0]
    for i in range(0, size-2, 1):
        for j in range(0, size-2, 1):
            segment = img[j:j+3, i:i+3]
            if segment.shape == (3,3):
                segments.append(segment)
    #print("No. of segments: ",len(segments))

    return segments

def get_lbp_values(segments):
    # Apply Local Binary Pattern on individual 3*3 segments in clockwise manner
    #--------------------#
    lbp_values = []
    positions = [[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0]]
    for segment in segments:
        center = segment[1][1]
        value = []
        #print(segment.shape)
        for position in positions:
            if center <= segment[position[0]][position[1]]:
                value.append("1")
            else:
                value.append("0")
        value_binary_string = "".join(value)
        lbp_values.append(int(value_binary_string, 2))
    #print(lbp_values)
    return lbp_values
