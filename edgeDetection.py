#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

import cv2
import numpy as np

def cannyEdges(image):
    edges = cv2.Canny(image, 100,200).astype(np.uint8)
    edges = np.repeat(np.reshape(edges,[edges.shape[0], edges.shape[1], 1]), 3, axis=2)
    return edges
