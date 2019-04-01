#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# External imports
import cv2
import numpy as np

# Just wraps the cv2 Canny edge detection routine
def cannyEdges(image):
    #edges = cv2.Canny(image, 100,200).astype(np.uint8)
    edges = cv2.Canny(image, 50,250).astype(np.uint8)
    edges = np.repeat(np.reshape(edges,[edges.shape[0], edges.shape[1], 1]), 3, axis=2)
    
    return edges
