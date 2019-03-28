#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

import cv2
def cannyEdges(image, sigma=3):
    edges = cv2.Canny(image, 100,200)
    return edges
