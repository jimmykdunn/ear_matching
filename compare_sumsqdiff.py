#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.
import numpy as np

# Super basic pixel-by-pixel sum-squared difference. Standin for
# real comparison algorithms.
def compare(image1, image2):
    score = 1.0 / (1.0 + np.mean( \
        (image1/np.mean(image1) - \
         image2/np.mean(image2))**2))
    
    return score