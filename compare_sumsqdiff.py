#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.
import numpy as np
import cv2

# Super basic pixel-by-pixel sum-squared difference. Standin for
# real comparison algorithms.
def compare(image1, image2):
    
    # Disallow any pixels that are exactly zero in either image from
    # contributing to the score
    zeropixels1 = np.sum(image1,axis=2) == 0
    zeropixels2 = np.sum(image2,axis=2) == 0
    zeropixels = zeropixels1*zeropixels2
    zeropixels = zeropixels == False
    zeropixels = zeropixels.astype(np.uint8)
    image1c = cv2.bitwise_and(image1, image1, mask=zeropixels)
    image2c = cv2.bitwise_and(image2, image2, mask=zeropixels)
    
    
    #cv2.imshow("nozeros", image1c)
    #cv2.waitKey(0)
    #cv2.destroyWindow("nozeros")
    
    score = 1.0 / (1.0 + np.mean( \
        (image1c/np.mean(image1c) - \
         image2c/np.mean(image2c))**2))
    
    return score