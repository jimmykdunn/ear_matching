#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Parameters
SHRINK_FACTOR = 4 # shrink images by this factor before doing anything. Set to 1 to do no shrinking.
DISPLAY_SHAPE = (504,672)  # Display images at this size. (504,672) is 1/6 raw image size


# Image type enumeration
FIRST = 0  # no donut, first image (no extension)
SECOND = 1 # no donut, 2nd image ('t' extension)
DONUT1 = 2 # donut, first image ('d' extension)
DONUT2 = 3 # donut, 2nd image ('dt' extension)


# Class for holding an ear image
class earImage:
    def __init__(self, file):
        # Identifying information
        self.nameString = file.split(os.sep)[-1].split('.')[0]
        self.number = int(self.nameString.split('_')[0])
        self.typeStr = self.nameString.split('_')[-1]
        if self.typeStr == '':
            self.type = FIRST
        if self.typeStr == 't':
            self.type = SECOND
        if self.typeStr == 'd':
            self.type = DONUT1
        if self.typeStr == 'dt':
            self.type = DONUT2
        
        # Read the image and shrink if desired
        self.rawImage = cv2.imread(file)
        if not SHRINK_FACTOR == 1:
            self.rawImage = cv2.resize(self.rawImage, 
                (int(self.rawImage.shape[0]/SHRINK_FACTOR),
                int(self.rawImage.shape[1]/SHRINK_FACTOR)))
        self.nx, self.ny, self.ncolors = self.rawImage.shape
        
        # Processed versions
        #self.rgbImage = cv2.cvtColor(self.rawImage, cv2.COLOR_BGR2RGB)
        self.scaled = []
        self.aligned = []
        self.backgroundRemoved = []
        
    def displayRawRGB(self, shape=DISPLAY_SHAPE, time=0):
        #plt.imshow(self.rgbImage)
        forDisplay = cv2.resize(self.rawImage, shape)
        cv2.imshow(self.nameString, forDisplay)
        cv2.waitKey(time)
        cv2.destroyWindow(self.nameString)
        
    def scale(self):
        pass
    
    def align(self):
        pass
    
    def removeBackground(self):
        pass
    
    # Run preprocessing on the ear image
    def preprocess(self):
        self.removeBackground()
        self.align()
        self.scale()
    
    # Compare this image to another image, get a score for it
    def compare(self, other):
        # This function runs a (yet to be determined) algorithm to compare
        # the current image to another image (other)
        
        # Super basic pixel-by-pixel sum-squared difference. Standin for
        # real algorithms.
        score = 1.0 / (1.0 + np.mean( \
                (self.rawImage/np.mean(self.rawImage) - \
                 other.rawImage/np.mean(other.rawImage))**2))
        return score

