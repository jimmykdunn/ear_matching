#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# External library imports
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Internal imports
import compare_sumsqdiff
import preprocess
import pca
import parameters as p
import edgeDetection

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
        if not p.SHRINK_FACTOR == 1:
            self.rawImage = cv2.resize(self.rawImage, 
                (int(self.rawImage.shape[0]/p.SHRINK_FACTOR),
                int(self.rawImage.shape[1]/p.SHRINK_FACTOR)))
            
        if p.BLACK_AND_WHITE:
            self.rawImage = np.mean(self.rawImage, axis=2).astype(np.uint8)
            self.rawImage = np.repeat(np.reshape(self.rawImage,
                [self.rawImage.shape[0],self.rawImage.shape[1],1]), 3, axis=2)
            
        self.nx, self.ny, self.ncolors = self.rawImage.shape
            
        
        # Processed versions
        #self.rgbImage = cv2.cvtColor(self.rawImage, cv2.COLOR_BGR2RGB)
        self.scaled = []
        self.aligned = []
        self.backgroundRemoved = []
            
        
    def displayRawRGB(self, shape=p.DISPLAY_SHAPE, time=0):
        #plt.imshow(self.rgbImage)
        forDisplay = cv2.resize(self.rawImage, shape)
        cv2.imshow(self.nameString, forDisplay)
        cv2.waitKey(time)
        cv2.destroyWindow(self.nameString)
        
    # Scale the image. May want to input something other than the raw image
    def scale(self):
        return preprocess.scale(self.rawImage)
    
    # Align the image. May want to input something other than the raw image
    def align(self):
        return preprocess.align(self.rawImage)
    
    # Remove (zero out) background. May want to input something other than the raw image
    def removeBackground(self):
        return preprocess.align(self.rawImage)
    
    # Run preprocessing suite on the ear image
    def preprocess(self):
        self.removeBackground()
        self.align()
        self.scale()
        
        
    # Decompose into eigencomponents using already-fit PCA object
    def pcaDecomposition(self, skl_pca):
        self.eigenweights = pca.decompose(self.rawImage, skl_pca)
        
        
    # Run edge detection
    def detectEdges(self):
        self.rawImage = edgeDetection.cannyEdges(self.rawImage)
    
    # Compare this image to another image, get a score for it
    def compare(self, other):
        # This function runs an algorithm to compare the current ear image to
        # another ear image (other).
        
        # Call the external image comparison algorithm.
        # These functions are the real meat of the project.
        # ONLY ONE should be used, the rest should be commented out.
        
        if not p.DO_PCA:
            # Option 1: simple pixel-by-pixel sum squared difference comparison
            score = compare_sumsqdiff.compare(self.rawImage, other.rawImage)
    
        else:        
            # Option 3: PCA/SVD decomposition?
            score = compare_sumsqdiff.compare(self.eigenweights, other.eigenweights)
        
        
        # Option 2: correlation with shifts?
        
        # Option 4: SIFT?
        
        # Option 5: Edge detection?
        
        return score

