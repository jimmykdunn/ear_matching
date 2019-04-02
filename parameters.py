#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# All adjustable parameters are in this file. It is imported by all the code
# files using import parameters as p, so that all parameters are 
# prefixed with "p."

# Folder, relative to run location, where imagery is stored
DATA_PATH = "data"

# True to read in images with the ear "donut", false to read in images without it
DONUT = False

# Shrink images by this factor before doing anything. Set to 1 to do no shrinking.
SHRINK_FACTOR = 8

# True to convert to black and white (required for edge detection)
BLACK_AND_WHITE = False

# Run background removal algorithm (True) or not (False)
REMOVE_BACKGROUND = True;

# Set to true to do a PCA decomposition before comparison
DO_PCA = False 

# Use keypoints from file for registration (True) or not (False)
USE_KEYPOINT_FILE = True

# csv file where manual keypoints are located
KEYPOINT_FILE = "myKeypoints.csv"

# Do automatic template alignment (True) or not (False) (different from keypoints)
DO_TEMPLATE_ALIGN = False

# Template image. This is a binary mask that is used for image alignment.
TEMPLATE_IMAGE = "customMeanStackTemplate8x.png"

# Do edge detection (True) or not (False)
DO_EDGE_DETECTION = False

# Edge detection dilation radius as a fraction of the image (width+height)/2
EDGE_DILATION_RADIUS = 0.05

# Number of eigencomponents to use in PCA decomposition
NUM_COMPONENTS = 30

# Number of ears to use (of 195) (use smaller numbers to make faster for debugging)
NUM_TO_READ = 195

# Size of thumbnails for final display (pixels)
THUMBSIZE = (63,84)

# Display images at this size (pixels). (504,672) is 1/6 raw image size
DISPLAY_SHAPE = (504,672) 
