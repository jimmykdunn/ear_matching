#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# All adjustable parameters are in this file. It is imported by all the code
# files using import parameters as p, so that all parameters are 
# prefixed with "p."

# Folder, relative to run location, where imagery is stored
DATA_PATH = "data"

# True to read in images with the ear "donut", false to read in images without it
DONUT = True

# Shrink images by this factor before doing anything. Set to 1 to do no shrinking.
SHRINK_FACTOR = 16

# Set to true to do a PCA decomposition before comparison
DO_PCA = True 

# Number of eigencomponents to use in PCA decomposition
NUM_COMPONENTS = 40

# Number of ears to use (of 195) (use smaller numbers to make faster for debugging)
NUM_TO_READ = 195

# Size of thumbnails for final display (pixels)
THUMBSIZE = (63,84)

# Display images at this size (pixels). (504,672) is 1/6 raw image size
DISPLAY_SHAPE = (504,672) 
