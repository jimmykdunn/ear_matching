#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import earImage

# Global parameters
DATA_PATH = "data" # folder, relative to run location, where data is stored


# Reads every image
def readImages():
    # Loop over the images, reading them in
    images = [] # all the images
    filelist = glob.glob(DATA_PATH + "/*jpg*")
    print("Reading images from file")
    for file in filelist:
        image = earImage.earImage(file) # read and initialize the image
        
        # Preprocess the image
        image.preprocess()
        
        # FOR TESTING ONLY
        #image.displayRawRGB(time=1000) # display image for 1s
        
        images.append(image) # add the image to the list of all images
        
        print("Successfully read image: ", image.nameString)
        
        # TO MAKE TESTING QUICKER. REMOVE FOR FINAL RUNS.
        # Stop after reading in the first 40 for testing more quickly
        if len(images) >= 16:
            break
    
    return images


# Calculates the similarity between each pair of images
def calculateSimilarity(images):
    # Pairwise image similarity measure in an NxN matrix
    similarityMatrix = np.zeros([len(images),len(images)])
        
    # Loop over each image, then see if it matches the rest
    for i1, image in enumerate(images):
        print("Calculating similarities for image ", i1)
        for i2, image2 in enumerate(images):
            score = image.compare(image2)
            similarityMatrix[i1,i2] = score
    
    return similarityMatrix


# Main function
def main():
    # Read in all the images
    images = readImages()
        
    # Calculate the pairwise similarity between images
    similarityMatrix = calculateSimilarity(images)
    
    # Display the similarity matrix as an image
    cv2.imshow("Similarity matrix", cv2.resize(similarityMatrix, (400,400)))
    cv2.waitKey(0)
    cv2.destroyWindow("Similarity matrix")
    
    
    # Return similarity matrix and images
    return similarityMatrix, images


# Actually execute the program
similarityMatrix, images = main()