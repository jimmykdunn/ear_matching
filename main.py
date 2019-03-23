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
DONUT = True # true to read in images with the ear "donut", false to read in images without it

# Reads every image
def readImages():
    # Loop over the images, reading them in
    firstSet = [] # all the images without a 't' in their title
    secondSet = [] # all the images with a 't' in their title
    filelist = glob.glob(DATA_PATH + "/*jpg*")
    print("Reading images from file")
    for file in filelist:
        # Skip images that do/don't have the "donut" device per the DONUT parameter
        itype = file.split(os.sep)[-1].split('.')[0].split('_')[-1]
        if DONUT: # skip non-donut images
            if not 'd' in itype:
                continue
        else: # skip donut images
            if 'd' in itype:
                continue
            
        image = earImage.earImage(file) # read and initialize the image
        
        # Preprocess the image
        image.preprocess()
        
        # FOR TESTING ONLY
        #image.displayRawRGB(time=1000) # display image for 1s
        
        # Add image to the first set if it doesn't have a 't' in its name,
        # add to the second set if it does have a 't' in its name.
        if not 't' in image.typeStr:
            firstSet.append(image)
        else:
            secondSet.append(image)
        
        print("Successfully read image: ", image.nameString)
        
        # TO MAKE TESTING QUICKER. REMOVE FOR FINAL RUNS.
        # Stop after reading in the first 40 for testing more quickly
        if len(secondSet) >= 24:
            break
    
    return firstSet, secondSet


# Calculates the similarity between each pair of images
def calculateSimilarity(firstSet, secondSet):
    # Pairwise image similarity measure in an NxN matrix
    similarityMatrix = np.zeros([len(firstSet),len(secondSet)])
        
    # Loop over each image, then see if it matches the rest
    for i1, image in enumerate(firstSet):
        print("Calculating similarities for image ", i1)
        for i2, image2 in enumerate(secondSet):
            score = image.compare(image2)
            similarityMatrix[i1,i2] = score
    
    return similarityMatrix


# Calculates accuracy by determining if the peak pixel of the similarity
# matrix in each row corresponds to the other image of the same truth ear.
def calculateAccuracy(similarityMatrix, firstSet, secondSet):
    peakId = []
    trueId = []
    for i,row in enumerate(similarityMatrix):
        peakVal = np.amax(row)
        peakIndex = np.argmax(row)
        
        # Get the ID of the image being compared and the ID of the most-similar
        # of all the other images
        trueId.append(firstSet[i].number) # id of the image
        peakId.append(secondSet[peakIndex].number) # id of the peak match
        
    # Overall accuracy is the number correct over the total number
    isCorrect = [a == b for a,b in zip(trueId, peakId)]
    accuracy = np.mean(isCorrect)
    
    return accuracy, isCorrect
  

# Displays results in a useful way
def displayResults(accuracy, isCorrect, similarityMatrix, firstSet, secondSet):         
    print("=========================") # dividing line
    if DONUT:
        print("Performance for images WITH donut-device")
    else:
        print("Performance for images \"in the wild\"")
    
    print("ACCURACY: ", accuracy)
    
    simOfBest = []
    for row in similarityMatrix:
        simOfBest.append(np.amax(row))
    print("AVG SIMILARITY SCORE OF BEST MATCH: ", np.mean(simOfBest))
   
    

# Main function
def main():
    
    # Read in all the images
    firstSet, secondSet = readImages()
        
    # Calculate the pairwise similarity between images
    similarityMatrix = calculateSimilarity(firstSet, secondSet)
    
    # Display the similarity matrix as an image
    #cv2.imshow("Similarity matrix", 
    #           cv2.resize(similarityMatrix**2, (400,400),interpolation=cv2.INTER_NEAREST))
    #cv2.waitKey(0)
    #cv2.destroyWindow("Similarity matrix")
    
    # Calculate accuracy using the similarity matrix peak (off-diagonal)
    accuracy, isCorrect = calculateAccuracy(similarityMatrix, firstSet, secondSet)
    
    # Displays results
    displayResults(accuracy, isCorrect, similarityMatrix, firstSet, secondSet)
    
    # Return similarity matrix and images
    return accuracy, similarityMatrix, isCorrect


# Actually execute the program
accuracy, similarityMatrix, isCorrect = main()