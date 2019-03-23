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
        if len(secondSet) >= 16:
            break
    
    return firstSet, secondSet


# Calculates the similarity between each pair of images and places into a matrix.
def calculateSimilarity(firstSet, secondSet):
    # Pairwise image similarity measure in an NxN matrix
    similarityMatrix = np.zeros([len(firstSet),len(secondSet)])
        
    # Loop over each image, then see if it matches the rest
    for i1, image in enumerate(firstSet):
        print("Calculating similarities for image ", i1)
        for i2, image2 in enumerate(secondSet):
            score = image.compare(image2) # function that actually does the comparison
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
    
    return accuracy, isCorrect, peakId
  
    
# Calculate rank of the true match ear images
def calculateRankOfTrueMatch(similarityMatrix, firstSet, secondSet):
    rankOfTruth = []
    for i,row in enumerate(similarityMatrix):
        trueId = firstSet[i].number
        compIds = []
        for j,item in enumerate(row):
            compIds.append(secondSet[j].number)
            
        sortedStuff = sorted((e,i) for i,e in zip(compIds,row))
        sortedStuff.reverse()
        
        # Rank of the true match is the index of sortedstuff where the compID,
        # i.e. the 2nd entry of sortedstuff, is equal to the trueID
        for i,sp in enumerate(sortedStuff):
            if sp[1] == trueId:
                rankOfTruth.append(i+1) # add 1 so that 1 is "perfect"
        
    return rankOfTruth


# Put a colored border onto the image (BGR color order assumed)
def giveBorder(image, color, npix=3):
    c3 = [0,0,0]
    if color == 'red':
        c3 = [0,0,255]
    elif color == 'green':
        c3 = [0,255,0]
    else:
        print("Inavlid color: ", color)
        return image
    
    # Add the colored border
    image[0:npix,:,0] = c3[0]
    image[0:npix,:,1] = c3[1]
    image[0:npix,:,2] = c3[2]
    image[:,0:npix,0] = c3[0]
    image[:,0:npix,1] = c3[1]
    image[:,0:npix,2] = c3[2]
    image[-npix:,:,0] = c3[0]
    image[-npix:,:,1] = c3[1]
    image[-npix:,:,2] = c3[2]
    image[:,-npix:,0] = c3[0]
    image[:,-npix:,1] = c3[1]
    image[:,-npix:,2] = c3[2]
    
    return image

# Displays results in a useful way
def displayResults(accuracy, isCorrect, similarityMatrix, 
                   firstSet, secondSet, rankOfTruth, peakIds):         
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
   
    # Rank of true match is a way of seeing how well the "true-match" ear
    # did in comparison to the others. If the true-match ear was correctly
    # chosen, then the rank of the true match is 1. If the true-match ear was
    # the 2nd best match among all the images, then the rank of the true match
    # is 2, etc...
    print("AVG RANK OF TRUE MATCH: ", np.mean(rankOfTruth))
    
    # Display an strip of thumbnails with each ear in the first set next
    # to the ear that the algorithm calculated as the best match.
    thumbstrip = []
    i = 0
    for image1,peakId in zip(firstSet,peakIds):
        thumb1 = cv2.resize(image1.rawImage, (int(image1.nx/10), int(image1.ny/10)))
        for image2 in secondSet:
            if image2.number == peakId:
                thumb2 = cv2.resize(image2.rawImage, (int(image1.nx/10), int(image1.ny/10)))

        thumbpair = np.concatenate((thumb1,thumb2), axis=0)
        
        # Green box if correct, red box if incorrect
        if isCorrect[i]:
            thumbpair = giveBorder(thumbpair, 'green')
        else:
            thumbpair = giveBorder(thumbpair, 'red')
            
        # Tack on the thumbpair to the final display image
        if i ==0:
            thumbstrip = thumbpair
        else:
            thumbstrip = np.concatenate((thumbstrip,thumbpair), axis=1)
        i += 1
    plt.imshow(cv2.cvtColor(thumbstrip, cv2.COLOR_BGR2RGB), extent=[0.5,len(peakIds)+0.5,2,0])
    plt.title("First set images (top) with their best matches in 2nd set (bottom)")

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
    accuracy, isCorrect, peakId = \
        calculateAccuracy(similarityMatrix, firstSet, secondSet)
    
    # Calculate rank of the true matched ears
    rankOfTruth = calculateRankOfTrueMatch(similarityMatrix, firstSet, secondSet)
    
    # Displays results
    displayResults(accuracy, isCorrect, similarityMatrix, 
                   firstSet, secondSet, rankOfTruth, peakId)
    
    # Return similarity matrix and images
    return accuracy, similarityMatrix, isCorrect, rankOfTruth


# Actually execute the program
accuracy, similarityMatrix, isCorrect, rankOfTruth = main()