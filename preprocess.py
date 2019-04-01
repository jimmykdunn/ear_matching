#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.
import numpy as np
import cv2

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.05 #0.15


# Shifts and rotates the image
# This function is based on:
# https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
def align(im1, im2):
 
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
       
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    # CUSTOM CS640: we are working with masked imagery at this point, so the 
    # ORB algorithm absolutely LOVES to pull the edges of the mask itself.
    # This is exactly what we DON'T want, so lets remove all keypoints that
    # are within a pixel on any side of a masked out (exactly zero) pixel!!!
    #keypoints1, descriptors1 = trimKeypoints(keypoints1, descriptors1, im1Gray)
    #keypoints2, descriptors2 = trimKeypoints(keypoints2, descriptors2, im2Gray)
        
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
       
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
     
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    print("Good matches: " + str(numGoodMatches))
    matches = matches[:numGoodMatches]
     
    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
       
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
     
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
       
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
     
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    im1Reg = np.array(im1Reg)
    return im1Reg, h
    

# CUSTOM CS640: we are working with masked imagery at this point, so the 
# ORB algorithm absolutely LOVES to pull the edges of the mask itself.
# This is exactly what we DON'T want, so lets remove all keypoints that
# are within a pixel on any side of a masked out (exactly zero) pixel!!!
def trimKeypoints(keypoints, descriptors, image):
    newkeypoints = []
    newdescriptors = []
    neighborOffsets = [(-1,-1), (-1,0), (-1,1), \
                       ( 0,-1), ( 0,0), ( 0,1), \
                       ( 1,-1), ( 1,0), ( 1,1)]
    
    for keypoint, descriptor in zip(keypoints,descriptors):
        y,x = keypoint.pt
        x = int(x)
        y = int(y)
        neighborVals = [image[x+offset[0], y+offset[1]] for offset in neighborOffsets]
        
        
        # Keep this keypoint if it doesn't neighbor any exactly-zero (i.e. 
        # likely masked) pixels
        if not 0 in neighborVals:
            newkeypoints.append(keypoint)
            newdescriptors.append(descriptor)
    
    return newkeypoints, np.array(newdescriptors)


# Removes background that is not part of the ear
def removeBackground(image):
    
    # Remove donut if it is present
    #image = removeDonut(image)
    #cv2.imshow("DonutGone", image)
    #cv2.waitKey(0)
    #cv2.destroyWindow("DonutGone")
    
    # Match skin color. DOES NOT WORK IF IMG IS ALREADY BLACK AND WHITE!!!
    skinMask = skinDetect(image)
    skinMask = cleanMask(skinMask)
    #cv2.imshow("SkinMask", skinMask*255)
    #cv2.waitKey(0)
    #cv2.destroyWindow("SkinMask")
    
    image = cv2.bitwise_and(image,image,mask = skinMask)
    #cv2.imshow("SkinDetected", image)
    #cv2.waitKey(0)
    #cv2.destroyWindow("SkinDetected")
    
    return image


# skinDetect function adapted from CS640 lab held on March 29, 2019
# Function that detects whether a pixel belongs to the skin based on RGB values
# src - the source color image
# dst - the destination grayscale image where skin pixels are colored white and the rest are colored black
def skinDetect(src):
    # Surveys of skin color modeling and detection techniques:
    # 1. Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
    # 2. Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
                
    # Fast array-based way
    dst = np.zeros((src.shape[0], src.shape[1], 1), dtype = "uint8")
    b = src[:,:,0]
    g = src[:,:,1]
    r = src[:,:,2]
    dst = (r>95) * (g>40) * (b>20) * (r>g) * (r>b) * (abs(r-g) > 15) * \
        ((np.amax(src,axis=2)-np.amin(src,axis=2)) > 15)
        
                
    return dst.astype(np.uint8)


# Cleans up the input mask
def cleanMask(mask):
    nx,ny = mask.shape
    x = np.arange(nx)-nx/2
    y = np.arange(ny)-ny/2
    x = np.repeat(x[:,np.newaxis],ny,axis=1)
    y = np.repeat(y[:,np.newaxis],nx,axis=1).T
    distFromCenter2 = x*x + y*y
    
    # Remove the inner piece of the mask - this usually forces the ear canal
    # entrance to remain unmasked.
    #innerRadiusFrac = 0.5
    #centerMask = distFromCenter2 < ((nx+ny)/4*innerRadiusFrac)**2
    #mask = ((mask + centerMask) > 0).astype(np.uint8)
    
    # Force the outer piece of the mask - the ear is never near the edges
    outerRadiusFrac = 0.9
    outerMask = distFromCenter2 < ((nx+ny)/4*outerRadiusFrac)**2
    outerMask = outerMask.astype(np.uint8)
    mask = cv2.bitwise_and(mask,mask,mask=outerMask)
    
    # Open the mask (erode+dilate)
    kernelsize = (int)(0.1*(nx+ny)/2)
    dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize,kernelsize))
    mask = cv2.dilate(mask, dkernel, iterations=1)
    mask = cv2.erode(mask, dkernel, iterations=1)
    
    
    return mask

# Removes the "donut" around the ear that is present in some of the image
# by color and/or position
def removeDonut(image):
    # Donut color is around (B,G,R) = (230,230,230)
    donutMinBGR = [210,210,210]
    b = image[:,:,0].astype(float)
    g = image[:,:,1].astype(float)
    r = image[:,:,2].astype(float)
    #donutMask = (b > donutMinBGR[0]) * (g > donutMinBGR[1]) * (r > donutMinBGR[2]) * \
    donutMask = (abs(b-r) < 20) * abs((b-g) < 20) * (abs(g-r) < 20)
    
    donutMask = np.invert(donutMask)
    donutMask = donutMask.astype(np.uint8)
    
    
    image = cv2.bitwise_and(image,image,mask = donutMask)
    
    return image