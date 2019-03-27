#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.
from sklearn.decomposition import PCA
import numpy as np
import cv2
import matplotlib.pyplot as plt

NUM_COMPONENTS = 40

# Calculates the eigenears based on the input stack of ear images.
def fit(images):
    # Set up the input data vector (images) in a sklearnPCA-friendly format
    baseImages = []
    for image in images:
        baseImages.append(image.rawImage.ravel())
    
    pca = PCA(n_components=NUM_COMPONENTS, whiten=True).fit(baseImages)
        
    
    pca.shape = images[0].rawImage.shape #tack on a shape parameter for later use
    return pca


# Decompose the input image per the already-fit pca basis images (with pca.fit())
def decompose(image, pca):
    unraveled = image.ravel().reshape(1,-1)
    decomposedImage = np.squeeze(pca.transform(unraveled))
    return decomposedImage


# Gets the eigenears for visual debugging
def getEigenbasis(pca):
    eigenbasis = []
    for eigenimage in pca.components_:
        eigenbasis.append(np.reshape(eigenimage, pca.shape))
        
    return eigenbasis
    
# Display the eigenears for visual debugging
def displayEigenbasis(pca):
    eigenbasis = getEigenbasis(pca)
    displayImage = []
    for i,eigenear in enumerate(eigenbasis):
        if i ==0:
            displayImage = eigenear
        else:
            displayImage = np.concatenate((displayImage,eigenear), axis=1)
            
    #plt.imshow(cv2.cvtColor(displayImage, cv2.COLOR_BGR2RGB))
    displayImage -= np.amin(displayImage)
    displayImage /= np.amax(displayImage)
    plt.imshow(displayImage)
    plt.title("Top- " + str(len(eigenbasis)) + " Eigenears")
