#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# Very simple code to take in the mean edge image and draw a template from it
import cv2
import numpy as np
import parameters as p


# Class for holding mask templates for alignment
class template:
    def __init__(self,base):
        self.mask = base
        self.nx, self.ny, self.nc = self.mask.shape
        self.xs  = -9999
        self.ys  = -9999
        self.rot = -9999
        self.name = ""
    
    # Applies the specified warp
    def applyWarp(self, xs, ys, rot):
        self.xs  = xs
        self.ys  = ys
        self.rot = rot
        self.mask = np.roll(self.mask,int(self.xs),axis=1)
        self.mask = np.roll(self.mask,int(self.ys),axis=0)
        R = cv2.getRotationMatrix2D((self.ny/2, self.nx/2), rot, 1.0)
        self.mask = cv2.warpAffine(self.mask, R, (self.ny, self.nx))
        self.name = "template_"+str(self.xs)+"_"+str(self.ys)+"_"+str(self.rot)
        #cv2.imshow("ear",self.mask)
        #cv2.waitKey(1000)
        
    # Determines how well the input edgemap fits this template
    def checkMatchStrength(self,edgemap):
        # This is a simple matter of counting how many pixels in the input
        # edgemap match the template mask
        pixelmatches = (edgemap * self.mask) > 0
        matchStrength = np.sum(pixelmatches)
        return matchStrength


# Perturb the template in an array of ways
def makeTemplates():
    templateBase = cv2.imread(p.TEMPLATE_IMAGE)
    nx,ny,nc = templateBase.shape
    xshifts   = (np.arange(10)-5)/5 * nx*0.05
    yshifts   = (np.arange(10)-5)/5 * ny*0.05 # 0.15
    rotations = (np.arange(10)-5) * 3 + 3
    
    xshifts = xshifts.astype(int)
    yshifts = yshifts.astype(int)
    
    templates = []
    for xs in xshifts:
        for ys in yshifts:
            for rot in rotations:
                thisTemplate = template(templateBase)
                thisTemplate.applyWarp(xs,ys,rot)
                templates.append(thisTemplate)
                #cv2.imshow(thisTemplate.name, thisTemplate.mask)
                #cv2.waitKey(0)
                #cv2.destroyWindow(thisTemplate.name)
                #cv2.imwrite(thisTemplate.name+".png", thisTemplate.mask)
    
    return templates


# Generate initial template (did some minor editing in MS paint after this)
def makeTemplateFromMeanstack():
    meanstack = cv2.imread("meanStackDonut8x.jpg")
    
    dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    template = cv2.dilate(meanstack, dkernel, iterations=3)
    template = cv2.erode(template, dkernel, iterations=4)
    template = template > 60
    template = template.astype(np.uint8) * 255
    template = cv2.dilate(template, dkernel, iterations=1)
    cv2.imshow("tempate", template)
    cv2.waitKey(0)
    cv2.destroyWindow("template")
    cv2.imwrite("meanStackTemplate8x.png", template)