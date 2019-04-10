#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7

@author: MOTEorg
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from optparse import OptionParser

#Parser of execution options
usage = "Usage: \n\t%prog -i INPUT_IMAGE -d DEBUG"
parser = OptionParser(usage=usage)
parser.add_option("-i", "--input", dest="inFile", default='image.jpg',
                  help="Name of the image file.", metavar="INPUT_IMAGE")
parser.add_option("-d", "--debug", dest="debug", default='0',
                  help="debugging messages", metavar="DEBUG")

(options, args) = parser.parse_args()

# If not input parameter , then set the default
if not options.inFile:
    print("Not input file!\n")

#1. Load image
original_image=cv.imread(options.inFile)
#2. Convert to grayscale
gray_image=cv.cvtColor(original_image,cv.COLOR_BGR2GRAY)
#3. Enhance brightness
gamma=0.7
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
bright_image = cv.LUT(gray_image, lookUpTable)
#4. Trheshold image
threshold=76
ret,thresh_image = cv.threshold(bright_image,threshold,255,cv.THRESH_TOZERO)
#5. Extract Edges
minVal=64
maxVal=108
edges_image = cv.Canny(thresh_image,minVal,maxVal)
#5.1 Edges to black
edges_inv_image=255*np.ones((np.shape(edges_image)))-edges_image
#6. Morph image
kernel_h = np.ones((3,1),np.uint8)
kernel_v = np.ones((1,3),np.uint8)
kernel = np.ones((4,2),np.uint8)
dilation = cv.dilate(edges_image,kernel,iterations = 2)
erosion_h = cv.erode(dilation,kernel_h,iterations = 1)
erosion = cv.erode(dilation,kernel_v,iterations = 1)

#7. Color channel mask 
#b_image=cv.merge([original_image[:,:,0],np.zeros((np.shape(edges_image)),np.uint8),np.zeros((np.shape(edges_image)),np.uint8)])
#g_image=cv.merge([np.zeros((np.shape(edges_image)),np.uint8),original_image[:,:,0],np.zeros((np.shape(edges_image)),np.uint8)])
r_image=cv.merge([np.zeros((np.shape(edges_image)),np.uint8),original_image[:,:,0],np.zeros((np.shape(edges_image)),np.uint8)])

#8. Blur bilateral filters
kernel_size=15
final_image_aux=cv.bilateralFilter(original_image,kernel_size,50,50)
final_image_r=cv.bilateralFilter(r_image,kernel_size,25,25)

#9. Brightness and contrast modification
gamma1=1.05
alpha=1.9
beta=0.8
lookUpTable_add = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable_add[0,i] = np.clip(pow(i*alpha / 255.0, gamma1) * 255.0*beta, 0, 255)

final_image_0=cv.LUT(cv.subtract(final_image_aux[:,:,0],erosion), lookUpTable_add)
final_image_1=cv.LUT(cv.subtract(final_image_r[:,:,1],erosion), lookUpTable_add)
final_image_2=cv.LUT(cv.subtract(final_image_aux[:,:,2],erosion), lookUpTable_add)

#8. Add color channels in a single image and apply a smooth Gaussian filter
final_image=cv.GaussianBlur(cv.merge([final_image_0,final_image_1,final_image_2]),(kernel_size,kernel_size),0)

#TODO: Shadows and magazine style

#Show edges and final filtered image
plt.figure()
plt.subplot(1,2,1)
plt.imshow(erosion,interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.subplot(1,2,2)
plt.imshow(final_image,interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#Save
filename_split=(options.inFile).split('.')
cv.imwrite(filename_split[0]+'_filtered.jpg',final_image)
    
