#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue 14 14:50:55 2020

"""

#Performs image to binary file conversion for image import in Lumerical
import numpy as np
import cv2
import os
import glob

def load_images(path):
    loadedImages = []
    # return array of images
    filenames = glob.glob(path)
    filenames.sort()
    for imgdata in filenames:
        # determine whether it is an image.
        if os.path.isfile(os.path.splitext(os.path.join(path, imgdata))[0] + ".png"):
            img_array = cv2.imread(os.path.join(path, imgdata))
            img_array = np.float32(img_array)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            loadedImages.append(img_array)
    return loadedImages

class Binary():
    
    def convert(results_folder):
        # set folder of images
        folder = results_folder  # Set 
        path = folder+'*-bw.png'
        filenames = glob.glob(path)
        filenames.sort()
        imgs = load_images(path) # load images
        imgs = np.asarray(imgs)
        print(np.shape(imgs))
        
        for i in range(len(filenames)):
            basename = os.path.basename(filenames[i])
            # Turn image array into binary array (black to 1, white to 0)
            binary = np.zeros(shape = (np.shape(imgs)[1], np.shape(imgs)[2]), dtype = np.uint8)
            img = imgs[i][:][:]
            print(np.shape(binary))
            print(np.shape(img[:][:][0]))
            binary[img[:][:] <= 50] = 1
            
            print(len(binary[binary==1]))
            # doubling the amount of pixels in both dimensions
            resize_fac = 2
            height, width = imgs.shape[:][:][1:]
            new = np.ones(shape = (resize_fac*height, resize_fac*width), dtype = np.uint8)
            print(np.shape(new))
            print(np.shape(binary))
            new[:binary.shape[0],:binary.shape[1]] = binary[:][:]
            for i in range(height-1,-1,-1):
                for j in range(width-1,-1,-1):
                    cur = new[i][j]
                    new[resize_fac*i:resize_fac*(i+1),resize_fac*j:resize_fac*(j+1)] = [[cur]*resize_fac] * resize_fac
                        
            print(len(new[new==1]))
            print(len(new[new==1])/(resize_fac**2) == len(binary[binary==1]))
            
            
            file1 = open(folder+basename[:-4]+'.txt', 'w')
            header = str(height*resize_fac)+' 1 '+str(height*resize_fac)+' \n'+str(width*resize_fac)+' 1 '+str(width*resize_fac)+' \n2 1 2\n'
            file1.write(header)
            body = new.reshape(new.shape[0]*new.shape[1],1)
            cnt=0
            for item in body:
                file1.write("%s\n" % item[0])
                cnt +=1
            for item in body:
                file1.write("%s\n" % item[0])
            
            file1.close()
