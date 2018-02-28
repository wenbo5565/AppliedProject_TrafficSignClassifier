"""
This file is to resize all ppm image files and output them to a pickle with with associated label/class
"""


import pickle
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.image as mpimg

def process_img(filename,new_shape):
    """ 
        - read in an image from ppm format
        - resize it to 32*32
        - return resized image
    """
    # @param new_shape: a triple
    img = mpimg.imread(filename)
    img = cv2.resize(img,new_shape)
    return img

train_img_path=r'E:\Project\Udacity - Computer Vision\Project 2\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images'
img_dict = {'features':[],'labels':[]}
for root,dirs,files in os.walk(train_img_path):
    for name in files: # loop through each files in a subfolder
        label = root[-2:] # extract image class from root path
        if not name.endswith('csv'):
            img_path = os.path.join(root,name) # absolute path for image
            img = process_img(img_path,(32,32)) # resize image
            img_dict['features'].append(img)
            img_dict['labels'].append(label)

""" dump dictionary into a pickle file """
img_picklefilename = r"E:\Project\Udacity - Computer Vision\Project 2\GTSRB_Final_Training_Images\GTSRB\Final_Training\img_train.pickle"
img_pickle = open(img_picklefilename,"wb")
pickle.dump(img_dict,img_pickle)
img_pickle.close()
        

""" load pickle file into python """
img_pickle = open(img_picklefilename,'rb')
img = pickle.load(img_pickle)
features = img['features']
labels = img['labels']

