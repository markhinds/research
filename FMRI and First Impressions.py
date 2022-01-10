#!/usr/bin/env python
# coding: utf-8

# ## First Impressions Bayesian Deep Learning
# 
# #### How does Laplace Approximation work in determining accuracy of facial impressions?

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
import glob as glob

from PIL import Image


# In[2]:


import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,MaxPooling2D
from keras.layers import Conv2D
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

from sklearn import metrics


# ### Prepping the image data

# In[3]:


#function for splitting array into 2 column array
def split_odd_even(df,col_name1, col_name2):
    even = df.loc[::2]
    even = even.reset_index(drop=True)
    even.columns = [col_name1]
    
    odd = df.loc[1::2]
    odd = odd.reset_index(drop=True)
    odd.columns = [col_name2]
    
    split_df = pd.concat([even,odd], axis = 1)
    
    return split_df


# In[4]:


#Image directory path
img_folder = 'C:\\Users\\hinds\\CSResearch\\Data\\Image'

#Reading images and filenames into array
def create_dataset_PIL(img_folder):
    img_data_array = []
    for file in os.listdir(os.path.join(img_folder)):
        img_data_array.append(file)
        image_path = os.path.join(img_folder,  file)
        image = np.array(Image.open(image_path))
        image = image.astype('float32')
        image /= 255  
        img_data_array.append(image)
        
    img_arr = pd.DataFrame(img_data_array)
    img_arr = img_arr.astype(object)
    #img_arr.columns = ["Image", "FileName"]
    
    return img_arr

#Array of files and images
img_arr = create_dataset_PIL(img_folder)


# In[5]:


#Seperating file and image
img_df = split_odd_even(img_arr, "FileName", "Image")


# In[6]:


#Directory for annotations
annotations_folder = 'C:\\Users\\hinds\\CSResearch\\Data\\Annotation'

#functions for reading annotations to multidimensional array
def get_annotations(path):
    os.chdir(path)
    all_files = glob.glob(path)

    li = []

    for filename in os.listdir()[1:]:
        df = pd.read_csv(filename, index_col=None, header=None)
        li.append(df)
        
    return li

ann_arr = get_annotations(annotations_folder)


# In[7]:


#Function for creating data frame splitting training, testing and validation sets
def make_ann_df(arr):
    #seperate testing, index 0-3
    testing = pd.concat([arr[0],arr[1], arr[2], arr[3]], axis = 1)
    testing.columns = ["TestFile1", "Age","TestFile2", "Dominance","TestFile3", "IQ","TestFile4", "Trustworthiness"]
    
    #seperate training, index 4-7
    training = pd.concat([arr[4],arr[5], arr[6], arr[7]], axis = 1)
    training.columns = ["TrainFile1", "Age","TrainFile2", "Dominance","TrainFile3", "IQ","TrainFile4", "Trustworthiness"]
    
    #seperate validation, index 8-11
    validation = pd.concat([arr[8],arr[9], arr[10], arr[11]], axis = 1)
    validation.columns = ["ValFile1", "Age","ValFile2", "Dominance","ValFile3", "IQ","ValFile4", "Trustworthiness"]
    
    return testing, training, validation

testing,training,validation = make_ann_df(ann_arr)


# In[8]:


#function for matching images with their respective features.
def make_image_ann_df(image,ann,ttv):
    merged_df = pd.merge(image, ann, left_on="FileName", right_on= ttv + "File1", how="inner")
    merged_df = merged_df.drop([ttv+'File1', ttv+'File2',ttv+'File3',ttv+'File4'], axis=1)
    return merged_df


# In[9]:


#Create sets

#test set
test = make_image_ann_df(img_df,testing,"Test").dropna()

#train set
train = make_image_ann_df(img_df,training,"Train").dropna()

#validation set
validate = make_image_ann_df(img_df,validation,"Val").dropna()


# In[10]:


def train_test_val_set(train,test,val,response):
    x_train,y_train = train['Image'].values,train[response].values
    x_test,y_test = test['Image'].values,test[response].values
    x_validate,y_validate = val['Image'].values,val[response].values
    
    return x_train, y_train, x_test, y_test, x_validate, y_validate


# In[11]:


#Set which response value you want to predict: Age, Dominance, IQ, Trustworthiness
response = "Age"
x_train, y_train, x_test, y_test, x_validate, y_validate = train_test_val_set(train,test,validate,response)


# In[12]:


#Viewing images with their response value
fig, axes = plt.subplots(3, 3, figsize=(10,10))

for i in range(3):
    for j in range(3):
        axes[i,j].imshow(x_train[i*4 + j], cmap='gray')
        axes[i,j].set_title(response+": " + str(y_train[i*4+j]))
        axes[i,j].axis('off')

plt.show()


# #### Problem
# I need to convert the numpy array to a tensor.

# ### Pretrain the Model

# In[21]:


np.random.seed(420)
# define the variable input_s here, which is the size of the images in CIFAR10.(width, height, channel) 

input_s = (150,130,1)

#base model
model = keras.Sequential()
#first layer
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_s))
model.add(MaxPooling2D((2,2)))

#second layer
model.add(Conv2D(64, (3, 3), activation='relu'))

#third layer
model.add(Conv2D(128, (3, 3), activation='relu'))

#convert matrix to single array
model.add(Flatten())

#Transform vector 
model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

#add dropout to reduce overfitting
model.add(layers.Dropout(0.2))

#add softmax prediction layer
model.add(Dense(10, activation='softmax'))

#compile
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#set epochs
epo = 10

#fit the model
hist = model.fit(x_train, y_train, epochs=epo, batch_size=64, validation_data=(x_test, y_test), verbose=1)

