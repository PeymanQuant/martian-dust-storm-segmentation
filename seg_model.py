# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 17:14:15 2022

@author: dat
"""

# https://youtu.be/F365vQ8EndQ
"""
Author: Sreenivas Bhattiprolu
Multiclass semantic segmentation using U-Net with VGG, ResNet, and Inception 
as backbones
Segmentation models: https://github.com/qubvel/segmentation_models
To annotate images and generate labels, you can use APEER (for free):
www.apeer.com 
"""

import tensorflow as tf
import segmentation_models as sm
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import keras 
from keras.metrics import MeanIoU
from PIL import Image
from numpy import asarray
from numpy import save
from numpy import load
import pandas as pd
from io import BytesIO
from patchify import patchify, unpatchify

os.chdir("D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/")

np.random.seed(0)
tf.random.set_seed(0)


#Resizing images, if needed
#SIZE_X = 1024 
#SIZE_Y = 1024
n_classes=2 #Number of classes for segmentation


image_directory = 'D:/peyman/Mars/207/dust_1024_patch_size/img_patches/'
mask_directory = 'D:/peyman/Mars/207/dust_1024_patch_size/mask_patches/'

SIZE = 1024
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    #print(image_directory+image_name)
    image = cv2.imread(image_directory+image_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    #image = image.resize((SIZE, SIZE))
    image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    image = cv2.imread(mask_directory+image_name, 0)
    image = Image.fromarray(image)
    #image = image.resize((SIZE, SIZE))
    mask_dataset.append(np.array(image))
    

#Normalize images
#image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),4)
#D not normalize masks, just rescale to 0 to 1.
#mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.


image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)
#mask_dataset = np.array(mask_dataset) /255.

train_images = image_dataset
train_masks = mask_dataset


#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, train_images.shape[0])
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(train_images[image_number])
plt.subplot(122)
plt.imshow(train_masks[image_number], cmap='gray')
plt.show()



###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#################################################
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.2, random_state = 0)


#Further split training data t a smaller subset for quick testing of models
#X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.5, random_state = 0)


X_train, X_do_not_use, y_train, y_do_not_use = X1, X_test, y1, y_test

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background

image_number = 10
plt.figure(figsize=(12, 6))
plt.imshow(y_test[image_number], cmap='gray')
plt.show()



from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))


######################################################
#Reused parameters in all models

n_classes=2
activation='sigmoid'

LR = 0.0001
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


########################################################################
###Model 1
BACKBONE1 = 'inceptionv3'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)


# save to csv file
save('X_train1.npy', X_train1)
save('X_test1.npy', X_test1)
save('y_train_cat.npy', y_train_cat)
save('y_test_cat.npy', y_test_cat)



X_train1 = load('X_train1.npy')
X_test1 = load('X_test1.npy')
y_train_cat = load('y_train_cat.npy')
y_test_cat = load('y_test_cat.npy')

# define model
model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model1.compile(optim, loss='binary_crossentropy', metrics=metrics)

print(model1.summary())

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='E:/peyman/Mars/207/7 models no black - binaryCrossEntropy/linknet_inceptionv3_100epochs_G05_1024patchsize.h5',
    monitor='val_iou_score',
    mode='max',
    save_best_only=True, verbose=1)

history1=model1.fit(X_train1, 
          y_train_cat,
          batch_size=1, 
          epochs=100,
          verbose=1,
          validation_data=(X_test1, y_test_cat),
          callbacks=[model_checkpoint_callback])


#model1.save('res34_backbone_500epochs_dust_G05_extra.hdf5')
############################################################
#plot the training and validation accuracy and loss at each epoch
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss linknet_inceptionv3_100epochs_G05_1024patchsize')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

acc = history1.history['iou_score']
val_acc = history1.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU linknet_inceptionv3_100epochs_G05_1024patchsize')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


import csv


rows = zip(loss, val_loss, acc, val_acc)
with open('D:/peyman/Mars/207/linknet_inceptionv3.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Training loss', 'Validation loss', 'Training IOU', 'Validation IOU'])
    for row in rows:
        writer.writerow(row)
    

#####################################################

from keras.models import load_model

#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/models/unet_resnet34_100epochs_G05_1024patchsize.h5', compile=False)
model2 = load_model('D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/models/unet_inceptionv3_100epochs_G05_1024patchsize.h5', compile=False)
model3 = load_model('D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/models/unet_vgg16_100epochs_G05_1024patchsize.h5', compile=False)
model4 = load_model('D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/models/linknet_resnet34_100epochs_G05_1024patchsize.h5', compile=False)
model5 = load_model('D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/models/linknet_inceptionv3_100epochs_G05_1024patchsize.h5', compile=False)
model6 = load_model('D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/models/fpn_resnet18_100epochs_G05_1024patchsize.h5', compile=False)
model7 = load_model('D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/models/fpn_resnet34_100epochs_G05_1024patchsize.h5', compile=False)


##############################################################

BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

BACKBONE3 = 'vgg16'
preprocess_input3 = sm.get_preprocessing(BACKBONE3)

BACKBONE4 = 'resnet34'
preprocess_input4 = sm.get_preprocessing(BACKBONE4)

BACKBONE5 = 'inceptionv3'
preprocess_input5 = sm.get_preprocessing(BACKBONE5)

BACKBONE6 = 'resnet18'
preprocess_input6 = sm.get_preprocessing(BACKBONE6)

BACKBONE7 = 'resnet34'
preprocess_input7 = sm.get_preprocessing(BACKBONE7)

#Test some random images
import random
#test_img_number = random.randint(0, len(X_test2))
test_img_number = 37


test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)

test_img_input1 = preprocess_input1(test_img_input)
test_pred1 = model1.predict(test_img_input1)
test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]

test_img_input2 = preprocess_input2(test_img_input)
test_pred2 = model2.predict(test_img_input2)
test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]

test_img_input3 = preprocess_input3(test_img_input)
test_pred3 = model3.predict(test_img_input3)
test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

test_img_input4 = preprocess_input4(test_img_input)
test_pred4 = model4.predict(test_img_input4)
test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]

test_img_input5 = preprocess_input5(test_img_input)
test_pred5 = model5.predict(test_img_input5)
test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]

test_img_input6 = preprocess_input6(test_img_input)
test_pred6 = model6.predict(test_img_input6)
test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]

test_img_input7 = preprocess_input7(test_img_input)
test_pred7 = model7.predict(test_img_input7)
test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]


p = X_test[test_img_number,:,:,:].squeeze()

plt.figure(figsize=(22, 5))
plt.subplot(1, 9, 1)
plt.title('Testing Image')
plt.imshow(p)
plt.subplot(1, 9, 2)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 3)
plt.title('unet_resnet34')
plt.imshow(test_prediction1)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 4)
plt.title('unet_inceptionv3')
plt.imshow(test_prediction2)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 5)
plt.title('unet_vgg16')
plt.imshow(test_prediction3)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 6)
plt.title('linknet_resnet34')
plt.imshow(test_prediction4)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 7)
plt.title('linknet_inceptionv3')
plt.imshow(test_prediction5)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 8)
plt.title('fpn_resnet18')
plt.imshow(test_prediction6)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 9)
plt.title('fpn_resnet34')
plt.imshow(test_prediction7)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.show()



day = 30

i = 0
j = 1

#test_img = X_test1[test_img_number]
test_img = cv2.imread("D:/peyman/Mars/207/dust_1024_patch_size/test_img/image_G05_day" 
                      + str(day) + "_" + str(i) + str(j) + ".png")
test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)

#ground_truth=y_test[test_img_number]
ground_truth=cv2.imread("D:/peyman/Mars/207/dust_1024_patch_size/test_mask/image_G05_day" 
                      + str(day) + "_" + str(i) + str(j) + ".png", 0)

test_img=np.expand_dims(test_img, 0)


test_img_input1 = preprocess_input1(test_img)
test_pred1 = model1.predict(test_img_input1)
test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]

test_img_input2 = preprocess_input2(test_img)
test_pred2 = model2.predict(test_img_input2)
test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]

test_img_input3 = preprocess_input3(test_img)
test_pred3 = model3.predict(test_img_input3)
test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

test_img_input4 = preprocess_input4(test_img)
test_pred4 = model4.predict(test_img_input4)
test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]

test_img_input5 = preprocess_input5(test_img)
test_pred5 = model5.predict(test_img_input5)
test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]

test_img_input6 = preprocess_input6(test_img)
test_pred6 = model6.predict(test_img_input6)
test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]

test_img_input7 = preprocess_input7(test_img)
test_pred7 = model7.predict(test_img_input7)
test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]


#prediction1 = (model1.predict(test_img_input1)[0,:,:,0] > 0.5).astype(np.uint8)
# prediction2 = (model2.predict(test_img_input2)[0,:,:,0] > 0.5).astype(np.uint8)
# prediction3 = (model3.predict(test_img_input3)[0,:,:,0] > 0.5).astype(np.uint8)
# prediction4 = (model4.predict(test_img_input4)[0,:,:,0] > 0.5).astype(np.uint8)
# prediction5 = (model5.predict(test_img_input5)[0,:,:,0] > 0.5).astype(np.uint8)
# prediction6 = (model6.predict(test_img_input6)[0,:,:,0] > 0.5).astype(np.uint8)
# prediction7 = (model7.predict(test_img_input7)[0,:,:,0] > 0.5).astype(np.uint8)


plt.figure(figsize=(22, 5))
plt.subplot(1, 9, 1)
plt.title('Testing Image')
plt.imshow(test_img.squeeze())
plt.subplot(1, 9, 2)
plt.title('Testing Label')
plt.imshow(ground_truth, cmap='gray')
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 3)
plt.title('unet_resnet34')
plt.imshow(test_prediction1)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 4)
plt.title('unet_inceptionv3')
plt.imshow(test_prediction2)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 5)
plt.title('unet_vgg16')
plt.imshow(test_prediction3)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 6)
plt.title('linknet_resnet34')
plt.imshow(test_prediction4)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 7)
plt.title('linknet_inceptionv3')
plt.imshow(test_prediction5)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 8)
plt.title('fpn_resnet18')
plt.imshow(test_prediction6)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(1, 9, 9)
plt.title('fpn_resnet34')
plt.imshow(test_prediction7)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.show()


#plt.imsave("D:/peyman/207/dust_1024_patch_size/test_mask/image_G05_day" + str(day) + "_" + str(i) + str(j) + "_500epochs_pred" + ".png", test_prediction1)




test_image_path = 'D:/peyman/Mars/207/dust_1024_patch_size/test_img/'
test_mask_path = 'D:/peyman/Mars/207/dust_1024_patch_size/test_mask/'

test_image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
test_mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(test_image_path)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    #print(image_directory+image_name)
    image = cv2.imread(test_image_path+image_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    #image = image.resize((SIZE, SIZE))
    test_image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(test_mask_path)
for i, image_name in enumerate(masks):
    image = cv2.imread(test_mask_path+image_name, 0)
    image = Image.fromarray(image)
    #image = image.resize((SIZE, SIZE))
    test_mask_dataset.append(np.array(image))
    

#Normalize images
#image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),4)
#D not normalize masks, just rescale to 0 to 1.
#mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.


test_image_dataset = np.array(test_image_dataset)
test_mask_dataset = np.array(test_mask_dataset)
#mask_dataset = np.array(mask_dataset) /255.

def calculate_iou(pred_mask, gt_mask):
    if gt_mask is None:
        return 0.0

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if (union != 0):
        iou = intersection / union
    else:
        iou = 1

    return iou


# create a list of models and preprocessing functions
models = [model1, model2, model3, model4, model5, model6, model7]
preprocess_functions = [preprocess_input1, preprocess_input2, preprocess_input3, preprocess_input4, preprocess_input5, preprocess_input6, preprocess_input7]

# create an empty dataframe to store the IOU values
iou_df = pd.DataFrame()

# loop through each model and test image, and add a new column to the dataframe for each model's IOU values
for model_idx, model in enumerate(models):
    iou_values = []
    for i in range(len(test_image_dataset)):
        # Use the model to generate a segmentation mask for the test image
        test_img = test_image_dataset[i]
        #ground_truth=y_test[test_img_number]
        test_img_input = np.expand_dims(test_img, 0)
        
        preprocess_input = preprocess_functions[model_idx]
        test_img_input = preprocess_input(test_img_input)
        
        test_pred = model.predict(test_img_input)
        test_prediction = np.argmax(test_pred, axis=3)[0,:,:]
        
        
        gt_mask = test_mask_dataset[i]
        gt_mask = gt_mask/255.
        
        # Calculate the IOU for this test image
        print("shape of test_prediction = ", test_prediction.shape)
        print("\nshape of gt_mask = ", gt_mask.shape)
        iou = calculate_iou(test_prediction, gt_mask)
        iou_values.append(iou)
    
    # add a new column to the dataframe for this model's IOU values
    iou_df[f'model{model_idx+1}'] = iou_values
  
    
# create a new dataframe to store the average IOU values for each model
avg_iou_df = pd.DataFrame(columns=[f'model{i}' for i in range(1, 8)])

# compute the average IOU values for each model and add them to the new dataframe
for model_idx, model in enumerate(models):
    avg_iou = sum(iou_df[f'model{model_idx+1}']) / len(iou_df[f'model{model_idx+1}'])
    avg_iou_df.loc[0, f'model{model_idx+1}'] = avg_iou
    
    
# Convert the dataframe to a CSV file in memory
csv_file = BytesIO()
# Save the dataframe to a CSV file on your computer
iou_df.to_csv('D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/iou_values_for_test_images.csv', index=False)
avg_iou_df.to_csv('D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/avg_iou_values_for_test_images.csv', index=False)


test_labels = ["25_03", "25_10", "25_11", "25_12", "25_13", "26_00", "26_01", 
               "26_02", "26_03", "26_11", "26_12", "27_00", "27_01", "27_02", 
               "27_03", "27_10", "27_11", "28_00", "28_01", "28_02", "28_03", 
               "28_10", "28_11", "29_00", "29_01", "29_02", "29_03", "29_10", 
               "30_00", "30_01"]


cnt = 0

for test_lbl in test_labels:
    test_img = cv2.imread("D:/peyman/Mars/207/dust_1024_patch_size/test_img/image_G05_day" 
                          + test_lbl + ".png")
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)

    #ground_truth=y_test[test_img_number]
    ground_truth=cv2.imread("D:/peyman/Mars/207/dust_1024_patch_size/test_mask/image_G05_day" 
                          + test_lbl + ".png", 0)
    

    test_img=np.expand_dims(test_img, 0)


    test_img_input1 = preprocess_input1(test_img)
    test_pred1 = model1.predict(test_img_input1)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]

    test_img_input2 = preprocess_input2(test_img)
    test_pred2 = model2.predict(test_img_input2)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]

    test_img_input3 = preprocess_input3(test_img)
    test_pred3 = model3.predict(test_img_input3)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

    test_img_input4 = preprocess_input4(test_img)
    test_pred4 = model4.predict(test_img_input4)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]

    test_img_input5 = preprocess_input5(test_img)
    test_pred5 = model5.predict(test_img_input5)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]

    test_img_input6 = preprocess_input6(test_img)
    test_pred6 = model6.predict(test_img_input6)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]
    
    test_img_input7 = preprocess_input7(test_img)
    test_pred7 = model7.predict(test_img_input7)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]



    #prediction1 = (model1.predict(test_img_input1)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction2 = (model2.predict(test_img_input2)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction3 = (model3.predict(test_img_input3)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction4 = (model4.predict(test_img_input4)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction5 = (model5.predict(test_img_input5)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction6 = (model6.predict(test_img_input6)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction7 = (model7.predict(test_img_input7)[0,:,:,0] > 0.5).astype(np.uint8)



    plt.figure(figsize=(22, 5))
    plt.subplot(1, 9, 1)
    plt.title('Testing Image')
    plt.imshow(test_img.squeeze())
    plt.subplot(1, 9, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth, cmap='gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 3)
    plt.title('unet_resnet34')
    plt.imshow(test_prediction1)
    plt.text(0.5, -0.1, str(round(iou_df["model1"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 4)
    plt.title('unet_inceptionv3')
    plt.imshow(test_prediction2)
    plt.text(0.5, -0.1, str(round(iou_df["model2"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 5)
    plt.title('unet_vgg16')
    plt.imshow(test_prediction3)
    plt.text(0.5, -0.1, str(round(iou_df["model3"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 6)
    plt.title('linknet_resnet34')
    plt.imshow(test_prediction4)
    plt.text(0.5, -0.1, str(round(iou_df["model4"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 7)
    plt.title('linknet_inceptionv3')
    plt.imshow(test_prediction5)
    plt.text(0.5, -0.1, str(round(iou_df["model5"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 8)
    plt.title('fpn_resnet18')
    plt.imshow(test_prediction6)
    plt.text(0.5, -0.1, str(round(iou_df["model6"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 9)
    plt.title('fpn_resnet34')
    plt.imshow(test_prediction7)
    plt.text(0.5, -0.1, str(round(iou_df["model7"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.savefig("D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/prediction of test images with IOU/" + test_lbl + ".png")
    plt.show()
    cnt += 1
    
    

for test_img_number in range(len(X_test)):
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    gt_mask = ground_truth
    gt_mask = ground_truth/255.
    gt_mask = gt_mask.squeeze()
    test_img_input=np.expand_dims(test_img, 0)

    test_img_input1 = preprocess_input1(test_img_input)
    test_pred1 = model1.predict(test_img_input1)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]
    iou1 = calculate_iou(test_prediction1, gt_mask)

    test_img_input2 = preprocess_input2(test_img_input)
    test_pred2 = model2.predict(test_img_input2)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]
    iou2 = calculate_iou(test_prediction2, gt_mask)

    test_img_input3 = preprocess_input3(test_img_input)
    test_pred3 = model3.predict(test_img_input3)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]
    iou3 = calculate_iou(test_prediction3, gt_mask)

    test_img_input4 = preprocess_input4(test_img_input)
    test_pred4 = model4.predict(test_img_input4)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]
    iou4 = calculate_iou(test_prediction4, gt_mask)

    test_img_input5 = preprocess_input5(test_img_input)
    test_pred5 = model5.predict(test_img_input5)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]
    iou5 = calculate_iou(test_prediction5, gt_mask)

    test_img_input6 = preprocess_input6(test_img_input)
    test_pred6 = model6.predict(test_img_input6)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]
    iou6 = calculate_iou(test_prediction6, gt_mask)

    test_img_input7 = preprocess_input7(test_img_input)
    test_pred7 = model7.predict(test_img_input7)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]
    iou7 = calculate_iou(test_prediction7, gt_mask)

    p = X_test[test_img_number,:,:,:].squeeze()

    plt.figure(figsize=(22, 5))
    plt.subplot(1, 9, 1)
    plt.title('Testing Image')
    plt.imshow(p)
    plt.subplot(1, 9, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 3)
    plt.title('unet_resnet34')
    plt.imshow(test_prediction1)
    plt.text(0.5, -0.1, str(round(iou1, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 4)
    plt.title('unet_inceptionv3')
    plt.imshow(test_prediction2)
    plt.text(0.5, -0.1, str(round(iou2, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 5)
    plt.title('unet_vgg16')
    plt.imshow(test_prediction3)
    plt.text(0.5, -0.1, str(round(iou3, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 6)
    plt.title('linknet_resnet34')
    plt.imshow(test_prediction4)
    plt.text(0.5, -0.1, str(round(iou4, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 7)
    plt.title('linknet_inceptionv3')
    plt.imshow(test_prediction5)
    plt.text(0.5, -0.1, str(round(iou5, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 8)
    plt.title('fpn_resnet18')
    plt.imshow(test_prediction6)
    plt.text(0.5, -0.1, str(round(iou6, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 9, 9)
    plt.title('fpn_resnet34')
    plt.imshow(test_prediction7)
    plt.text(0.5, -0.1, str(round(iou7, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    if test_img_number < 10:
        plt.savefig("D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/prediction of val images with IOU/" + "0" + str(test_img_number) + ".png")
    else:
        plt.savefig("D:/peyman/Mars/207/7 models no black - binaryCrossEntropy/prediction of val images with IOU/" + str(test_img_number) + ".png")
 
    plt.show()
    
    
models = [model1, model2, model3, model4, model5, model6, model7]
preprocess_functions = [preprocess_input1, preprocess_input2, preprocess_input3, preprocess_input4, preprocess_input5, preprocess_input6, preprocess_input7]


def prediction(model_idx, image, patch_size):
    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
    patch_num=1
    for i in range(0, image.shape[0], 1024):   #Steps of 256
        for j in range(0, image.shape[1], 1024):  #Steps of 256
            #print(i, j)
            single_patch = image[i:i+patch_size, j:j+patch_size, :]
            single_patch_shape = single_patch.shape[:2]
            single_patch_input = np.expand_dims(single_patch, 0)
            
            preprocess_input = preprocess_functions[model_idx]
            single_patch_input = preprocess_input(single_patch_input)
            
            model = models[model_idx]
            single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
            segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
          
            print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1
    return segm_img   

    


image_path = "D:/peyman/Mars/207/dust_1024_G05_26_27_28_29/images/"
mask_path = "D:/peyman/Mars/207/dust_1024_G05_26_27_28_29/masks/"
colored_mask_path = "D:/peyman/Mars/207/dust_1024_G05_26_27_28_29/color coded masks/"

save_path = "D:/peyman/Mars/207/dust_1024_G05_26_27_28_29/predictions/"


for img in next(os.walk(image_path))[2]:
    image = Image.open(image_path + img)
    mask = Image.open(mask_path + img)
    colored_mask = Image.open(colored_mask_path + img + ".png")
    
    gt_mask = np.array(mask)
    gt_mask = gt_mask/255.
    
    # Resize the image while adding black pixels to the borders
    new_image = Image.new("RGB", (4096, 2048), color="black")
    new_image.paste(image, ((4096 - image.size[0]) // 2, (2048 - image.size[1]) // 2))
    new_image = np.array(new_image)
    
    segmented_image1 = prediction(0, new_image, 1024)
    segmented_image1 = Image.fromarray(segmented_image1)
    segmented_image1 = segmented_image1.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image1 = np.array(segmented_image1)
    segmented_image1 = np.where(segmented_image1 == 1, 0, 1)
    iou1 = calculate_iou(segmented_image1, gt_mask)
    
    
    segmented_image2 = prediction(1, new_image, 1024)
    segmented_image2 = Image.fromarray(segmented_image2)
    segmented_image2 = segmented_image2.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image2 = np.array(segmented_image2)
    segmented_image2 = np.where(segmented_image2 == 1, 0, 1)
    iou2 = calculate_iou(segmented_image2, gt_mask)
    
    
    segmented_image3 = prediction(2, new_image, 1024)
    segmented_image3 = Image.fromarray(segmented_image3)
    segmented_image3 = segmented_image3.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image3 = np.array(segmented_image3)
    segmented_image3 = np.where(segmented_image3 == 1, 0, 1)
    iou3 = calculate_iou(segmented_image3, gt_mask)
    
    
    segmented_image4 = prediction(3, new_image, 1024)
    segmented_image4 = Image.fromarray(segmented_image4)
    segmented_image4 = segmented_image4.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image4 = np.array(segmented_image4)
    segmented_image4 = np.where(segmented_image4 == 1, 0, 1)
    iou4 = calculate_iou(segmented_image4, gt_mask)
    
    
    segmented_image5 = prediction(4, new_image, 1024)
    segmented_image5 = Image.fromarray(segmented_image5)
    segmented_image5 = segmented_image5.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image5 = np.array(segmented_image5)
    segmented_image5 = np.where(segmented_image5 == 1, 0, 1)
    iou5 = calculate_iou(segmented_image5, gt_mask)
    
    
    segmented_image6 = prediction(5, new_image, 1024)
    segmented_image6 = Image.fromarray(segmented_image6)
    segmented_image6 = segmented_image6.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image6 = np.array(segmented_image6)
    segmented_image6 = np.where(segmented_image6 == 1, 0, 1)
    iou6 = calculate_iou(segmented_image6, gt_mask)
    
    
    segmented_image7 = prediction(6, new_image, 1024)
    segmented_image7 = Image.fromarray(segmented_image7)
    segmented_image7 = segmented_image7.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image7 = np.array(segmented_image7)
    segmented_image7 = np.where(segmented_image7 == 1, 0, 1)
    iou7 = calculate_iou(segmented_image7, gt_mask)
    
    
    plt.figure(figsize=(22, 5))
    plt.subplot(2, 10, 1)
    plt.title('Large Image')
    plt.imshow(image)
    plt.subplot(2, 10, 2)
    plt.title('Color Coded Mask')
    plt.imshow(colored_mask)
    plt.text(0.5, -0.15, img[:9], ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(2, 10, 3)
    plt.title('Mask')
    plt.imshow(mask, cmap = 'gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(2, 10, 4)
    plt.title('unet_resnet34')
    plt.imshow(segmented_image1)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou1, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(2, 10, 5)
    plt.title('unet_inceptionv3')
    plt.imshow(segmented_image2)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou2, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(2, 10, 6)
    plt.title('unet_vgg16')
    plt.imshow(segmented_image3)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou3, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(2, 10, 7)
    plt.title('linknet_resnet34')
    plt.imshow(segmented_image4)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou4, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(2, 10, 8)
    plt.title('linknet_inceptionv3')
    plt.imshow(segmented_image5)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou5, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(2, 10, 9)
    plt.title('fpn_resnet18')
    plt.imshow(segmented_image6)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou6, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(2, 10, 10)
    plt.title('fpn_resnet34')
    plt.imshow(segmented_image7)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou7, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.savefig(save_path + img)
    plt.show()



    

# Open the original image
image = Image.open("D:/peyman/Mars/207/dust_1024_G05_26_27_28_29/images/G05_day29.png")
mask = Image.open("D:/peyman/Mars/207/dust_1024_G05_26_27_28_29/masks/G05_day29.png")
colored_mask = Image.open("D:/peyman/Mars/207/dust_1024_G05_26_27_28_29/color coded masks/G05_day29.png.png")

# Resize the image while adding black pixels to the borders
new_image = Image.new("RGB", (4096, 2048), color="black")
new_image.paste(image, ((4096 - image.size[0]) // 2, (2048 - image.size[1]) // 2))

new_image = np.array(new_image)

segmented_image = prediction(6, new_image, 1024)

segmented_image = Image.fromarray(segmented_image)

segmented_image = segmented_image.crop((248, 124, 3848, 1925))  # left, top, right, bottom

segmented_image = np.array(segmented_image)

segmented_image = np.where(segmented_image == 1, 0, 1)

plt.hist(segmented_image.flatten())





gt_mask = np.array(mask)

gt_mask = gt_mask/255.

iou = calculate_iou(segmented_image, gt_mask)



plt.figure(figsize=(24, 8))
plt.subplot(241)
plt.title('Large Image')
plt.imshow(image)
plt.subplot(242)
plt.title('Color Coded Mask')
plt.imshow(colored_mask)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(243)
plt.title('Mask')
plt.imshow(mask, cmap = 'gray')
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.subplot(244)
plt.title('segmented_mask')
plt.imshow(segmented_image)
plt.text(0.5, -0.15, "IOU = " + str(round(iou, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=35)
plt.xticks([])  # remove x-axis values
plt.yticks([])  # remove y-axis values
plt.show()



