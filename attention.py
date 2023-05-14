# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 18:32:57 2023

@author: dat
"""

# https://youtu.be/ZoJuhRbzEiM
"""
Mitochondria semantic segmentation using U-net, Attention Unet and R2 Unet
and others using keras-unet-collection library.
# https://github.com/yingkaisha/keras-unet-collection
Author: Dr. Sreenivas Bhattiprolu
Dataset from: https://www.epfl.ch/labs/cvlab/data/data-em/
Images and masks are divided into patches of 256x256. 
"""

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from datetime import datetime 
import cv2
from PIL import Image
from numpy import asarray
from numpy import save
from numpy import load
import pandas as pd
#from keras import backend, optimizers

os.chdir("D:/peyman/Mars/207/attention unet/1024 patch size/")

np.random.seed(0)
tf.random.set_seed(0)


# force channels-first ordering for all loaded images
#backend.set_image_data_format('channels_last')  #The models are designed to use channels first


image_directory = 'D:/peyman/Mars/207/attention unet/dust_1024_patch_size/img_patches/'
mask_directory = 'D:/peyman/Mars/207/attention unet/dust_1024_patch_size/mask_patches/'



#SIZE = 1024
#RESIZED_SIZE = 512
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    #print(image_directory+image_name)
    image = cv2.imread(image_directory+image_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (RESIZED_SIZE, RESIZED_SIZE))
    image = Image.fromarray(image)
    #image = image.resize((SIZE, SIZE))
    image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    image = cv2.imread(mask_directory+image_name, 0)
    #image = cv2.resize(image, (RESIZED_SIZE, RESIZED_SIZE), interpolation=cv2.INTER_NEAREST)
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



# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.20, random_state = 0)





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


n_classes=2
from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

# save to csv file
save('X_train.npy', X_train)
save('X_test.npy', X_test)
save('y_train_cat.npy', y_train_cat)
save('y_test_cat.npy', y_test_cat)



X_train = load('X_train.npy')
X_test = load('X_test.npy')
y_train_cat = load('y_train_cat.npy')
y_test_cat = load('y_test_cat.npy')

#######################################

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 2  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 1
#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.
#from focal_loss import BinaryFocalLoss

###############################################################################
#Try various models: Unet, Attention_UNet, and Attention_ResUnet


from keras_unet_collection import models, losses

def iou_score(y_true, y_pred):
    # convert y_pred to binary mask
    y_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)
    # calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2]) - intersection
    # calculate iou score
    iou = intersection / union
    # return mean iou score across batch
    return tf.reduce_mean(iou)



###############################################################################
#Model 1: Unet with ImageNet trained VGG16 backbone
#help(models.att_unet_2d)

model_Unet = models.unet_plus_2d((IMG_HEIGHT, IMG_WIDTH, 3), filter_num=[16, 32, 64, 128, 256], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=False, unpool=False, 
                           backbone='ResNet50V2', weights='imagenet', 
                           freeze_backbone=True, freeze_batch_norm=True)

#metrics = [IOUScore(threshold=0.5)]

model_Unet.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), metrics=[iou_score])

# model_Unet.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
#               metrics=metrics)


#print(model_Unet.summary())

#start1 = datetime.now() 


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='D:/peyman/Mars/207/attention unet/1024 patch size/unet_3plus_2d_ResNet50V2_100epochs_G05_1024patchsize.h5',
    monitor='val_iou_score',
    mode='max',
    save_best_only=True, verbose=1)

Unet_history = model_Unet.fit(X_train, 
                   y_train_cat, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test_cat), 
                    epochs=100)

#stop1 = datetime.now()
#Execution time of the model 
#execution_time_Unet = stop1-start1
#print("UNet execution time is: ", execution_time_Unet)

model_Unet.save('mitochondria_unet_collection_UNet_50epochs.hdf5')

#plot the training and validation accuracy and loss at each epoch
loss = Unet_history.history['loss']
val_loss = Unet_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss unet_3plus_2d_ResNet50V2_100epochs_G05_1024patchsize')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


acc = Unet_history.history['iou_score']
val_acc = Unet_history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU unet_3plus_2d_ResNet50V2_100epochs_G05_1024patchsize')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


import csv


rows = zip(loss, val_loss, acc, val_acc)
with open('D:/peyman/Mars/207/attention unet/1024 patch size/unet_3plus_2d_ResNet50V2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Training loss', 'Validation loss', 'Training IOU', 'Validation IOU'])
    for row in rows:
        writer.writerow(row)
    

#############################################################
# Unet Plus
help(models.unet_plus_2d)

model_Unet_plus = models.unet_plus_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=False, unpool=False, 
                           backbone='VGG16', weights='imagenet', 
                           freeze_backbone=True, freeze_batch_norm=True, 
                           name='unet_plus')



model_Unet_plus.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_Unet_plus.summary())

start2 = datetime.now() 

Unet_plus_history = model_Unet_plus.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop2 = datetime.now()
#Execution time of the model 
execution_time_Unet_plus = stop2-start2
print("UNet plus execution time is: ", execution_time_Unet_plus)

model_Unet_plus.save('mitochondria_unet_collection_UNet_plus_50epochs.hdf5')
##############################################################################
#Attention U-net with an ImageNet-trained backbone

help(models.att_unet_2d)

model_att_unet = models.att_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           atten_activation='ReLU', attention='add', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=False, unpool=False, 
                           backbone='VGG16', weights='imagenet', 
                           freeze_backbone=True, freeze_batch_norm=True, 
                           name='attunet')


model_att_unet.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_att_unet.summary())

start3 = datetime.now() 

att_unet_history = model_att_unet.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop3 = datetime.now()
#Execution time of the model 
execution_time_att_Unet = stop3-start3
print("Attention UNet execution time is: ", execution_time_att_Unet)

model_att_unet.save('mitochondria_unet_collection_att_UNet_50epochs.hdf5')

#######################################################################
#Without loading weights
#####################################################################
#Model 4: Unet with ImageNet trained VGG16 backbone
help(models.unet_2d)

model_Unet_from_scratch = models.unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=True, unpool=True, 
                           backbone=None, weights=None, 
                           freeze_backbone=False, freeze_batch_norm=False, 
                           name='unet')


model_Unet_from_scratch.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_Unet_from_scratch.summary())

start4 = datetime.now() 

Unet_from_scratch_history = model_Unet_from_scratch.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop4 = datetime.now()
#Execution time of the model 
execution_time_Unet_from_scratch = stop4-start4
print("UNet from scratch execution time is: ", execution_time_Unet_from_scratch)

model_Unet_from_scratch.save('mitochondria_unet_collection_UNet_from_scratch_50epochs.hdf5')

####################################################################################
#Model 5: Recurrent Residual (R2) U-Net
help(models.r2_unet_2d)

model_r2_Unet_from_scratch = models.r2_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           recur_num=2,
                           activation='ReLU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=True, unpool=True, 
                           name='r2_unet')


model_r2_Unet_from_scratch.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_r2_Unet_from_scratch.summary())

start5 = datetime.now() 

r2_Unet_from_scratch_history = model_r2_Unet_from_scratch.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop5 = datetime.now()
#Execution time of the model 
execution_time_r2_Unet_from_scratch = stop5-start5
print("R2 UNet from scratch execution time is: ", execution_time_r2_Unet_from_scratch)

model_r2_Unet_from_scratch.save('mitochondria_unet_collection_r2_UNet_from_scratch_50epochs.hdf5')

############################################################################
#Model 6: Attention Unet from scratch - no backbone or weights.
help(models.att_unet_2d)

model_att_unet_from_scratch = models.att_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           atten_activation='ReLU', attention='add', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=True, unpool=True, 
                           backbone=None, weights=None, 
                           freeze_backbone=False, freeze_batch_norm=False, 
                           name='attunet')


model_att_unet_from_scratch.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_att_unet_from_scratch.summary())

start6 = datetime.now() 

att_unet_from_scratch_history = model_att_unet_from_scratch.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop6 = datetime.now()
#Execution time of the model 
execution_time_att_Unet_from_scratch = stop6-start6
print("Attention UNet from scratch execution time is: ", execution_time_att_Unet_from_scratch)

model_att_unet_from_scratch.save('mitochondria_unet_collection_att_UNet_from_scratch_50epochs.hdf5')
############################################################################
# convert the history.history dict to a pandas DataFrame and save as csv for
# future plotting
import pandas as pd    
unet_history_df = pd.DataFrame(Unet_history.history) 
unet_plus_history_df = pd.DataFrame(Unet_plus_history.history) 
att_unet_history_df = pd.DataFrame(att_unet_history.history) 

unet_from_scratch_history_df = pd.DataFrame(Unet_from_scratch_history.history) 
r2_Unet_from_scratch_history_df = pd.DataFrame(r2_Unet_from_scratch_history.history) 
att_unet_from_scratch_history_df = pd.DataFrame(att_unet_from_scratch_history.history) 

with open('unet_history_df.csv', mode='w') as f:
    unet_history_df.to_csv(f)
    
with open('unet_plus_history_df.csv', mode='w') as f:
    unet_plus_history_df.to_csv(f)

with open('att_unet_history_df.csv', mode='w') as f:
    att_unet_history_df.to_csv(f)    

with open('unet_from_scratch_history_df.csv', mode='w') as f:
    unet_from_scratch_history_df.to_csv(f)    
    
with open('r2_Unet_from_scratch_history_df.csv', mode='w') as f:
    r2_Unet_from_scratch_history_df.to_csv(f)    

with open('att_unet_from_scratch_history_df.csv', mode='w') as f:
    att_unet_from_scratch_history_df.to_csv(f)        


#######################################################################
#Check history plots, one model at a time
history = Unet_history
history = Unet_plus_history
history = att_unet_history
history = Unet_from_scratch_history
history = r2_Unet_from_scratch_history
history = att_unet_from_scratch_history

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['dice_coef']
#acc = history.history['accuracy']
val_acc = history.history['val_dice_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Dice')
plt.plot(epochs, val_acc, 'r', label='Validation Dice')
plt.title('Training and validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()

#######################################################



from keras.models import load_model

#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_DenseNet121_100epochs_G05_1024patchsize.h5', compile=False)
model2 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_DenseNet169_100epochs_G05_1024patchsize.h5', compile=False)
model3 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_DenseNet201_100epochs_G05_1024patchsize.h5', compile=False)
model4 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_ResNet50_100epochs_G05_1024patchsize.h5', compile=False)
model5 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_ResNet50V2_100epochs_G05_1024patchsize.h5', compile=False)
model6 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_ResNet101_100epochs_G05_1024patchsize.h5', compile=False)
model7 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_ResNet101V2_100epochs_G05_1024patchsize.h5', compile=False)
model8 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_resnet152_100epochs_G05_1024patchsize.h5', compile=False)
model9 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_resnet152v2_100epochs_G05_1024patchsize.h5', compile=False)
model10 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_VGG16_100epochs_G05_1024patchsize.h5', compile=False)
model11 = load_model('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/unet_plus_2d_VGG19_100epochs_G05_1024patchsize.h5', compile=False)

##############################################################
#Test some random images
import random
#test_img_number = random.randint(0, len(X_test2))


for test_img_number in range(len(X_test)):
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    test_img_input=np.expand_dims(test_img, 0)

    test_pred1 = model1.predict(test_img_input)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]

    test_pred2 = model2.predict(test_img_input)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]

    test_pred3 = model3.predict(test_img_input)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

    test_pred4 = model4.predict(test_img_input)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]

    test_pred5 = model5.predict(test_img_input)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]

    test_pred6 = model6.predict(test_img_input)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]

    test_pred7 = model7.predict(test_img_input)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]

    test_pred8 = model8.predict(test_img_input)
    test_prediction8 = np.argmax(test_pred8, axis=3)[0,:,:]

    test_pred9 = model9.predict(test_img_input)
    test_prediction9 = np.argmax(test_pred9, axis=3)[0,:,:]

    test_pred10 = model10.predict(test_img_input)
    test_prediction10 = np.argmax(test_pred10, axis=3)[0,:,:]

    test_pred11 = model11.predict(test_img_input)
    test_prediction11 = np.argmax(test_pred11, axis=3)[0,:,:]


    p = X_test[test_img_number,:,:,:].squeeze()

    plt.figure(figsize=(22, 5))
    plt.subplot(1, 13, 1)
    plt.title('Testing Image')
    plt.imshow(p)
    plt.subplot(1, 13, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 3)
    plt.title('densenet121')
    plt.imshow(test_prediction1)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 4)
    plt.title('densenet169')
    plt.imshow(test_prediction2)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 5)
    plt.title('densenet201')
    plt.imshow(test_prediction3)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 6)
    plt.title('resnet50')
    plt.imshow(test_prediction4)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 7)
    plt.title('resnet50v2')
    plt.imshow(test_prediction5)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 8)
    plt.title('resnet101')
    plt.imshow(test_prediction6)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 9)
    plt.title('resnet101v2')
    plt.imshow(test_prediction7)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 10)
    plt.title('resnet152')
    plt.imshow(test_prediction8)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 11)
    plt.title('resnet152v2')
    plt.imshow(test_prediction9)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 12)
    plt.title('vgg16')
    plt.imshow(test_prediction10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 13)
    plt.title('vgg19')
    plt.imshow(test_prediction11)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    if test_img_number < 10:
        plt.savefig("D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/prediction of val images/" + "0" + str(test_img_number) + ".png")
    else:
        plt.savefig("D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/prediction of val images/" + str(test_img_number) + ".png")
 
    plt.show()    


test_labels = ["25_03", "25_10", "25_11", "25_12", "25_13", "26_00", "26_01", 
               "26_02", "26_03", "26_11", "26_12", "27_00", "27_01", "27_02", 
               "27_03", "27_10", "27_11", "28_00", "28_01", "28_02", "28_03", 
               "28_10", "28_11", "29_00", "29_01", "29_02", "29_03", "29_10", 
               "30_00", "30_01"]

for test_lbl in test_labels:
    test_img = cv2.imread("D:/peyman/Mars/207/dust_1024_patch_size/test_img/image_G05_day" 
                          + test_lbl + ".png")
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)

    #ground_truth=y_test[test_img_number]
    ground_truth=cv2.imread("D:/peyman/Mars/207/dust_1024_patch_size/test_mask/image_G05_day" 
                          + test_lbl + ".png", 0)

    test_img=np.expand_dims(test_img, 0)


    test_pred1 = model1.predict(test_img)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]

    test_pred2 = model2.predict(test_img)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]

    test_pred3 = model3.predict(test_img)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

    test_pred4 = model4.predict(test_img)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]

    test_pred5 = model5.predict(test_img)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]

    test_pred6 = model6.predict(test_img)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]

    test_pred7 = model7.predict(test_img)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]

    test_pred8 = model8.predict(test_img)
    test_prediction8 = np.argmax(test_pred8, axis=3)[0,:,:]

    test_pred9 = model9.predict(test_img)
    test_prediction9 = np.argmax(test_pred9, axis=3)[0,:,:]

    test_pred10 = model10.predict(test_img)
    test_prediction10 = np.argmax(test_pred10, axis=3)[0,:,:]

    test_pred11 = model11.predict(test_img)
    test_prediction11 = np.argmax(test_pred11, axis=3)[0,:,:]



    #prediction1 = (model1.predict(test_img_input1)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction2 = (model2.predict(test_img_input2)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction3 = (model3.predict(test_img_input3)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction4 = (model4.predict(test_img_input4)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction5 = (model5.predict(test_img_input5)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction6 = (model6.predict(test_img_input6)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction7 = (model7.predict(test_img_input7)[0,:,:,0] > 0.5).astype(np.uint8)



    plt.figure(figsize=(22, 5))
    plt.subplot(1, 13, 1)
    plt.title('Testing Image')
    plt.imshow(test_img.squeeze())
    plt.subplot(1, 13, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth, cmap='gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 3)
    plt.title('densenet121')
    plt.imshow(test_prediction1)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 4)
    plt.title('densenet169')
    plt.imshow(test_prediction2)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 5)
    plt.title('densenet201')
    plt.imshow(test_prediction3)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 6)
    plt.title('resnet50')
    plt.imshow(test_prediction4)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 7)
    plt.title('resnet50v2')
    plt.imshow(test_prediction5)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 8)
    plt.title('resnet101')
    plt.imshow(test_prediction6)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 9)
    plt.title('resnet101v2')
    plt.imshow(test_prediction7)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 10)
    plt.title('resnet152')
    plt.imshow(test_prediction8)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 11)
    plt.title('resnet152v2')
    plt.imshow(test_prediction9)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 12)
    plt.title('vgg16')
    plt.imshow(test_prediction10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 13)
    plt.title('vgg19')
    plt.imshow(test_prediction11)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.savefig("D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/prediction of test images/" + test_lbl + ".png")
 
    plt.show()

    


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
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11]

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
        
        test_pred = model.predict(test_img_input)
        test_prediction = np.argmax(test_pred, axis=3)[0,:,:]
        
        
        gt_mask = test_mask_dataset[i]
        gt_mask = gt_mask/255.
        
        # Calculate the IOU for this test image
        iou = calculate_iou(test_prediction, gt_mask)
        iou_values.append(iou)
    
    # add a new column to the dataframe for this model's IOU values
    iou_df[f'model{model_idx+1}'] = iou_values
  
    
# create a new dataframe to store the average IOU values for each model
avg_iou_df = pd.DataFrame(columns=[f'model{i}' for i in range(1, 12)])

# compute the average IOU values for each model and add them to the new dataframe
for model_idx, model in enumerate(models):
    avg_iou = sum(iou_df[f'model{model_idx+1}']) / len(iou_df[f'model{model_idx+1}'])
    avg_iou_df.loc[0, f'model{model_idx+1}'] = avg_iou
    
    
# Save the dataframe to a CSV file on your computer
iou_df.to_csv('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/iou_values_for_test_images.csv', index=False)
avg_iou_df.to_csv('D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/avg_iou_values_for_test_images.csv', index=False)




cnt = 0

test_labels = ["25_03", "25_10", "25_11", "25_12", "25_13", "26_00", "26_01", 
               "26_02", "26_03", "26_11", "26_12", "27_00", "27_01", "27_02", 
               "27_03", "27_10", "27_11", "28_00", "28_01", "28_02", "28_03", 
               "28_10", "28_11", "29_00", "29_01", "29_02", "29_03", "29_10", 
               "30_00", "30_01"]

for test_lbl in test_labels:
    test_img = cv2.imread("D:/peyman/Mars/207/dust_1024_patch_size/test_img/image_G05_day" 
                          + test_lbl + ".png")
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)

    #ground_truth=y_test[test_img_number]
    ground_truth=cv2.imread("D:/peyman/Mars/207/dust_1024_patch_size/test_mask/image_G05_day" 
                          + test_lbl + ".png", 0)

    test_img=np.expand_dims(test_img, 0)


    test_pred1 = model1.predict(test_img)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]

    test_pred2 = model2.predict(test_img)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]

    test_pred3 = model3.predict(test_img)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

    test_pred4 = model4.predict(test_img)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]

    test_pred5 = model5.predict(test_img)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]

    test_pred6 = model6.predict(test_img)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]

    test_pred7 = model7.predict(test_img)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]

    test_pred8 = model8.predict(test_img)
    test_prediction8 = np.argmax(test_pred8, axis=3)[0,:,:]

    test_pred9 = model9.predict(test_img)
    test_prediction9 = np.argmax(test_pred9, axis=3)[0,:,:]

    test_pred10 = model10.predict(test_img)
    test_prediction10 = np.argmax(test_pred10, axis=3)[0,:,:]

    test_pred11 = model11.predict(test_img)
    test_prediction11 = np.argmax(test_pred11, axis=3)[0,:,:]



    #prediction1 = (model1.predict(test_img_input1)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction2 = (model2.predict(test_img_input2)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction3 = (model3.predict(test_img_input3)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction4 = (model4.predict(test_img_input4)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction5 = (model5.predict(test_img_input5)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction6 = (model6.predict(test_img_input6)[0,:,:,0] > 0.5).astype(np.uint8)
    # prediction7 = (model7.predict(test_img_input7)[0,:,:,0] > 0.5).astype(np.uint8)



    plt.figure(figsize=(22, 5))
    plt.subplot(1, 13, 1)
    plt.title('Testing Image')
    plt.imshow(test_img.squeeze())
    plt.subplot(1, 13, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth, cmap='gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 3)
    plt.title('densenet121')
    plt.imshow(test_prediction1)
    plt.text(0.5, -0.1, str(round(iou_df["model1"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 4)
    plt.title('densenet169')
    plt.imshow(test_prediction2)
    plt.text(0.5, -0.1, str(round(iou_df["model2"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 5)
    plt.title('densenet201')
    plt.imshow(test_prediction3)
    plt.text(0.5, -0.1, str(round(iou_df["model3"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 6)
    plt.title('resnet50')
    plt.imshow(test_prediction4)
    plt.text(0.5, -0.1, str(round(iou_df["model4"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 7)
    plt.title('resnet50v2')
    plt.imshow(test_prediction5)
    plt.text(0.5, -0.1, str(round(iou_df["model5"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 8)
    plt.title('resnet101')
    plt.text(0.5, -0.1, str(round(iou_df["model6"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.imshow(test_prediction6)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 9)
    plt.title('resnet101v2')
    plt.imshow(test_prediction7)
    plt.text(0.5, -0.1, str(round(iou_df["model7"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 10)
    plt.title('resnet152')
    plt.imshow(test_prediction8)
    plt.text(0.5, -0.1, str(round(iou_df["model8"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 11)
    plt.title('resnet152v2')
    plt.imshow(test_prediction9)
    plt.text(0.5, -0.1, str(round(iou_df["model9"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 12)
    plt.title('vgg16')
    plt.imshow(test_prediction10)
    plt.text(0.5, -0.1, str(round(iou_df["model10"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 13)
    plt.title('vgg19')
    plt.imshow(test_prediction11)
    plt.text(0.5, -0.1, str(round(iou_df["model11"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.savefig("D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/prediction of test images with IOU/" + test_lbl + ".png")
    plt.show()
    cnt += 1





for test_img_number in range(len(X_test)):
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    gt_mask = ground_truth
    gt_mask = ground_truth/255.
    gt_mask = gt_mask.squeeze()
    test_img_input=np.expand_dims(test_img, 0)

    test_pred1 = model1.predict(test_img_input)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]
    iou1 = calculate_iou(test_prediction1, gt_mask)

    test_pred2 = model2.predict(test_img_input)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]
    iou2 = calculate_iou(test_prediction2, gt_mask)

    test_pred3 = model3.predict(test_img_input)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]
    iou3 = calculate_iou(test_prediction3, gt_mask)

    test_pred4 = model4.predict(test_img_input)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]
    iou4 = calculate_iou(test_prediction4, gt_mask)

    test_pred5 = model5.predict(test_img_input)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]
    iou5 = calculate_iou(test_prediction5, gt_mask)

    test_pred6 = model6.predict(test_img_input)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]
    iou6 = calculate_iou(test_prediction6, gt_mask)

    test_pred7 = model7.predict(test_img_input)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]
    iou7 = calculate_iou(test_prediction7, gt_mask)
    
    test_pred8 = model8.predict(test_img_input)
    test_prediction8 = np.argmax(test_pred8, axis=3)[0,:,:]
    iou8 = calculate_iou(test_prediction8, gt_mask)

    test_pred9 = model9.predict(test_img_input)
    test_prediction9 = np.argmax(test_pred9, axis=3)[0,:,:]
    iou9 = calculate_iou(test_prediction9, gt_mask)

    test_pred10 = model10.predict(test_img_input)
    test_prediction10 = np.argmax(test_pred10, axis=3)[0,:,:]
    iou10 = calculate_iou(test_prediction10, gt_mask)

    test_pred11 = model11.predict(test_img_input)
    test_prediction11 = np.argmax(test_pred11, axis=3)[0,:,:]
    iou11 = calculate_iou(test_prediction11, gt_mask)


    p = X_test[test_img_number,:,:,:].squeeze()

    plt.figure(figsize=(22, 5))
    plt.subplot(1, 13, 1)
    plt.title('Testing Image')
    plt.imshow(p)
    plt.subplot(1, 13, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 3)
    plt.title('densenet121')
    plt.imshow(test_prediction1)
    plt.text(0.5, -0.1, str(round(iou1, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 4)
    plt.title('densenet169')
    plt.imshow(test_prediction2)
    plt.text(0.5, -0.1, str(round(iou2, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 5)
    plt.title('densenet201')
    plt.imshow(test_prediction3)
    plt.text(0.5, -0.1, str(round(iou3, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 6)
    plt.title('resnet50')
    plt.imshow(test_prediction4)
    plt.text(0.5, -0.1, str(round(iou4, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 7)
    plt.title('resnet50v2')
    plt.imshow(test_prediction5)
    plt.text(0.5, -0.1, str(round(iou5, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 8)
    plt.title('resnet101')
    plt.imshow(test_prediction6)
    plt.text(0.5, -0.1, str(round(iou6, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 9)
    plt.title('resnet101v2')
    plt.imshow(test_prediction7)
    plt.text(0.5, -0.1, str(round(iou7, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 10)
    plt.title('resnet152')
    plt.imshow(test_prediction8)
    plt.text(0.5, -0.1, str(round(iou8, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 11)
    plt.title('resnet152v2')
    plt.imshow(test_prediction9)
    plt.text(0.5, -0.1, str(round(iou9, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 12)
    plt.title('vgg16')
    plt.imshow(test_prediction10)
    plt.text(0.5, -0.1, str(round(iou10, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 13)
    plt.title('vgg19')
    plt.imshow(test_prediction11)
    plt.text(0.5, -0.1, str(round(iou11, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    if test_img_number < 10:
        plt.savefig("D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/prediction of val images with IOU/" + "0" + str(test_img_number) + ".png")
    else:
        plt.savefig("D:/peyman/Mars/207/attention unet/1024 patch size/models/unet_plus_2d/prediction of val images with IOU/" + str(test_img_number) + ".png")
 
    plt.show()