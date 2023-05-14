# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:53:42 2022

@author: Peyman
"""

#from simple_unet_model import simple_unet_model 
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#from patchify import patchify, unpatchify
import random
import math

#import rioxarray
#import xarray

import shutil

import pandas as pd








############################################################################
#ALL DUST IMAGES - patching 1024x1024 - extracting 1000 patches
##############################################################################


# n = 300

# idxs = random.sample(range(0, 17432), n)


# TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/05 - dust masks/all_1024_img_patches/"

# patch_size = 1024



# p = next(os.walk(TRAIN_PATH))[2]

# for folder in idxs:
#     img = cv2.imread(TRAIN_PATH + "/" + p[folder])
#     img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#     cv2.imwrite("D:/FHJ/Thesis/Giacomo/05 - dust masks/dust_300_img_patches/" + p[folder] + '.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))



# TRAIN_PATH_MASK = "D:/FHJ/Thesis/Giacomo/05 - dust masks/all_1024_mask_patches/"
# PATCH_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/all_1024_mask_patches/'

# patch_size = 1024

# p = next(os.walk(TRAIN_PATH_MASK))[2]

# for folder in idxs:
#     img = cv2.imread(TRAIN_PATH_MASK + "/" + p[folder])
#     img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#     cv2.imwrite("D:/FHJ/Thesis/Giacomo/05 - dust masks/dust_300_mask_patches/" + p[folder] + '.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

  
    

###########################################################################
#ALL DUST IMAGES - patching 256 x 256
#############################################################################

# TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/05 - dust masks/images/"
# PATCH_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/all_256_img_patches/'

# patch_size = 256


# for folder in next(os.walk(TRAIN_PATH))[1]:
#     for sub_folder in next(os.walk(TRAIN_PATH + folder))[1]:
#         for img_file in next(os.walk(TRAIN_PATH + folder + "/" + sub_folder))[2]:
#             path = TRAIN_PATH + folder + "/" + sub_folder + "/" + img_file
#             img = cv2.imread(path)
#             img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#             for i in range(7):   #Steps of 256
#                 for j in range(14):  #Steps of 256
#                     single_patch = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size:]
#                     cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))

# TRAIN_PATH_MASK = "D:/FHJ/Thesis/Giacomo/05 - dust masks/masks/"
# MASK_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/all_256_mask_patches/'

# for folder in next(os.walk(TRAIN_PATH_MASK))[1]:
#     for sub_folder in next(os.walk(TRAIN_PATH_MASK + folder))[1]:
#         for img_file in next(os.walk(TRAIN_PATH_MASK + folder + "/" + sub_folder))[2]:
#             path = TRAIN_PATH_MASK + folder + "/" + sub_folder + "/" + img_file
#             img = cv2.imread(path, 0)
#             #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#             for i in range(7):   #Steps of 256
#                 for j in range(14):  #Steps of 256
#                     single_patch = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
#                     cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)





###########################################################################
#ALL DUST IMAGES - patching 256 x 256 - Removing black mask files
#############################################################################

# TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/05 - dust masks/all_256_img_patches/"
# TRAIN_PATH_MASK = "D:/FHJ/Thesis/Giacomo/05 - dust masks/all_256_mask_patches/"
# PATCH_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/all_256_img_patches_not_black/'
# PATCH_PATH_MASK = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/all_256_mask_patches_not_black/'

# for img_file in next(os.walk(TRAIN_PATH_MASK))[2]:
#     path = TRAIN_PATH_MASK + img_file
#     img = cv2.imread(path, 0)
    
#     if img.any() == True:
#         cv2.imwrite(PATCH_PATH_MASK + img_file, img)
#         col_img = cv2.imread(TRAIN_PATH + img_file)
#         cv2.imwrite(PATCH_PATH + img_file , col_img)

        

############################################################################
#ALL DUST IMAGES - patching 1024x1024
##############################################################################

# TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/05 - dust masks/images/"
# PATCH_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/all_1024_img_patches/'

# patch_size = 1024


# for folder in next(os.walk(TRAIN_PATH))[1]:
#     for sub_folder in next(os.walk(TRAIN_PATH + folder))[1]:
#         for img_file in next(os.walk(TRAIN_PATH + folder + "/" + sub_folder))[2]:
#             path = TRAIN_PATH + folder + "/" + sub_folder + "/" + img_file
#             img = cv2.imread(path)
#             img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#             for i in range(2):   #Steps of 256
#                 for j in range(4):  #Steps of 256
#                     if j != 3:
#                         if i == 0:
#                             single_patch = img[0:patch_size, j*patch_size:(j+1)*patch_size:]
#                             cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
#                         if i == 1:
#                             single_patch = img[777:, j*patch_size:(j+1)*patch_size:]
#                             cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
    
#                     if j == 3:
#                         if i == 0:
#                             single_patch = img[0:patch_size, 2576::]
#                             cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
#                         if i == 1:
#                             single_patch = img[777:, 2576::]
#                             cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))


# TRAIN_PATH_MASK = "D:/FHJ/Thesis/Giacomo/05 - dust masks/masks/"
# MASK_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/all_1024_mask_patches/'

# patch_size = 1024

# for folder in next(os.walk(TRAIN_PATH_MASK))[1]:
#     for sub_folder in next(os.walk(TRAIN_PATH_MASK + folder))[1]:
#         for img_file in next(os.walk(TRAIN_PATH_MASK + folder + "/" + sub_folder))[2]:
#             path = TRAIN_PATH_MASK + folder + "/" + sub_folder + "/" + img_file
#             img = cv2.imread(path, 0)
#             #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#             for i in range(2):   #Steps of 256
#                 for j in range(4):  #Steps of 256
#                     if j != 3:
#                         if i == 0:
#                             single_patch = img[0:patch_size, j*patch_size:(j+1)*patch_size]
#                             cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
#                         if i == 1:
#                             single_patch = img[777:, j*patch_size:(j+1)*patch_size]
#                             cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
    
#                     if j == 3:
#                         if i == 0:
#                             single_patch = img[0:patch_size, 2576:]
#                             cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
#                         if i == 1:
#                             single_patch = img[777:, 2576:]
#                             cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
    
    
    



############################################################################
#Test dust G05 - patching 1024x1024
##############################################################################

# TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/05 - dust masks/test_1024_patch_size/images/"
# PATCH_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/test_1024_patch_size/img_patches/'

# patch_size = 1024

# #cnt = 0
# for folder in next(os.walk(TRAIN_PATH))[1]:
#     for img_file in next(os.walk(TRAIN_PATH + folder))[2]:
#         #cnt +=1
#         path = TRAIN_PATH + folder + "/" + img_file
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#         for i in range(2):   #Steps of 256
#             for j in range(4):  #Steps of 256
#                 if j != 3:
#                     if i == 0:
#                         single_patch = img[0:patch_size, j*patch_size:(j+1)*patch_size:]
#                         cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
#                     if i == 1:
#                         single_patch = img[777:, j*patch_size:(j+1)*patch_size:]
#                         cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))

#                 if j == 3:
#                     if i == 0:
#                         single_patch = img[0:patch_size, 2576::]
#                         cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
#                     if i == 1:
#                         single_patch = img[777:, 2576::]
#                         cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))


# TRAIN_PATH_MASK = "D:/FHJ/Thesis/Giacomo/05 - dust masks/test_1024_patch_size/masks/"
# MASK_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/test_1024_patch_size/mask_patches/'

# patch_size = 1024

# #cnt = 0
# for folder in next(os.walk(TRAIN_PATH_MASK))[1]:
#     for img_file in next(os.walk(TRAIN_PATH_MASK + folder))[2]:
#         #cnt +=1
#         path = TRAIN_PATH_MASK + folder + "/" + img_file
#         img = cv2.imread(path, 0)
#         #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#         for i in range(2):   #Steps of 256
#             for j in range(4):  #Steps of 256
#                 if j != 3:
#                     if i == 0:
#                         single_patch = img[0:patch_size, j*patch_size:(j+1)*patch_size]
#                         cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
#                     if i == 1:
#                         single_patch = img[777:, j*patch_size:(j+1)*patch_size]
#                         cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)

#                 if j == 3:
#                     if i == 0:
#                         single_patch = img[0:patch_size, 2576:]
#                         cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
#                     if i == 1:
#                         single_patch = img[777:, 2576:]
#                         cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)






############################################################################
#Test dust G05 - patching 1024x1024 - CONFLEV
##############################################################################

# TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/05 - dust masks/test_1024_patch_size/images/"
# PATCH_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/test_1024_patch_size/img_patches/'

# patch_size = 1024

# #cnt = 0
# for folder in next(os.walk(TRAIN_PATH))[1]:
#     for img_file in next(os.walk(TRAIN_PATH + folder))[2]:
#         #cnt +=1
#         path = TRAIN_PATH + folder + "/" + img_file
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#         for i in range(2):   #Steps of 256
#             for j in range(4):  #Steps of 256
#                 if j != 3:
#                     if i == 0:
#                         single_patch = img[0:patch_size, j*patch_size:(j+1)*patch_size:]
#                         cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
#                     if i == 1:
#                         single_patch = img[777:, j*patch_size:(j+1)*patch_size:]
#                         cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))

#                 if j == 3:
#                     if i == 0:
#                         single_patch = img[0:patch_size, 2576::]
#                         cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
#                     if i == 1:
#                         single_patch = img[777:, 2576::]
#                         cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))


# TRAIN_PATH_MASK = "D:/FHJ/Thesis/Giacomo/G05_conflev/"
# MASK_PATH = 'D:/FHJ/Thesis/Giacomo/G05_CONFLEV_MASK_PATCH/'

# patch_size = 1024

# #cnt = 0
# for img_file in next(os.walk(TRAIN_PATH_MASK))[2]:
#     #cnt +=1
#     path = TRAIN_PATH_MASK + img_file
#     img = cv2.imread(path, 0)
#     #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#     for i in range(2):   #Steps of 256
#         for j in range(4):  #Steps of 256
#             if j != 3:
#                 if i == 0:
#                     single_patch = img[0:patch_size, j*patch_size:(j+1)*patch_size]
#                     cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
#                 if i == 1:
#                     single_patch = img[777:, j*patch_size:(j+1)*patch_size]
#                     cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)

#             if j == 3:
#                 if i == 0:
#                     single_patch = img[0:patch_size, 2576:]
#                     cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
#                 if i == 1:
#                     single_patch = img[777:, 2576:]
#                     cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)


# rgb_mask = np.full((1024, 1024, 3), 255)
# rgb_mask[:, :, 2] = 0 # yellow


# rgb_mask = np.full((1024, 1024, 3), 255)
# rgb_mask[:, :, 0] = 0 
# rgb_mask[:, :, 1] = 0 # blue


# rgb_mask = np.full((1024, 1024, 3), 255)
# rgb_mask[:, :, 1] = 0 
# rgb_mask[:, :, 2] = 0 # red


# rgb_mask = np.full((1024, 1024, 3), 255)
# rgb_mask[:, :, 0] = 0 
# rgb_mask[:, :, 2] = 0 # green


# path = "D:/FHJ/Thesis/Giacomo/G05 conflev real test masks/"

# p = next(os.walk(path))

# for mask in next(os.walk(path))[2]:
#     img = cv2.imread(path + mask, 0)
#     rgb_mask = np.full((1024, 1024, 3), 0)

#     for i in range(1024):
#         for j in range(1024):
#             if img[j, i] == 4:
#                 rgb_mask[j, i, 1] = 255 # green
                
#             elif img[j, i] == 3:
#                 rgb_mask[j, i, 2] = 255 # blue
                
#             elif img[j, i] == 2:
#                 rgb_mask[j, i, 0] = 255 # yellow
#                 rgb_mask[j, i, 1] = 255 # yellow
                
#             elif img[j, i] == 1:
#                 rgb_mask[j, i, 0] = 255 # red
    
#     rgb_mask = np.float32(rgb_mask)
#     rgb_mask = cv2.cvtColor(rgb_mask,cv2.COLOR_RGB2BGR)            
#     cv2.imwrite("D:/FHJ/Thesis/Giacomo/G05 conflev real test masks rgb/" + mask + '.png', rgb_mask)


############################################################################
#CONFLEV 1 - croppings
##############################################################################


# color_path_img = "D:/FHJ/Thesis/Giacomo/05 - dust masks/images/"
# color_path_mask = "D:/FHJ/Thesis/Giacomo/05 - dust masks/masks/"
# color_path = "E:/color/color masks RGB/"

# save_path = "E:/color/conv4/"

# x = 752
# y = 1038
# w = 418
# h = 502

# day = "19"


# def make_pixel_green(pixel):
#     pixel[0] = 0
#     pixel[1] = 255
#     pixel[2] = 0
    
# def make_pixel_blue(pixel):
#     pixel[0] = 0
#     pixel[1] = 0
#     pixel[2] = 255
    
# def make_pixel_yellow(pixel):
#     pixel[0] = 255
#     pixel[1] = 255
#     pixel[2] = 0
    
# def make_pixel_red(pixel):
#     pixel[0] = 255
#     pixel[1] = 0
#     pixel[2] = 0


# for folder in next(os.walk(color_path_img))[1]:
#     if folder == "G":
#     #os.mkdir(save_path + folder)
#         for subfolder in next(os.walk(color_path_img + folder))[1]:
#             if subfolder == "G05":
#             #os.mkdir(save_path + folder + "/" + subfolder)
#                 for mask in next(os.walk(color_path_img + folder + "/" + subfolder))[2]:
#                     if mask == "G05_day" + str(day) + ".png":
#                         img = cv2.imread(color_path_img + folder + "/" + subfolder + "/" + mask)
#                         img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#                         img = np.array(img)
                        
#                         img_cropped = np.full((h, w, 3), 0)
#                         img_cropped = img[y -1:y + h,x-1:x+w,:]
                        
#                         img2 = img_cropped.copy()
                        
#                         mask_l = cv2.imread(color_path_mask + folder + "/" + subfolder + "/" + mask, 0)
#                         mask_l = np.array(mask_l)
                        
#                         mask_cropped = np.full((h, w, 1), 0)
#                         mask_cropped = mask_l[y -1:y + h,x-1:x+w]
                        
#                         color_mask = cv2.imread(color_path + "data_" + folder + "/" + subfolder + "/" + mask + ".png")
#                         color_mask = cv2.cvtColor(color_mask,cv2.COLOR_RGB2BGR)
#                         color_mask = np.array(color_mask)
                        
#                         color_mask_cropped = np.full((h, w, 3), 0)
#                         color_mask_cropped = color_mask[y -1:y + h,x-1:x+w,:]
                        
#                         border_mask = np.full((h, w, 1), 0)
                        
#                         for i in range(w):
#                             for j in range(h):
#                                 if mask_cropped[j, i] == 255 and mask_cropped[j-1, i] == 0 and j != 0:
#                                     border_mask[j-1, i] = 255 
                                    
#                                     for k in range(1,11):
#                                         #make_pixel_green(img_cropped[j - k, i])
#                                         #make_pixel_blue(img_cropped[j - k, i])
#                                         #make_pixel_yellow(img_cropped[j - k, i])
#                                         make_pixel_red(img_cropped[j - k, i])
                                    
#                                 if mask_cropped[j, i] == 0 and mask_cropped[j-1, i] == 255:
#                                     border_mask[j, i] = 255 
                                    
#                                     for k in range(10):
#                                         #make_pixel_green(img_cropped[j + k, i])
#                                         #make_pixel_blue(img_cropped[j + k, i])
#                                         #make_pixel_yellow(img_cropped[j + k, i])
#                                         make_pixel_red(img_cropped[j + k, i])
                                    
                                    
#                         for j in range(h):
#                             for i in range(w):
#                                 if mask_cropped[j, i] == 255 and mask_cropped[j, i-1] == 0 and i != 0:
#                                     border_mask[j, i-1] = 255 
                                    
                                    
#                                     for k in range(1, 11):
#                                         #make_pixel_green(img_cropped[j, i - k])
#                                         #make_pixel_blue(img_cropped[j, i - k])
#                                         #make_pixel_yellow(img_cropped[j, i - k])
#                                         make_pixel_red(img_cropped[j, i - k])
                                    
#                                 if mask_cropped[j, i] == 0 and mask_cropped[j, i-1] == 255:
#                                     border_mask[j, i] = 255 
                                    
#                                     for k in range(10):
#                                         #make_pixel_green(img_cropped[j, i + k])
#                                         #make_pixel_blue(img_cropped[j, i + k])
#                                         #make_pixel_yellow(img_cropped[j, i + k])
#                                         make_pixel_red(img_cropped[j, i + k])
                                    
#                         plt.figure(figsize=(15, 10))
#                         plt.subplot(1, 3, 1)
#                         plt.title('Cropped Image')
#                         plt.imshow(img2)
#                         plt.subplot(1, 3, 2)
#                         plt.title('Colored Mask')
#                         plt.imshow(color_mask_cropped)
#                         plt.xticks([])  # remove x-axis values
#                         plt.yticks([])  # remove y-axis values
#                         plt.subplot(1, 3, 3)
#                         plt.title('Image with border of dust storm')
#                         plt.imshow(img_cropped)
#                         plt.xticks([])  # remove x-axis values
#                         plt.yticks([])  # remove y-axis values
#                         plt.savefig(save_path + subfolder + "_day" + str(day) + "_conv2.png")
#                         plt.show()
                        
#                         cv2.imwrite(save_path + subfolder + "_day" + str(day) + "_conv4_img.png", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
#                         cv2.imwrite(save_path + subfolder + "_day" + str(day) + "_conv4_colored_mask.png", cv2.cvtColor(color_mask_cropped, cv2.COLOR_RGB2BGR))
#                         cv2.imwrite(save_path + subfolder + "_day" + str(day) + "_conv4_mask.png", mask_cropped)
#                         cv2.imwrite(save_path + subfolder + "_day" + str(day) + "_conv4_border_mask.png", border_mask)
#                         cv2.imwrite(save_path + subfolder + "_day" + str(day) + "_conv4_img_with_mask_border.png", cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))



def is_pixel_green(pixel):
    if pixel[0] == 0 and pixel[1] == 255 and pixel[2] == 0:
        return True
    else:
        return False
    
def is_pixel_blue(pixel):
    if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
        return True
    else:
        return False
    
def is_pixel_yellow(pixel):
    if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 0:
        return True
    else:
        return False
    
def is_pixel_red(pixel):
    if pixel[0] == 255 and pixel[1] == 0 and pixel[2] == 0:
        return True
    else:
        return False


colored_mask_directory = 'E:/color/color masks RGB/'

# create an empty dataframe to store the IOU values
dust_df = pd.DataFrame(columns=["Day", "Repository", "Black", "Green", "Blue", "Yellow", "Red"])

for folder in next(os.walk(colored_mask_directory))[1]:
    path = colored_mask_directory + folder
    for subfolder in next(os.walk(path))[1]:
        path2 = path + "/" + subfolder
        for colored_mask in next(os.walk(path2))[2]:
            img = cv2.imread(path2 + "/" + colored_mask)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            img = np.array(img)
            
            green = 0
            blue = 0
            yellow = 0
            red = 0    
            black = 0
            
            for i in range(3600):
                print(i,"\n")
                for j in range(1801):
                    if is_pixel_green(img[j, i]):
                        green += 1
                        
                    elif is_pixel_blue(img[j, i]):
                        blue += 1
                        
                    elif is_pixel_yellow(img[j, i]):
                        yellow += 1
                        
                    elif is_pixel_red(img[j, i]):
                        red += 1
                        
                    else:
                        black += 1
            
                        
            dust_df.loc[len(dust_df)] = [colored_mask[7:9], subfolder, black, green, blue, yellow, red]


df = pd.read_csv('H:/color/dust.csv')

df['Total Pixels'] = df[['Black', 'Green', 'Blue', 'Yellow', 'Red']].sum(axis=1)

df['Total Dust Pixels'] = df[['Green', 'Blue', 'Yellow', 'Red']].sum(axis=1)

df['Total Pixels'].unique()

df['Folder'] = df['Repository'].str[0]


df.to_csv('D:/FHJ/Thesis/mdssd.csv', index=False)

top_rows = df.nlargest(20, 'Total Dust Pixels')['cow id'].tolist()


# path = "D:/peyman/color/color masks/"

# save_path = "D:/peyman/color/color masks RGB/"

# for folder in next(os.walk(path))[1]:
#     os.mkdir(save_path + folder)
#     for subfolder in next(os.walk(path + folder))[1]:
#         os.mkdir(save_path + folder + "/" + subfolder)
#         for mask in next(os.walk(path + folder + "/" + subfolder))[2]:
#             img = cv2.imread(path + folder + "/" + subfolder + "/" + mask, 0)
#             rgb_mask = np.full((1801, 3600, 3), 0)
        
#             for i in range(3600):
#                 print(i)
#                 print(i)
#                 for j in range(1801):
#                     if img[j, i] == 4:
#                         rgb_mask[j, i, 1] = 255 # green
                        
#                     elif img[j, i] == 3:
#                         rgb_mask[j, i, 2] = 255 # blue
                        
#                     elif img[j, i] == 2:
#                         rgb_mask[j, i, 0] = 255 # yellow
#                         rgb_mask[j, i, 1] = 255 # yellow
                        
#                     elif img[j, i] == 1:
#                         rgb_mask[j, i, 0] = 255 # red
            
#             rgb_mask = np.float32(rgb_mask)
#             rgb_mask = cv2.cvtColor(rgb_mask,cv2.COLOR_RGB2BGR)            
#             cv2.imwrite(save_path + folder + "/" + subfolder + "/" + mask + '.png', rgb_mask)
    


####################################################################################

# TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/04 - cloud/clouds/"

# cnt = 0
# for folder in next(os.walk(TRAIN_PATH))[1]:
#     for sub_folder in next(os.walk(TRAIN_PATH + folder))[1]:
#         path = TRAIN_PATH + folder + "/" + sub_folder + "/mdgms/"
#         for img_file in next(os.walk(path))[2]:
#             img = cv2.imread(path + img_file) 
#             img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#             new_img = img[120:1400, 8:3592, :]
#             patches = patchify(new_img, (256, 256, 3), step=256)  #Step=256 for 256 patches means no overlap
#             for i in range(0, 5):
#                 for j in range(0, 14):
#                     image = np.full((1280, 3584, 3), 0)
#                     image = patches[i, j, :, :, :, :] 
#                     image = image.squeeze()
#                     cv2.imwrite('D:/FHJ/Thesis/Giacomo/04 - cloud/clouds/cloud_img_patches/' + 'image' + str(cnt) + '.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#                     cnt += 1
           

# for folder in next(os.walk(TRAIN_PATH))[1]:
#     for sub_folder in next(os.walk(TRAIN_PATH + folder))[1]:
#         path = TRAIN_PATH + folder + "/" + sub_folder + "/cloudmask/"
#         save_path_mask = TRAIN_PATH + folder + "/" + sub_folder + "/masks"
#         os.mkdir(save_path_mask)
#         for img_file in next(os.walk(path))[2]:
#             img = rioxarray.open_rasterio(path + img_file) 
#             img = np.array(img).squeeze()
#             img = np.flip(img, axis = 0)
#             img[img > 0] = 255
#             img[img == -999] = 0
#             cv2.imwrite(save_path_mask + "/" + img_file[0:(len(img_file)-5)] + ".png", img)
            
           
# cnt = 0
# for folder in next(os.walk(TRAIN_PATH))[1]:
#     for sub_folder in next(os.walk(TRAIN_PATH + folder))[1]:
#         path = TRAIN_PATH + folder + "/" + sub_folder + "/masks/"
#         for img_file in next(os.walk(path))[2]:
#             img = cv2.imread(path + img_file, 0) 
#             #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#             new_img = img[120:1400, 8:3592]
#             patches = patchify(new_img, (256, 256), step=256)  #Step=256 for 256 patches means no overlap
#             for i in range(0, 5):
#                 for j in range(0, 14):
#                     mask = np.full((1280, 3584), 0)
#                     mask = patches[i, j, :, :] 
#                     mask = mask.squeeze()
#                     cv2.imwrite('D:/FHJ/Thesis/Giacomo/04 - cloud/cloud_mask_patches/' + 'mask' + str(cnt) + '.png', mask)
#                     cnt += 1
           

# test_fraction = 0.1
# test_idxs = random.sample(range(0, 49770), math.ceil(test_fraction*49770))

# IMG_PATH = "D:/FHJ/Thesis/Giacomo/04 - cloud/cloud_img_patches"

# for i in range(0, 49770):
#     src_path = IMG_PATH + "/image" + str(i) + ".png"
#     if i not in test_idxs:
#         dst_path = "D:/FHJ/Thesis/Giacomo/04 - cloud/train_cloud_img_patches" + "/image" + str(i) + ".png"
#     else:
#         dst_path = "D:/FHJ/Thesis/Giacomo/04 - cloud/test_cloud_img_patches" + "/image" + str(i) + ".png"
#     shutil.copy(src_path, dst_path)
    

# MASK_PATH = "D:/FHJ/Thesis/Giacomo/04 - cloud/cloud_mask_patches"

# for i in range(0, 49770):
#     src_path = MASK_PATH + "/mask" + str(i) + ".png"
#     if i not in test_idxs:
#         dst_path = "D:/FHJ/Thesis/Giacomo/04 - cloud/train_cloud_mask_patches" + "/mask" + str(i) + ".png"
#     else:
#         dst_path = "D:/FHJ/Thesis/Giacomo/04 - cloud/test_cloud_mask_patches" + "/mask" + str(i) + ".png"
#     shutil.copy(src_path, dst_path)
    
    
# ##############################################################################
# # SUB_DATA
# ##############################################################################
# sub_data_idxs = random.sample(range(0, 49770), 10000)

# for i in range(0, 49770):
#     if i in sub_data_idxs:
#         src_path = IMG_PATH + "/image" + str(i) + ".png"
#         dst_path = "D:/FHJ/Thesis/Giacomo/04 - cloud/sub_data/train_img" + "/image" + str(i) + ".png"
#         shutil.copy(src_path, dst_path)
        
        
# for i in range(0, 49770):
#     if i in sub_data_idxs:
#         src_path = MASK_PATH + "/mask" + str(i) + ".png"
#         dst_path = "D:/FHJ/Thesis/Giacomo/04 - cloud/sub_data/train_mask" + "/mask" + str(i) + ".png"
#         shutil.copy(src_path, dst_path)
        


    
# cnt = 0
# for i in range(0, 49770):
#     if i not in sub_data_idxs:
#         src_path = IMG_PATH + "/image" + str(i) + ".png"
#         dst_path = "D:/FHJ/Thesis/Giacomo/04 - cloud/sub_data/test_img" + "/image" + str(i) + ".png"
#         shutil.copy(src_path, dst_path)
#         cnt +=1
        
#     if cnt == 1000:
#         break
        
        
# cnt = 0
# for i in range(0, 49770):
#     if i not in sub_data_idxs:
#         src_path = MASK_PATH + "/mask" + str(i) + ".png"
#         dst_path = "D:/FHJ/Thesis/Giacomo/04 - cloud/sub_data/test_mask" + "/mask" + str(i) + ".png"
#         shutil.copy(src_path, dst_path)
#         cnt +=1
        
#     if cnt == 1000:
#         break


###########################################################################
# B02 - day01-day32 DUST IMAGES - patching 256 x 256
#############################################################################

# TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/05 - dust masks/images/B/B02/"
# PATCH_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/B02_256_patches/img_patches/'

# patch_size = 256

# p = next(os.walk(TRAIN_PATH))
# for img_file in next(os.walk(TRAIN_PATH))[2]:
#     path = TRAIN_PATH + img_file
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#     for i in range(7):   #Steps of 256
#         for j in range(14):  #Steps of 256
#             single_patch = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size:]
#             cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))

# TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/05 - dust masks/masks/B/B02/"
# PATCH_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/B02_256_patches/mask_patches/'

# patch_size = 256

# p = next(os.walk(TRAIN_PATH))
# for img_file in next(os.walk(TRAIN_PATH))[2]:
#     path = TRAIN_PATH + img_file
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#     for i in range(7):   #Steps of 256
#         for j in range(14):  #Steps of 256
#             single_patch = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size:]
#             cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))




###########################################################################
#Test dust G05 - patching 512x512
#############################################################################

TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/05 - dust masks/G05 512 patch size/images/"
PATCH_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/G05 512 patch size/img_patches/'

patch_size = 512

#p = next(os.walk(TRAIN_PATH + folder))

#cnt = 0
for folder in next(os.walk(TRAIN_PATH))[1]:
    for img_file in next(os.walk(TRAIN_PATH + folder))[2]:
        #cnt +=1
        path = TRAIN_PATH + folder + "/" + img_file
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for i in range(4):   #Steps of 512
            for j in range(7):  #Steps of 512
                if j != 6:
                    if i != 3:
                        single_patch = img[i*patch_size:patch_size*i+patch_size, j*patch_size:(j+1)*patch_size:]
                        cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
                    if i == 3:
                        single_patch = img[1289:, j*patch_size:(j+1)*patch_size:]
                        cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))

                if j == 6:
                    if i != 3:
                        single_patch = img[i*patch_size:patch_size*i+patch_size, 3088::]
                        cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
                    if i == 3:
                        single_patch = img[1289:, 3088::]
                        cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))


TRAIN_PATH = "D:/FHJ/Thesis/Giacomo/05 - dust masks/G05 512 patch size/masks/"
PATCH_PATH = 'D:/FHJ/Thesis/Giacomo/05 - dust masks/G05 512 patch size/mask_patches/'

#patch_size = 512

#p = next(os.walk(TRAIN_PATH + folder))

#cnt = 0
for folder in next(os.walk(TRAIN_PATH))[1]:
    for img_file in next(os.walk(TRAIN_PATH + folder))[2]:
        #cnt +=1
        path = TRAIN_PATH + folder + "/" + img_file
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for i in range(4):   #Steps of 512
            for j in range(7):  #Steps of 512
                if j != 6:
                    if i != 3:
                        single_patch = img[i*patch_size:patch_size*i+patch_size, j*patch_size:(j+1)*patch_size:]
                        cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
                    if i == 3:
                        single_patch = img[1289:, j*patch_size:(j+1)*patch_size:]
                        cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))

                if j == 6:
                    if i != 3:
                        single_patch = img[i*patch_size:patch_size*i+patch_size, 3088::]
                        cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
                    if i == 3:
                        single_patch = img[1289:, 3088::]
                        cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))


