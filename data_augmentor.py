#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import albumentations as A
import imageio as io
import random


# In[ ]:


# Define input/output paths for data augmentation
im1_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_trainimg/original'
im2_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/ndvi_trainimg/original/png'
im3_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/ndwi_trainimg/original/png'
im4_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/highpass_trainimg/original/png'
im5_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/blackhat_trainimg/original/png'
im6_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/erosion_trainimg/original/png'
im7_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/gray_trainimg/original'
masks_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/train_masks/original'
im1_augmented_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_trainimg/aug/all'
im2_augmented_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/ndvi_trainimg/aug/all'
im3_augmented_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/ndwi_trainimg/aug/all'
im4_augmented_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/highpass_trainimg/aug/all'
im5_augmented_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/blackhat_trainimg/aug/all'
im6_augmented_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/erosion_trainimg/aug/all'
im7_augmented_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/gray_trainimg/aug/all'
mask_augmented_path = 'C:/Users/manos/Desktop/unetResearch/trough/dataset2/train_masks/aug/all'
im1 = []
im2 = []
im3 = []
im4 = []
im5 = []
im6 = []
im7 = []
im8 = []
im9 = []
im10 = []
masks = []

# Append original training images and mask to new arrays of data that will be augmented
for im in os.listdir(im1_path):
    im1.append(os.path.join(im1_path, im))

for im in os.listdir(im2_path):
    im2.append(os.path.join(im2_path, im))
    
for im in os.listdir(im3_path):
    im3.append(os.path.join(im3_path, im))

for im in os.listdir(im4_path):
    im4.append(os.path.join(im4_path, im))

for im in os.listdir(im5_path):
    im5.append(os.path.join(im5_path, im))

for im in os.listdir(im6_path):
    im6.append(os.path.join(im6_path, im))
    
for im in os.listdir(im7_path):
    im7.append(os.path.join(im7_path, im))
    
for mask in os.listdir(masks_path):
    masks.append(os.path.join(masks_path, mask))
    
# Define transformations that will be performed in augmentation
aug = A.Compose([
                A.HorizontalFlip(p=0.8)
                A.VerticalFlip(p=0.8)
                A.RandomRotate90(p=0.8)
                A.Transpose(p=0.8),
                ],
                additional_targets={
                    'image2':'image',
                    'image3':'image',
                    'image4':'image',
                    'image5':'image',
                    'image6':'image',
                    'image7':'image',
                })


# In[ ]:


# Generate augmented training images/masks
images_to_generate = 90

i=1

while i<=images_to_generate:
    number = random.randint(0, len(im1)-1)
    img1 = im1[number]
    img2 = im2[number]
    img3 = im3[number]
    img4 = im4[number]
    img5 = im5[number]
    img6 = im6[number]
    img7 = im7[number]
    mask = masks[number]
    print(img1, img2, img3, img4, img5, img6, img7, mask)
    
    original_im1 = io.imread(img1)
    original_im2 = io.imread(img2)
    original_im3 = io.imread(img3)
    original_im4 = io.imread(img4)
    original_im5 = io.imread(img5)
    original_im6 = io.imread(img6)
    original_im7 = io.imread(img7)
    original_mask = io.imread(mask)
    
    augmented = aug(image=original_im1,
                    image2=original_im2,
                    image3=original_im3,
                    image4=original_im4,
                    image5=original_im5,
                    image6=original_im6,
                    image7=original_im7,
                    mask=original_mask)
    transformed_im1=augmented['image']
    transformed_im2=augmented['image2']
    transformed_im3=augmented['image3']
    transformed_im4=augmented['image4']
    transformed_im5=augmented['image5']
    transformed_im6=augmented['image6']
    transformed_im7=augmented['image7']
    transformed_mask=augmented['mask']
    
    new_im1_path= "%s/augmented_image_%s.png" %(im1_augmented_path, i)
    new_im2_path= "%s/augmented_image_%s.png" %(im2_augmented_path, i)
    new_im3_path= "%s/augmented_image_%s.png" %(im3_augmented_path, i)
    new_im4_path= "%s/augmented_image_%s.png" %(im4_augmented_path, i)
    new_im5_path= "%s/augmented_image_%s.png" %(im5_augmented_path, i)
    new_im6_path= "%s/augmented_image_%s.png" %(im6_augmented_path, i)
    new_im7_path= "%s/augmented_image_%s.png" %(im7_augmented_path, i)
    new_mask_path= "%s/augmented_mask_%s.png" %(mask_augmented_path, i)
    io.imsave(new_im1_path, transformed_im1)
    io.imsave(new_im2_path, transformed_im2)
    io.imsave(new_im3_path, transformed_im3)
    io.imsave(new_im4_path, transformed_im4)
    io.imsave(new_im5_path, transformed_im5)
    io.imsave(new_im6_path, transformed_im6)
    io.imsave(new_im7_path, transformed_im7)
    io.imsave(new_mask_path, transformed_mask)
    i= i+1


# In[ ]:




