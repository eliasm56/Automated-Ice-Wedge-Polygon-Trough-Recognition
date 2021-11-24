#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2, os
import numpy as np
from tqdm import tqdm
from skimage.io import imshow, imread
import matplotlib.pyplot as plt


# In[ ]:


# Iterate through training images
train_img = next(os.walk('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_trainimg/original/'))[2]
print("No. of images = ", len(train_img))

val_img = next(os.walk('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_valimg/'))[2]
print("No. of images = ", len(val_img))

test_img = next(os.walk('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_testimg/'))[2]
print("No. of images = ", len(test_img))

# Create structural element for morphological operations
kernel = np.ones((5,5),np.uint8)


# In[ ]:


# Perform morphological erosion operation on training images

for n, id_ in tqdm(enumerate(train_img), total=len(train_img)):
    img = imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_trainimg/original/'+id_, 0)
    erosion = cv2.erode(img, kernel)
    final = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    imshow(final)
    plt.show()
    cv2.imwrite('C:/Users/manos/Desktop/unetResearch/trough/dataset2/erosion_trainimg/original/png/' + str(id_), final)


# In[ ]:


# Perform morphological erosion operation on validation images

for n, id_ in tqdm(enumerate(val_img), total=len(val_img)):
    img = imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_valimg/'+id_, 0)
    erosion = cv2.erode(img, kernel)
    final = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    imshow(final)
    plt.show()
    cv2.imwrite('C:/Users/manos/Desktop/unetResearch/trough/dataset2/erosion_valimg/png/' + str(id_), final)


# In[ ]:


# Perform morphological erosion operation on test images

for n, id_ in tqdm(enumerate(test_img), total=len(test_img)):
    img = imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_valimg/'+id_, 0)
    erosion = cv2.erode(img, kernel)
    final = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    imshow(final)
    plt.show()
    cv2.imwrite('C:/Users/manos/Desktop/unetResearch/trough/dataset2/erosion_testimg/png/' + str(id_), final)


# In[ ]:


# Perform morphological black hat operation on train images

for n, id_ in tqdm(enumerate(train_img), total=len(train_img)):
    img = imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_trainimg/original/'+id_, 0)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    final = cv2.cvtColor(blackhat, cv2.COLOR_BGR2GRAY)
    imshow(final)
    plt.show()
    cv2.imwrite('C:/Users/manos/Desktop/unetResearch/trough/dataset2/blackhat_trainimg/original/png/' + str(id_), final)


# In[ ]:


# Perform morphological black hat operation on validation images

for n, id_ in tqdm(enumerate(val_img), total=len(val_img)):
    img = imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_valimg/'+id_, 0)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    final = cv2.cvtColor(blackhat, cv2.COLOR_BGR2GRAY)
    imshow(final)
    plt.show()
    cv2.imwrite('C:/Users/manos/Desktop/unetResearch/trough/dataset2/blackhat_valimg/png/' + str(id_), final)


# In[ ]:


# Perform morphological black hat operation on test images

kernel = np.ones((5,5),np.uint8)

for n, id_ in tqdm(enumerate(test_img), total=len(test_img)):
    img = imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_testimg/'+id_, 0)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    final = cv2.cvtColor(blackhat, cv2.COLOR_BGR2GRAY)
    imshow(final)
    plt.show()
    cv2.imwrite('C:/Users/manos/Desktop/unetResearch/trough/dataset2/blackhat_testimg/png/' + str(id_), final)


# In[ ]:




