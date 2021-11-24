#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import tensorflow as tf, keras
from tensorflow.compat.v1.keras.backend import set_session
from keras import backend as K
import cv2, os, sys, random, numpy as np, itertools
from tqdm import tqdm 
from skimage.io import imshow, imread
import matplotlib.pyplot as plt
from unet_model import *
from sklearn.metrics import *
from ImageDataAugmentor.image_data_augmentor import *
import albumentations as A
import tifffile as tiff
from natsort import natsorted

get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
import segmentation_models as sm


# In[2]:


curr_session = tf.compat.v1.get_default_session()
# close current session
if curr_session is not None:
    curr_session.close()
# reset graph
K.clear_session()
# create new session
s = tf.compat.v1.InteractiveSession()
set_session(s)

# Configure GPU for use
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)


# In[5]:


# Iterate through images
img = natsorted(next(os.walk('C:/Users/manos/Desktop/unetResearch/trough/time_series/1949_img/lowpass_gray'))[2])
print("No. of images = ", len(img))

# Define parameters for image/mask resizing
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

# Create new arrays to store augmented training images/masks
X = np.zeros((len(img), IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)


# In[6]:


# Load images
for n, id_ in tqdm(enumerate(img), total=len(img)):
    # Resize
    img = cv2.imread('C:/Users/manos/Desktop/unetResearch/trough/time_series/1949_img/lowpass_gray/'+id_, 0)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    # Save images
    X[n] = img/255


# In[7]:


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

dependencies = {
    'dice_coef': dice_coef
}


# In[8]:


model = keras.models.load_model("C:/Users/manos/Desktop/unetResearch/trough/code/gray_trough.h5", custom_objects=dependencies)


# In[9]:


preds_test = model.predict(X, batch_size = 1, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.float32)

for n, id_ in tqdm(enumerate(X), total=len(X)):
    
    # Display image
    #imshow(np.squeeze(X[n]))
    #plt.show()
    # Display predicted mask
    #imshow(np.squeeze(preds_test_t[n]))
    #plt.show()
    # Save each predicted mask
    name = 'C:/Users/manos/Desktop/unetResearch/trough/time_series/1949_preds/lowpass/img_' + str(n) + '.png'
    cv2.imwrite(name, np.squeeze(preds_test_t[n]))


# In[ ]:




