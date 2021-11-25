# Lightweight-U-Net-for-Satellite-Image-Segmentation
## About
This repository contains Python code for U-Net based-semantic segmentation (**in Keras**) of satellite images, aided by data augmentation and morphological filtering. The scripts were developed for a research study on automated recognition of ice-wedge polygon troughs in the Arctic permafrost landscapes using commercial satellite imagery. Experimental results showed how: (1) basic spatial data augmentations (e.g., horizontal/vertical flipping, rotation, transposition) can overcome the constraints placed on deep learning models by limited training data (2) morphological filtering (e.g., erosion, blackhat) can exploit the geometric structure of image features to aid CNNs in learning accurate feature representations.

### Example of results
(Left: original 3-channel satellite image | Right: mask predicted by U-Net)

<img src="https://user-images.githubusercontent.com/77365021/143181377-90d31669-15fa-4797-97e2-ee5f0a544647.png" width="400">     <img src="https://user-images.githubusercontent.com/77365021/143181383-6bf2f5a6-c5a3-4640-9cb5-f5196a99ee74.png" width="400">

## Code
Five scripts are included. Please change the paths to your data, since they are the paths related to my directory:
```
1) unet_model.py: Contains function for U-Net model built in Keras.
2) train_unet.py: Contains full training/validation/testing pipeline with data loaders, as well as code for detailed scikit-learn segmentation metrics and a confusion matrix.
3) data_augmentor.py: Contains image data augmentation pipeline using the Albumentations library.
4) preprocessing.py: Contains image preproessing pipeline for application of morphological filters on input imagery using OpenCV.
5) predict.py: Contains inference pipeline for use of trained U-Net in inference mode (predicting on unseen imagery).
```

## Data
Since the original data is commercially-licensed, it cannot be shared in this repository for legal reasons. However, this pipeline should work with any satellite imagery/masks as long as input data is split into tiles. Your dataset directory should look similar to the following:
```   
dataset
└───training_images
│   │   img_1.png
│   │   ...
│
└───training_masks 
│   |  mask_1.png      
|   |  ...
│   
└───validation_images
|    │  img_1.png
|    │  ...
|
└───validation_masks 
|    |  mask_1.png
|    |  ...
| 
└───test_images
|   |  img_1.png
|   |  ...
|
└───test_masks
|   |  mask_1.png
|   |  ...
|
```

## Workflow
In the case of this approach, preprocessing.py should be used to apply the desired morphological filters on the oiginal image patches first, then data_augmentor.py should be used first to obtain the desired number of augmented images from the preprocessed images. However, augmentation should only be applied on training images/masks. Offline augmentation is used in this case, rather than online augmentation, meaning augmentation is applied before training as opposed to applying augmentations on images as they are being fed to the model in real time. The dataset for the original project contained very few images, so synthesizing new images beforehand allowed for better performance and was not constrained by performance. The workflow for this U-Net-based segmentation approach with augmentation and morphological filtering looks like this:
![RCOP_workflow](https://user-images.githubusercontent.com/77365021/143179726-b8a85806-3876-467e-942a-faf63ddc7620.png)

## U-Net
The model is an efficient vanilla U-Net with dropout and batch normalization built in Keras. Input channels, intial number of filters, and filter growth factor can be adjusted in train_unet.py. A GPU was employed in the original project, with code included for GPU configuration in train_unet.py (which can be removed if not using GPU). 
```
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 256, 256, 3) 0                                            
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 256, 256, 32) 896         input_2[0][0]                    
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 256, 256, 32) 128         conv2d_19[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_18 (TFOpLambda)      (None, 256, 256, 32) 0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 256, 256, 32) 0           tf.nn.relu_18[0][0]              
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 256, 256, 32) 9248        dropout_9[0][0]                  
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 256, 256, 32) 128         conv2d_20[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_19 (TFOpLambda)      (None, 256, 256, 32) 0           batch_normalization_19[0][0]     
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 128, 128, 32) 0           tf.nn.relu_19[0][0]              
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 128, 128, 64) 18496       max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 128, 128, 64) 256         conv2d_21[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_20 (TFOpLambda)      (None, 128, 128, 64) 0           batch_normalization_20[0][0]     
__________________________________________________________________________________________________
dropout_10 (Dropout)            (None, 128, 128, 64) 0           tf.nn.relu_20[0][0]              
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 128, 128, 64) 36928       dropout_10[0][0]                 
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 128, 128, 64) 256         conv2d_22[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_21 (TFOpLambda)      (None, 128, 128, 64) 0           batch_normalization_21[0][0]     
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 64, 64, 64)   0           tf.nn.relu_21[0][0]              
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 64, 64, 128)  73856       max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 64, 64, 128)  512         conv2d_23[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_22 (TFOpLambda)      (None, 64, 64, 128)  0           batch_normalization_22[0][0]     
__________________________________________________________________________________________________
dropout_11 (Dropout)            (None, 64, 64, 128)  0           tf.nn.relu_22[0][0]              
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 64, 64, 128)  147584      dropout_11[0][0]                 
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 64, 64, 128)  512         conv2d_24[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_23 (TFOpLambda)      (None, 64, 64, 128)  0           batch_normalization_23[0][0]     
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 32, 32, 128)  0           tf.nn.relu_23[0][0]              
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 32, 32, 256)  295168      max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 32, 32, 256)  1024        conv2d_25[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_24 (TFOpLambda)      (None, 32, 32, 256)  0           batch_normalization_24[0][0]     
__________________________________________________________________________________________________
dropout_12 (Dropout)            (None, 32, 32, 256)  0           tf.nn.relu_24[0][0]              
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 32, 32, 256)  590080      dropout_12[0][0]                 
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 32, 32, 256)  1024        conv2d_26[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_25 (TFOpLambda)      (None, 32, 32, 256)  0           batch_normalization_25[0][0]     
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 16, 16, 256)  0           tf.nn.relu_25[0][0]              
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 16, 16, 512)  1180160     max_pooling2d_7[0][0]            
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 16, 16, 512)  2048        conv2d_27[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_26 (TFOpLambda)      (None, 16, 16, 512)  0           batch_normalization_26[0][0]     
__________________________________________________________________________________________________
dropout_13 (Dropout)            (None, 16, 16, 512)  0           tf.nn.relu_26[0][0]              
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 16, 16, 512)  2359808     dropout_13[0][0]                 
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 16, 16, 512)  2048        conv2d_28[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_27 (TFOpLambda)      (None, 16, 16, 512)  0           batch_normalization_27[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_4 (Conv2DTrans (None, 32, 32, 256)  524544      tf.nn.relu_27[0][0]              
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 32, 32, 512)  0           conv2d_transpose_4[0][0]         
                                                                 tf.nn.relu_25[0][0]              
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 32, 32, 256)  1179904     concatenate_4[0][0]              
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 32, 32, 256)  1024        conv2d_29[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_28 (TFOpLambda)      (None, 32, 32, 256)  0           batch_normalization_28[0][0]     
__________________________________________________________________________________________________
dropout_14 (Dropout)            (None, 32, 32, 256)  0           tf.nn.relu_28[0][0]              
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 32, 32, 256)  590080      dropout_14[0][0]                 
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 32, 32, 256)  1024        conv2d_30[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_29 (TFOpLambda)      (None, 32, 32, 256)  0           batch_normalization_29[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_5 (Conv2DTrans (None, 64, 64, 128)  131200      tf.nn.relu_29[0][0]              
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 64, 64, 256)  0           conv2d_transpose_5[0][0]         
                                                                 tf.nn.relu_23[0][0]              
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 64, 64, 128)  295040      concatenate_5[0][0]              
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 64, 64, 128)  512         conv2d_31[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_30 (TFOpLambda)      (None, 64, 64, 128)  0           batch_normalization_30[0][0]     
__________________________________________________________________________________________________
dropout_15 (Dropout)            (None, 64, 64, 128)  0           tf.nn.relu_30[0][0]              
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 64, 64, 128)  147584      dropout_15[0][0]                 
__________________________________________________________________________________________________
batch_normalization_31 (BatchNo (None, 64, 64, 128)  512         conv2d_32[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_31 (TFOpLambda)      (None, 64, 64, 128)  0           batch_normalization_31[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_6 (Conv2DTrans (None, 128, 128, 64) 32832       tf.nn.relu_31[0][0]              
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 128, 128, 128 0           conv2d_transpose_6[0][0]         
                                                                 tf.nn.relu_21[0][0]              
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 128, 128, 64) 73792       concatenate_6[0][0]              
__________________________________________________________________________________________________
batch_normalization_32 (BatchNo (None, 128, 128, 64) 256         conv2d_33[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_32 (TFOpLambda)      (None, 128, 128, 64) 0           batch_normalization_32[0][0]     
__________________________________________________________________________________________________
dropout_16 (Dropout)            (None, 128, 128, 64) 0           tf.nn.relu_32[0][0]              
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 128, 128, 64) 36928       dropout_16[0][0]                 
__________________________________________________________________________________________________
batch_normalization_33 (BatchNo (None, 128, 128, 64) 256         conv2d_34[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_33 (TFOpLambda)      (None, 128, 128, 64) 0           batch_normalization_33[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_7 (Conv2DTrans (None, 256, 256, 32) 8224        tf.nn.relu_33[0][0]              
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 256, 256, 64) 0           conv2d_transpose_7[0][0]         
                                                                 tf.nn.relu_19[0][0]              
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 256, 256, 32) 18464       concatenate_7[0][0]              
__________________________________________________________________________________________________
batch_normalization_34 (BatchNo (None, 256, 256, 32) 128         conv2d_35[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_34 (TFOpLambda)      (None, 256, 256, 32) 0           batch_normalization_34[0][0]     
__________________________________________________________________________________________________
dropout_17 (Dropout)            (None, 256, 256, 32) 0           tf.nn.relu_34[0][0]              
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 256, 256, 32) 9248        dropout_17[0][0]                 
__________________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, 256, 256, 32) 128         conv2d_36[0][0]                  
__________________________________________________________________________________________________
tf.nn.relu_35 (TFOpLambda)      (None, 256, 256, 32) 0           batch_normalization_35[0][0]     
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 256, 256, 1)  33          tf.nn.relu_35[0][0]              
==================================================================================================
Total params: 7,771,873
Trainable params: 7,765,985
Non-trainable params: 5,888
__________________________________________________________________________________________________
```
