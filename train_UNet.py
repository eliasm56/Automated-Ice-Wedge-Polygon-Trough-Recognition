# PREPARATION



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

get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
import segmentation_models as sm





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


# DATA AUGMENTATION AND PREPROCESSING




# Iterate through training images
train_img = next(os.walk('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_trainimg/aug/all'))[2]
print("No. of images = ", len(train_img))

# Iterate through training masks
train_masks = next(os.walk('C:/Users/manos/Desktop/unetResearch/trough/dataset2/train_masks/aug/all'))[2]
print("No. of masks = ", len(train_masks))

# Iterate through validation images
val_img = next(os.walk('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_valimg/'))[2]
print("No. of images = ", len(val_img))

# Iterate through validation masks
val_masks = next(os.walk('C:/Users/manos/Desktop/unetResearch/trough/dataset2/val_masks/png/'))[2]
print("No. of masks = ", len(val_masks))

# Iterate through testing images
test_img = next(os.walk('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_testimg/'))[2]
print("No. of images = ", len(test_img))
      
# Iterate through testing masks
test_masks = next(os.walk('C:/Users/manos/Desktop/unetResearch/trough/dataset2/test_masks/png'))[2]
print("No. of masks = ", len(test_masks))




# Define parameters for image/mask resizing
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 4





# Create new arrays to store augmented training images/masks
X = np.zeros((len(train_img), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y = np.zeros((len(train_masks), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)

# Create new arrays to store validation images/masks
X_val = np.zeros((len(val_img), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_val = np.zeros((len(val_masks), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)

# Create new arrays to store testing images/masks
X_test = np.zeros((len(test_img), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_test = np.zeros((len(test_masks), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)




# Resize training images
for n, id_ in tqdm(enumerate(train_img), total=len(train_img)):
    # Load images
    img1 = cv2.imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_trainimg/aug/all/'+id_, 1)
    img1 = cv2.cvtColor(img1, 4)
    img2 = cv2.imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/blackhat_trainimg/aug/all/'+id_, 0)
    l1 = cv2.resize(img1, (IMG_HEIGHT, IMG_WIDTH))
    l2 = cv2.resize(img2, (IMG_HEIGHT, IMG_WIDTH))
    img = cv2.merge((l1, l2))
    # Save images
    X[n] = img/255
    
# Resize training masks
for n, id_ in tqdm(enumerate(train_masks), total=len(train_masks)):
    # Load masks
    mask = cv2.imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/train_masks/aug/all/'+id_, 0)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_NEAREST)
    Y[n] = mask
    
image_x = random.randint(0, len(train_img))
tiff.imshow(X[image_x])
plt.show()
tiff.imshow(Y[image_x])
plt.show()




# Resize validation images
for n, id_ in tqdm(enumerate(val_img), total=len(val_img)):
    # Load images
    img1 = cv2.imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_valimg/'+id_, 1)
    img1 = cv2.cvtColor(img1, 4)
    img2 = cv2.imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/blackhat_valimg/png/'+id_, 0)

    l1 = cv2.resize(img1, (IMG_HEIGHT, IMG_WIDTH))
    l2 = cv2.resize(img2, (IMG_HEIGHT, IMG_WIDTH))
    img = cv2.merge((l1, l2))
    # Save images
    X_val[n] = img/255
    
# Resize validation masks
for n, id_ in tqdm(enumerate(val_masks), total=len(val_masks)):
    # Load masks
    mask = cv2.imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/val_masks/png/'+id_, 0)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_NEAREST)
    Y_val[n] = mask

image_x = random.randint(0, len(val_img))
tiff.imshow(X_val[image_x])
plt.show()
tiff.imshow(Y_val[image_x])
plt.show()




# Resize test images
for n, id_ in tqdm(enumerate(test_img), total=len(test_img)):
    # Load images
    img1 = cv2.imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/falsecol_testimg/'+id_, 1)
    img1 = cv2.cvtColor(img1, 4)
    img2 = cv2.imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/blackhat_testimg/png/'+id_, 0)
    l1 = cv2.resize(img1, (IMG_HEIGHT, IMG_WIDTH))
    l2 = cv2.resize(img2, (IMG_HEIGHT, IMG_WIDTH))
    img = cv2.merge((l1, l2))
    # Save images
    X_test[n] = img/255
    
# Resize test masks
for n, id_ in tqdm(enumerate(test_masks), total=len(test_masks)):
    # Load masks
    mask = cv2.imread('C:/Users/manos/Desktop/unetResearch/trough/dataset2/test_masks/png/'+id_, 0)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_NEAREST)
    Y_test[n] = mask
    
image_x = random.randint(0, len(test_img))
tiff.imshow(X_test[image_x])
plt.show()
tiff.imshow(Y_test[image_x])
plt.show()


# BUILD AND TRAIN UNET MODEL



# Call UNet model function to build model
model = unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 32, 2)




# Define callbacks
callbacks = [ 
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, 
                      patience=5, 
                      min_lr=0.00001, 
                      verbose=1),
    tf.keras.callbacks.ModelCheckpoint('C:/Users/manos/Desktop/unetResearch/unetproto/code/checkpoints', 
                    verbose=1, 
                    save_best_only=True, 
                    save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss')
]




# Train model
batch_size = 8
train_steps = len(X)//batch_size
valid_steps = len(X_val)//batch_size


results = model.fit(X, Y,
                    batch_size = batch_size,
                    steps_per_epoch = train_steps,
                    validation_data = (X_val, Y_val),
                    validation_batch_size= 1,
                    validation_steps = valid_steps,
                    epochs = 200,
                    callbacks=callbacks)





# Save model
model.save("C:/Users/manos/Desktop/unetResearch/trough/code/trough_BestModel.h5")


# MODEL EVALUATION




# Visualize training and validation loss curves
plt.figure(figsize=(60, 30))
plt.plot(results.history['loss'], linewidth=8, color='r')                   
plt.plot(results.history['val_loss'], linewidth=8, color='b')
plt.title('Training and Validation Loss', fontsize=100, fontweight="bold")
plt.ylabel('Loss', fontsize=80)
plt.xlabel('Epoch', fontsize=80)
plt.legend(['Train', 'Validation'], loc='upper right', fontsize=50)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.show()





# Evaluate the performance of the model on test data
evaluate = model.evaluate(X_test, Y_test, batch_size = 1, verbose = 1)

print('Accuracy Test : {}'.format(evaluate[1]))





# Visualize model predictions on test images and load ground truth
# predicted masks into arrays

# Create empty arrays to store ground truth masks and predicted masks 
# for use in further evaluation
labels = np.empty([8, 256, 256])
preds = np.empty([8, 256, 256])

preds_test = model.predict(X_test, batch_size = 1, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.float32)

for n, id_ in tqdm(enumerate(X_test), total=len(X_test)):
    # Display ground truth mask
    imshow(np.squeeze(Y_test[n]))
    plt.show()
    # Store each mask into the array
    labels[n] = np.squeeze(Y_test[n])
    
    # Display predicted mask
    imshow(np.squeeze(preds_test_t[n]))
    plt.show()
    # Store each mask into the array
    preds[n] = np.squeeze(preds_test_t[n]).round()
    # Save each predicted mask
    #name = 'C:/Users/manos/Desktop/unetResearch/trough/model_eval/trough_aug/no_augs/masks/' + str(n) + '.png'
    #cv2.imwrite(name, np.squeeze(preds_test_t[n]))





# Prepare prediction and label arrays for model evaluation
# by flattening the arrays
preds_max_f = preds.flatten()

labels_max_f = labels.flatten()





# Construct confusion matrix with sklearn
cm = confusion_matrix(labels_max_f, preds_max_f)

# Retrieve precision, recall, F1 score and support for background and trough classes
report = classification_report(labels_max_f, preds_max_f)

print(report)





# Define function to plot confusion matrix 
classes = ['Background', 'Trough']

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')





# Plot confusion matrix
plt.figure(figsize=(5,5))
plot_confusion_matrix(cm, classes)

