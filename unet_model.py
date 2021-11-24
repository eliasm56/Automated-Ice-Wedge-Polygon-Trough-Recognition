# Import libraries
import tensorflow as tf, keras
from keras import backend as K

# Define evaluation metric

def f1(y_true, y_pred):
    '''
    Calculates the F1 by using keras.backend
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def unet_model(img_height, img_width, img_channels, n_filters_start, growth_factor):
    #Encoder
    n_filters = n_filters_start
    inputs = tf.keras.layers.Input((img_height, img_width, img_channels))
    c1 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    n1 = tf.keras.layers.BatchNormalization()(c1)
    a1 = tf.keras.activations.relu(n1, alpha=0.0, max_value=None, threshold=0)
    c1 = tf.keras.layers.Dropout(0.1)(a1)
    c1 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c1)
    n1 = tf.keras.layers.BatchNormalization()(c1)
    a1 = tf.keras.activations.relu(n1, alpha=0.0, max_value=None, threshold=0)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(a1)
    
    n_filters *= growth_factor
    c2 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    n2 = tf.keras.layers.BatchNormalization()(c2)
    a2 = tf.keras.activations.relu(n2, alpha=0.0, max_value=None, threshold=0)
    c2 = tf.keras.layers.Dropout(0.1)(a2)
    c2 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
    n2 = tf.keras.layers.BatchNormalization()(c2)
    a2 = tf.keras.activations.relu(n2, alpha=0.0, max_value=None, threshold=0)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(a2)
 
    n_filters *= growth_factor
    c3 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    n3 = tf.keras.layers.BatchNormalization()(c3)
    a3 = tf.keras.activations.relu(n3, alpha=0.0, max_value=None, threshold=0)
    c3 = tf.keras.layers.Dropout(0.2)(a3)
    c3 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
    n3 = tf.keras.layers.BatchNormalization()(c3)
    a3 = tf.keras.activations.relu(n3, alpha=0.0, max_value=None, threshold=0)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(a3)
 
    n_filters *= growth_factor
    c4 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
    n4 = tf.keras.layers.BatchNormalization()(c4)
    a4 = tf.keras.activations.relu(n4, alpha=0.0, max_value=None, threshold=0)
    c4 = tf.keras.layers.Dropout(0.2)(a4)
    c4 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c4)
    n4 = tf.keras.layers.BatchNormalization()(c4)
    a4 = tf.keras.activations.relu(n4, alpha=0.0, max_value=None, threshold=0)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(a4)
 
    # Bridge
    n_filters *= growth_factor
    c5 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
    n5 = tf.keras.layers.BatchNormalization()(c5)
    a5 = tf.keras.activations.relu(n5, alpha=0.0, max_value=None, threshold=0)
    c5 = tf.keras.layers.Dropout(0.3)(a5)
    c5 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
    n5 = tf.keras.layers.BatchNormalization()(c5)
    a5 = tf.keras.activations.relu(n5, alpha=0.0, max_value=None, threshold=0)

    # Decoder
    n_filters //= growth_factor
    u6 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(a5)
    u6 = tf.keras.layers.concatenate([u6, a4])
    c6 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
    n6 = tf.keras.layers.BatchNormalization()(c6)
    a6 = tf.keras.activations.relu(n6, alpha=0.0, max_value=None, threshold=0)
    c6 = tf.keras.layers.Dropout(0.2)(a6)
    c6 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c6)
    n6 = tf.keras.layers.BatchNormalization()(c6)
    a6 = tf.keras.activations.relu(n6, alpha=0.0, max_value=None, threshold=0)
    
    n_filters //= growth_factor
    u7 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(a6)
    u7 = tf.keras.layers.concatenate([u7, a3])
    c7 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
    n7 = tf.keras.layers.BatchNormalization()(c7)
    a7 = tf.keras.activations.relu(n7, alpha=0.0, max_value=None, threshold=0)
    c7 = tf.keras.layers.Dropout(0.2)(a7)
    c7 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
    n7 = tf.keras.layers.BatchNormalization()(c7)
    a7 = tf.keras.activations.relu(n7, alpha=0.0, max_value=None, threshold=0)
 
    n_filters //= growth_factor
    u8 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(a7)
    u8 = tf.keras.layers.concatenate([u8, a2])
    c8 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
    n8 = tf.keras.layers.BatchNormalization()(c8)
    a8 = tf.keras.activations.relu(n8, alpha=0.0, max_value=None, threshold=0)
    c8 = tf.keras.layers.Dropout(0.1)(a8)
    c8 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
    n8 = tf.keras.layers.BatchNormalization()(c8)
    a8 = tf.keras.activations.relu(n8, alpha=0.0, max_value=None, threshold=0)
 
    n_filters //= growth_factor
    u9 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(a8)
    u9 = tf.keras.layers.concatenate([u9, a1], axis=3)
    c9 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
    n9 = tf.keras.layers.BatchNormalization()(c9)
    a9 = tf.keras.activations.relu(n9, alpha=0.0, max_value=None, threshold=0)
    c9 = tf.keras.layers.Dropout(0.1)(a9)
    c9 = tf.keras.layers.Conv2D(n_filters, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
    n9 = tf.keras.layers.BatchNormalization()(c9)
    a9 = tf.keras.activations.relu(n9, alpha=0.0, max_value=None, threshold=0)
 
 
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(a9)

    # Define model and compile
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics = [dice_coef])
    model.summary()
    return model

