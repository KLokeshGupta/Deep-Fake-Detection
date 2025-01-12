import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    ReLU,
    MaxPooling2D,
    Dense,
    Dropout,
    Softmax,
    Flatten,
)

def get_vgg16_conv5(input_tensor, params):
    x = Conv2D(64, (3, 3), padding='same', name='conv1_1', **params)(input_tensor)
    x = ReLU(name='conv1_1_relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv1_2', **params)(x)
    x = ReLU(name='conv1_2_relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    x = Conv2D(128, (3, 3), padding='same', name='conv2_1', **params)(x)
    x = ReLU(name='conv2_1_relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='conv2_2', **params)(x)
    x = ReLU(name='conv2_2_relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    x = Conv2D(256, (3, 3), padding='same', name='conv3_1', **params)(x)
    x = ReLU(name='conv3_1_relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_2', **params)(x)
    x = ReLU(name='conv3_2_relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_3', **params)(x)
    x = ReLU(name='conv3_3_relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv4_1', **params)(x)
    x = ReLU(name='conv4_1_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv4_2', **params)(x)
    x = ReLU(name='conv4_2_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv4_3', **params)(x)
    x = ReLU(name='conv4_3_relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv5_1', **params)(x)
    x = ReLU(name='conv5_1_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv5_2', **params)(x)
    x = ReLU(name='conv5_2_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv5_3', **params)(x)
    x = ReLU(name='conv5_3_relu')(x)
    return x

def get_vgg16_pool5(input_tensor, params):
    x = get_vgg16_conv5(input_tensor, params)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool5')(x)
    return x

def get_prob(input_tensor, params, num_classes=1000, is_train=True):
    x = get_vgg16_pool5(input_tensor, params)
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(4096, activation='relu', name='fc6', **params)(x)
    if is_train:
        x = Dropout(0.5)(x)  # Dropout works differently in train vs inference automatically
    
    x = Dense(4096, activation='relu', name='fc7', **params)(x)
    if is_train:
        x = Dropout(0.5)(x)
    
    x = Dense(num_classes, activation='softmax', name='fc8', **params)(x)
    return x
