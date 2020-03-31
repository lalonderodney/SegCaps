"""
This file contains an implementation of P-HNN based on the paper
"Progressive and Multi-Path Holistically Nested Neural networks for Pathological Lung Segmentation from CT Images"
(https://arxiv.org/abs/1706.03702).
Note we attempted to train exactly as the original implementation but the models performed terribly.
Therefore we removed the individual layer learning rates and changed from zero kernel initialization to Xavier normal.
"""
from keras import initializers
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation, Add

def PHNN(input_shape=(512,512,1), downscale=1.):
    inputs = Input(input_shape)
    conv1 = Conv2D(int(64//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv1_1')(inputs)
    conv1 = Conv2D(int(64//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv1_2')(conv1)
    bn1 = BatchNormalization(name='bn1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(bn1)

    conv2 = Conv2D(int(128//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv2_1')(pool1)
    conv2 = Conv2D(int(128//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv2_2')(conv2)
    bn2 = BatchNormalization(name='bn2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(bn2)

    conv3 = Conv2D(int(256//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv3_1')(pool2)
    conv3 = Conv2D(int(256//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv3_2')(conv3)
    conv3 = Conv2D(int(256//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv3_3')(conv3)
    bn3 = BatchNormalization(name='bn3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(bn3)

    conv4 = Conv2D(int(512//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv4_1')(pool3)
    conv4 = Conv2D(int(512//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv4_2')(conv4)
    conv4 = Conv2D(int(512//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv4_3')(conv4)
    bn4 = BatchNormalization(name='bn4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(bn4)

    conv5 = Conv2D(int(512//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv5_1')(pool4)
    conv5 = Conv2D(int(512//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv5_2')(conv5)
    conv5 = Conv2D(int(512//downscale), (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.glorot_normal(), name='conv5_3')(conv5)
    bn5 = BatchNormalization(name='bn5')(conv5)

    score1 = Conv2D(1, (1, 1), padding='same', kernel_initializer=initializers.glorot_normal(), name='score1')(bn1)
    out1 = Activation('sigmoid', name='out1')(score1)

    score2 = Conv2D(1, (1, 1), padding='same', kernel_initializer=initializers.glorot_normal(), name='score2')(bn2)
    up2 = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same',
                          kernel_initializer=initializers.glorot_normal(), name='up2')(score2)
    sum2 = Add(name='sum2')([score1, up2])
    out2 = Activation('sigmoid', name='out2')(sum2)

    score3 = Conv2D(1, (1, 1), padding='same', kernel_initializer=initializers.glorot_normal(), name='score3')(bn3)
    up3 = Conv2DTranspose(1, (8, 8), strides=(4, 4), padding='same',
                          kernel_initializer=initializers.glorot_normal(), name='up3')(score3)
    sum3 = Add(name='sum3')([sum2, up3])
    out3 = Activation('sigmoid', name='out3')(sum3)

    score4 = Conv2D(1, (1, 1), padding='same', kernel_initializer=initializers.glorot_normal(), name='score4')(bn4)
    up4 = Conv2DTranspose(1, (16, 16), strides=(8, 8), padding='same',
                          kernel_initializer=initializers.glorot_normal(), name='up4')(score4)
    sum4 = Add(name='sum4')([sum3, up4])
    out4 = Activation('sigmoid', name='out4')(sum4)

    score5 = Conv2D(1, (1, 1), padding='same', kernel_initializer=initializers.glorot_normal(), name='score5')(bn5)
    up5 = Conv2DTranspose(1, (32, 32), strides=(16, 16), padding='same',
                          kernel_initializer=initializers.glorot_normal(), name='up5')(score5)
    sum5 = Add(name='sum5')([sum4, up5])
    out5 = Activation('sigmoid', name='out5')(sum5)

    model = Model(inputs=[inputs], outputs=[out1, out2, out3, out4, out5])

    return model
