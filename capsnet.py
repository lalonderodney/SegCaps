'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the network definitions for the various capsule network architectures.
'''

from keras import layers, models
from keras import backend as K
K.set_image_data_format('channels_last')

from capsule_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Mask, Length

def CapsNetR3(input_shape, n_class=2):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='conv_cap_2_1')(primary_caps)

    # Layer 2: Convolutional Capsule
    conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=3, name='conv_cap_2_2')(conv_cap_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap_3_1')(conv_cap_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=3, name='conv_cap_3_2')(conv_cap_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap_4_1')(conv_cap_3_2)

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_1_1')(conv_cap_4_1)

    # Skip connection
    up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=3, name='deconv_cap_1_2')(up_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_2_1')(deconv_cap_1_2)

    # Skip connection
    up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=3, name='deconv_cap_2_2')(up_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_3_1')(deconv_cap_2_2)

    # Skip connection
    up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=3, name='seg_caps')(up_3)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()
    y = layers.Input(shape=input_shape[:-1]+(1,))
    masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_2)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_seg, shared_decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=((H.value, W.value, C.value, A.value)))
    noised_seg_caps = layers.Add()([seg_caps, noise])
    masked_noised_y = Mask()([noised_seg_caps, y])
    manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model


def CapsNetR1(input_shape, n_class=2):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=1, name='conv_cap_2_1')(primary_caps)

    # Layer 2: Convolutional Capsule
    conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=3, name='conv_cap_2_2')(conv_cap_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='conv_cap_3_1')(conv_cap_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=3, name='conv_cap_3_2')(conv_cap_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='conv_cap_4_1')(conv_cap_3_2)

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_1_1')(conv_cap_4_1)

    # Skip connection
    up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=1, name='deconv_cap_1_2')(up_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_2_1')(deconv_cap_1_2)

    # Skip connection
    up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=1, name='deconv_cap_2_2')(up_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_3_1')(deconv_cap_2_2)

    # Skip connection
    up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=1, name='seg_caps')(up_3)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()
    y = layers.Input(shape=input_shape[:-1]+(1,))
    masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_2)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_seg, shared_decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=((H.value, W.value, C.value, A.value)))
    noised_seg_caps = layers.Add()([seg_caps, noise])
    masked_noised_y = Mask()([noised_seg_caps, y])
    manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model


def CapsNetBasic(input_shape, n_class=2):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=3, name='seg_caps')(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()
    y = layers.Input(shape=input_shape[:-1]+(1,))
    masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_2)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_seg, shared_decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=((H.value, W.value, C.value, A.value)))
    noised_seg_caps = layers.Add()([seg_caps, noise])
    masked_noised_y = Mask()([noised_seg_caps, y])
    manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model
