'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for training models. Please see the README for details about training.
'''

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from os.path import join
import numpy as np

from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf

from custom_losses import dice_hard, weighted_binary_crossentropy_loss, dice_loss, margin_loss
from load_3D_data import load_class_weights, generate_train_batches, generate_val_batches


def get_loss(root, split, net, recon_wei, choice):
    if choice == 'w_bce':
        pos_class_weight = load_class_weights(root=root, split=split)
        loss = weighted_binary_crossentropy_loss(pos_class_weight)
    elif choice == 'bce':
        loss = 'binary_crossentropy'
    elif choice == 'dice':
        loss = dice_loss
    elif choice == 'w_mar':
        pos_class_weight = load_class_weights(root=root, split=split)
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=pos_class_weight)
    elif choice == 'mar':
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    else:
        raise Exception("Unknow loss_type")

    if net.find('caps') != -1:
        return {'out_seg': loss, 'out_recon': 'mse'}, {'out_seg': 1., 'out_recon': recon_wei}
    else:
        return loss, None

def get_callbacks(arguments):
    if arguments.net.find('caps') != -1:
        monitor_name = 'val_out_seg_dice_hard'
    else:
        monitor_name = 'val_dice_hard'

    csv_logger = CSVLogger(join(arguments.log_dir, arguments.output_name + '_log_' + arguments.time + '.csv'), separator=',')
    tb = TensorBoard(arguments.tf_log_dir, batch_size=arguments.batch_size, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(join(arguments.check_dir, arguments.output_name + '_model_' + arguments.time + '.hdf5'),
                                       monitor=monitor_name, save_best_only=True, save_weights_only=True,
                                       verbose=1, mode='max')
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=5,verbose=1, mode='max')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=25, verbose=0, mode='max')

    return [model_checkpoint, csv_logger, lr_reducer, early_stopper, tb]

def compile_model(args, net_input_shape, uncomp_model):
    # Set optimizer loss and metrics
    opt = Adam(lr=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6)
    if args.net.find('caps') != -1:
        metrics = {'out_seg': dice_hard}
    else:
        metrics = [dice_hard]

    loss, loss_weighting = get_loss(root=args.data_root_dir, split=args.split_num, net=args.net,
                                    recon_wei=args.recon_wei, choice=args.loss)

    # If using CPU or single GPU
    if args.gpus <= 1:
        uncomp_model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return uncomp_model
    # If using multiple GPUs
    else:
        with tf.device("/cpu:0"):
            uncomp_model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
            model = multi_gpu_model(uncomp_model, gpus=args.gpus)
            model.__setattr__('callback_model', uncomp_model)
        model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
        return model


def plot_training(training_history, arguments):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(arguments.net, fontsize=18)

    if arguments.net.find('caps') != -1:
        ax1.plot(training_history.history['out_seg_dice_hard'])
        ax1.plot(training_history.history['val_out_seg_dice_hard'])
    else:
        ax1.plot(training_history.history['dice_hard'])
        ax1.plot(training_history.history['val_dice_hard'])
    ax1.set_title('Dice Coefficient')
    ax1.set_ylabel('Dice', fontsize=12)
    ax1.legend(['Train', 'Val'], loc='upper left')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    if arguments.net.find('caps') != -1:
        ax1.set_xticks(np.arange(0, len(training_history.history['out_seg_dice_hard'])))
    else:
        ax1.set_xticks(np.arange(0, len(training_history.history['dice_hard'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history['loss'])
    ax2.plot(training_history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Val'], loc='upper right')
    ax1.set_xticks(np.arange(0, len(training_history.history['loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(join(arguments.output_dir, arguments.output_name + '_plots_' + arguments.time + '.png'))
    plt.close()

def train(args, train_list, val_list, u_model, net_input_shape):
    # Compile the loaded model
    model = compile_model(args=args, net_input_shape=net_input_shape, uncomp_model=u_model)
    # Set the callbacks
    callbacks = get_callbacks(args)

    # Training the network
    history = model.fit_generator(
        generate_train_batches(args.data_root_dir, train_list, net_input_shape, net=args.net,
                               batchSize=args.batch_size, numSlices=args.slices, subSampAmt=args.subsamp,
                               stride=args.stride, shuff=args.shuffle_data, aug_data=args.aug_data),
        max_queue_size=40, workers=4, use_multiprocessing=False,
        steps_per_epoch=10000,
        validation_data=generate_val_batches(args.data_root_dir, val_list, net_input_shape, net=args.net,
                                             batchSize=args.batch_size,  numSlices=args.slices, subSampAmt=0,
                                             stride=20, shuff=args.shuffle_data),
        validation_steps=500, # Set validation stride larger to see more of the data.
        epochs=200,
        callbacks=callbacks,
        verbose=1)

    # Plot the training data collected
    plot_training(history, args)
