'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is the main file for the project. From here you can train, test, and manipulate the SegCaps of models.
Please see the README for detailed instructions for this project.
'''

from __future__ import print_function, division

import os
import argparse
from glob import glob
from time import gmtime, strftime
time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

from PIL import Image
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from keras.utils import print_summary
from keras.callbacks import ModelCheckpoint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from networks.model_helper import create_model
from train import compile_model
from utils.metrics import dc, jc


def main(args):
    out_net_name = args.net
    if args.net.find('caps') != -1 and args.net != 'segcapsbasic':
        out_net_name = out_net_name + 'r{}r{}'.format(args.r1, args.r2)
    args.output_name = out_net_name
    args.time = ''

    args.check_dir = os.path.join(args.data_root_dir, 'saved_models', '{}_iters'.format(args.epochs))
    try:
        os.makedirs(args.check_dir)
    except:
        pass

    args.results_dir = os.path.join(args.data_root_dir, 'results', '{}_iters_nprot'.format(args.epochs))
    try:
        os.makedirs(args.results_dir)
    except:
        pass

    net_input_shape = (None, None, 3)
    # Create the model for training/testing/manipulation
    model_list = create_model(args=args, input_shape=net_input_shape)
    print_summary(model=model_list[0], positions=[.38, .65, .75, 1.])

    # Load the images as a list
    img_list = glob(os.path.os.path.join(args.data_root_dir, args.image_path, '*.jpg'))

    # Loop over the images
    for img_path in img_list:
        # Load the image
        img_name = os.path.basename(img_path)[:-4]
        im = Image.open(img_path)
        im = im.resize((im.size[0] // (2 ** 6) * (2 ** 6), im.size[1] // (2 ** 6) * (2 ** 6)), resample=1)  # Assume 6 downsamples
        mask = Image.open(os.path.os.path.join(args.data_root_dir, args.mask_path, '{}.png'.format(img_name)))
        mask = mask.resize((mask.size[0] // (2 ** 6) * (2 ** 6), mask.size[1] // (2 ** 6) * (2 ** 6)), resample=1)  # Assume 6 downsamples

        # Create numpy versions of the image and mask
        np_img = np.expand_dims(np.array(im), axis=0)
        np_img = np_img / 255.
        np_mask = np.expand_dims(np.expand_dims(np.array(mask), axis=0), axis=-1)
        np_mask[np_mask != 0] = 1

        # Compile the loaded model
        train_model = compile_model(args=args, net_input_shape=net_input_shape, uncomp_model=model_list[0])

        # Set the callbacks
        if arguments.net.find('caps') != -1:
            monitor_name = 'val_out_seg_dice_hard'
        elif arguments.net.find('phnn') != -1:
            monitor_name = 'val_out5_dice_hard'
        else:
            monitor_name = 'val_dice_hard'
        model_checkpoint = ModelCheckpoint(
            os.path.join(args.check_dir, '{}_{}_model.hdf5'.format(args.output_name, img_name)),
            monitor=monitor_name, save_best_only=False, save_weights_only=True, verbose=1, period=args.epochs)

        # Training the network
        # if args.net.find('caps') != -1:
        #     train_model.fit(x=[np_img, np_mask], y=[np_mask, np_mask * np_img], batch_size=1, epochs=args.epochs,
        #                     validation_freq=args.epochs, validation_data=([np_img, np_mask], [np_mask, np_mask * np_img]),
        #                     verbose=args.verbose, callbacks=[model_checkpoint], shuffle=False)
        # else:
        #     train_model.fit(x=np_img, y=np_mask, batch_size=1, epochs=args.epochs, verbose=args.verbose,
        #                     validation_freq=args.epochs, validation_data=(np_img, np_mask),
        #                     callbacks=[model_checkpoint], shuffle=False)

        # Load the trained model for predictions
        if len(model_list) > 1:
            eval_model = model_list[1]
        else:
            eval_model = model_list[0]

        if args.weights_path != '':
            try:
                eval_model.load_weights(args.weights_path)
            except Exception as e:
                print(e)
                raise Exception('Failed to load weights from specified file_path.')
        else:
            try:
                eval_model.load_weights(os.path.join(args.check_dir, '{}_{}_model.hdf5'.format(args.output_name,
                                                                                               img_name)))
            except Exception as e:
                print(e)
                raise Exception('Failed to load weights from training.')


        for rot in range(4): #range(0, 360, 45): # Commented out parts here are using imrotate which gives black borders and messes with results a lot.
            #x = np.expand_dims(np.array(im.rotate(rot)), axis=0)
            #x = x / 255.
            x = np.expand_dims(np.rot90(np_img[0,...], k=rot), axis=0)
            #y = np.array(mask.rotate(rot))
            #y[y != 0] = 1
            y = np.rot90(np_mask[0,...,0], k=rot)
            plt.imsave(os.path.join(args.results_dir, '{}_{}_img.png'.format(img_name, rot)), x[0, ...])
            plt.imsave(os.path.join(args.results_dir, '{}_{}_gt.png'.format(img_name, rot)), y)

            output_array = eval_model.predict(x, batch_size=None, verbose=1)

            if args.net.find('caps') != -1:
                output = output_array[0][0,:,:,0]
                #recon = output_array[1][0,:,:,0]
            else:
                output = output_array[0,:,:,0]

            # Threshold the output
            output[output < 0.5] = 0
            output[output >= 0.5] = 1
            output = binary_fill_holes(output).astype(np.uint8)

            # Compute dice and Jaccard (IoU) scores
            dsc = dc(output, y)
            jacc = jc(output, y)

            plt.imsave(os.path.join(args.results_dir, '{}_{}_{}_dsc-{:.4f}_jacc-{:.4f}.png'.format(
                img_name, rot, out_net_name, dsc, jacc)), output)

        x = np.expand_dims(np.array(im.transpose(Image.FLIP_LEFT_RIGHT)), axis=0)
        x = x / 255.
        y = np.array(mask.transpose(Image.FLIP_LEFT_RIGHT))
        y[y != 0] = 1
        plt.imsave(os.path.join(args.results_dir, '{}_mirror_img.png'.format(img_name)), x[0, ...])
        plt.imsave(os.path.join(args.results_dir, '{}_mirror_gt.png'.format(img_name)), y)

        output_array = eval_model.predict(x, batch_size=None, verbose=1)

        if args.net.find('caps') != -1:
            output = output_array[0][0, :, :, 0]
            # recon = output_array[1][0,:,:,0]
        else:
            output = output_array[0, :, :, 0]

        # Threshold the output
        output[output < 0.5] = 0
        output[output >= 0.5] = 1
        output = binary_fill_holes(output).astype(np.uint8)

        # Compute dice and Jaccard (IoU) scores
        dsc = dc(output, y)
        jacc = jc(output, y)

        plt.imsave(os.path.join(args.results_dir, '{}_mirror_{}_dsc-{:.4f}_jacc-{:.4f}.png'.format(
            img_name, out_net_name, dsc, jacc)), output)

        print('Done with {}.'.format(img_name))
    print('Done with all images.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on Medical Data')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory for your data.')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to image relative to data_root_dir.')
    parser.add_argument('--mask_path', type=str, required=True,
                        help='Path to mask relative to data_root_dir.')


    parser.add_argument('--weights_path', type=str, default='',
                        help='full/path/to/trained_model.hdf5. Set to "" for none.')
    parser.add_argument('--rand_seed', type=int, default=5,
                        help='Random seed for training splits.')
    parser.add_argument('--net', type=str.lower, default='segcaps',
                        choices=['segcaps', 'segcapsbasic', 'unet', 'd_unet', 'tiramisu56',
                                 'tiramisu67', 'tiramisu103', 'phnn', 'd_phnn'],
                        help='Choose your network.')
    parser.add_argument('--r1', type=int, default=3,
                        help='Number of routing iterations to perform when the spatial resolution stays the same.')
    parser.add_argument('--r2', type=int, default=3,
                        help='Number of routing iterations to perform when the spatial resolution changes.')
    parser.add_argument('--loss', type=str.lower, default='bce', choices=['bce', 'dice'],
                        help='Which loss to use. "bce": binary cross entropy'
                             '"dice": soft dice coefficient')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training/testing.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train for.')
    parser.add_argument('--initial_lr', type=float, default=0.001,
                        help='Initial learning rate for Adam.')
    parser.add_argument('--recon_wei', type=float, default=20.0,
                        help="If using capsnet: The coefficient (weighting) for the loss of decoder")
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Set the verbose value for training. 0: Silent, 1: per iteration, 2: per epoch.')
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Prefix to append to saved CSV.')
    parser.add_argument('--compute_dice', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_jaccard', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_assd', type=int, default=0,
                        help='0 or 1')
    parser.add_argument('--compute_hd', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--which_gpus', type=str, default="0",
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the --which_gpus arg or if using CPU, '
                             'then this number will be inferred, else this argument must be included.')

    arguments = parser.parse_args()

    # Mask out the GPUs based on the user arguments.
    if arguments.which_gpus == -2:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif arguments.which_gpus == '-1':
        assert (arguments.gpus != -1), 'Use all GPUs option selected under --which_gpus, with this option the user MUST ' \
                                  'specify the number of GPUs available with the --gpus option.'
    else:
        arguments.gpus = len(arguments.which_gpus.split(','))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.which_gpus)

    if arguments.gpus > 1:
        assert arguments.batch_size >= arguments.gpus, 'Error: Must have at least as many items per batch as GPUs ' \
                                                       'for multi-GPU training. For model parallelism instead of ' \
                                                       'data parallelism, modifications must be made to the code.'

    main(arguments)
