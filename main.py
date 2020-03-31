'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is the main file for the project. From here you can train, test, predict, and manipulate the SegCaps of models.
Please see the README for detailed instructions for this project.
'''
from __future__ import print_function

from typing import Any
import os
import argparse
import csv
from time import gmtime, strftime

time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

from tqdm import tqdm
import tensorflow as tf

from utils.load_3D_data import load_data, convert_data_to_numpy
from utils.utils import safe_mkdir
from networks.model_helper import create_model


def main(args: Any) -> None:
    """
    Main function for running all training, testing, manipulation, or predictions. Steps are as follows:
    1. Create naming structure for all output.
    2. Create output directories
    3. Create the model and print a summary.
    4. Load the training/testing/validation/prediction data
    5. (optional) pre-process all data and save as numpy arrays.
    6. Run the training, testing, manipulation, and/or prediction functions.

    :param args: User-provided arguments

    :return: None
    """

    # Ensure training, testing, and manip are not all turned off
    assert (args.train or args.test or args.manip or args.pred), \
        'Cannot have train, test, pred, and manip all set to 0, Nothing to do.'

    # Setup the unique naming structure based on the user-defined arguments provided
    args.batch_size = args.batch_size_per_device * args.compute_resource_strategy.num_replicas_in_sync
    args.output_name = \
        'split-{}_batch-{}_shuff-{}_aug-{}_loss-{}_slic-{}_sub-{}_strid-{}_lr-{}_recon-{}_r1-{}_r2-{}'.format(
            args.split_num, args.batch_size, args.shuffle_data, args.aug_data, args.loss, args.slices, args.subsamp,
            args.stride, args.initial_lr, args.recon_wei, args.r1, args.r2)
    args.time = time
    out_net_name = args.net
    if args.net.find('caps') != -1 and args.net != 'segcapsbasic':
        out_net_name = out_net_name + 'r{}r{}'.format(args.r1, args.r2)

    # Create the directories for storing network, training, and logging outputs
    args.check_dir = os.path.join(args.data_root_dir, 'saved_models', out_net_name)
    args.log_dir = os.path.join(args.data_root_dir, 'logs', out_net_name)
    args.tf_log_dir = os.path.join(args.log_dir, 'tf_logs')
    args.output_dir = os.path.join(args.data_root_dir, 'plots', out_net_name, args.time)
    for dir_i in [args.check_dir, args.log_dir, args.tf_log_dir, args.output_dir]:
        safe_mkdir(dir_i)

    # Create the model for training/testing/manipulation
    net_input_shape = (None, None, args.slices)
    model_list = create_model(net=args.net, r1=args.r1, r2=-args.r2, input_shape=net_input_shape,
                              compute_resource_strategy=args.compute_resource_strategy)
    model_list[0].summary(positions=[.38, .65, .75, 1.])

    # Load the training, validation, and testing data if running train, test, or manip
    all_imgs_list = list()
    if args.train or args.test or args.manip:
        train_list, val_list, test_list = load_data(root=args.data_root_dir, split=args.split_num,
                                                    k_folds=args.k_folds, val_split=args.val_split,
                                                    rand_seed=args.rand_seed)
        all_imgs_list = all_imgs_list + list(train_list) + list(val_list) + list(test_list)
        print('\nFound a total of {} images with lables.'.format(len(all_imgs_list)))
        print('\t{} images for training.'.format(len(train_list)))
        print('\t{} images for validation.'.format(len(val_list)))
        print('\t{} images for testing.'.format(len(test_list)))
        # Print the list of images selected for validation
        print('\nRandomly selected validation images:')
        print(val_list)
        print('\n')

    # Load the images for prediction. There are no ground truth annotations for these.
    if args.pred:
        with open(os.path.join(args.data_root_dir, 'split_lists', 'pred_split_{}.csv'.format(args.split_num)),
                  'r') as f:
            reader = csv.reader(f)
            pred_list = list(reader)

        all_imgs_list = all_imgs_list + list(pred_list)
        print('\nFound a total of {} images for prediction.'.format(len(list(pred_list))))

    # Optional pre-processing stage to pre-compute and save numpy arrays for all data.
    if args.create_all_imgs:
        print('-' * 98, '\nCreating all images... This will take some time.\n', '-' * 98)
        for img_name in tqdm(all_imgs_list):
            _, _ = convert_data_to_numpy(root_path=args.data_root_dir, img_name=img_name[0],
                                         no_masks=args.pred, overwrite=True)

    # Run the training, testing, manipulation and/or prediction functions
    if args.train:
        from train import train
        print('-' * 98, '\nRunning Training\n', '-' * 98)
        train(args, train_list, val_list, model_list[0], net_input_shape)

    if args.test:
        from test import test
        print('-' * 98, '\nRunning Testing\n', '-' * 98)
        test(args, test_list, model_list, net_input_shape)

    if args.manip and args.net.find('caps') != -1:
        from manip import manip
        print('-' * 98, '\nRunning Manipulate\n', '-' * 98)
        assert (len(model_list) == 3), "Must be using segcaps with the three models."
        manip(args, test_list, model_list[2], net_input_shape)

    if args.pred:
        from predict import predict
        print('-' * 98, '\nRunning Prediction\n', '-' * 98)
        predict(args, pred_list, model_list, net_input_shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation of 3D CT Images.')

    # REQUIRED ARGUMENT:
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory for your data and experiments.')

    # OPTIONAL ARGUMENTS:

    # Path related arguments
    parser.add_argument('--img_dir', type=str, default='imgs',
                        help='Path to the directory containing your 3D CT scans (relative to root_path).')
    parser.add_argument('--mask_dir', type=str, default='masks',
                        help='Path to the directory containing your 3D CT masks (relative to root_path).')

    # What to run (Training/Testing/Manipulation/Prediction) arguments
    parser.add_argument('--train', action='store_true',
                        help='Set this flag to enable training.')
    parser.add_argument('--test', action='store_true',
                        help='Set this flag to enable testing.')
    parser.add_argument('--manip', action='store_true',
                        help='Set this flag to enable manipulation of capsule vectors.')
    parser.add_argument('--pred', action='store_true',
                        help='Set this flag to enable prediction.')

    # Data related arguments
    parser.add_argument('--k_folds', type=int, default=10,
                        help='Number of training splits to create for k-fold cross-validation.')
    parser.add_argument('--split_num', type=str, default='0',
                        help='Which training split to train/test on.')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Percentage between 0 and 1 of training split to use as validation.')

    parser.add_argument('--shuffle_data', type=int, default=1, choices=[0, 1],
                        help='Whether or not to shuffle the training data (both per epoch and in slice order.')
    parser.add_argument('--aug_data', type=int, default=1, choices=[0, 1],
                        help='Whether or not to use data augmentation during training.')

    parser.add_argument('--slices', type=int, default=1,
                        help='Number of slices to include for training/testing.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Number of slices to move when generating the next sample.')
    parser.add_argument('--subsamp', type=int, default=-1,
                        help='For 3D samples only: Number of slices to skip when forming 3D samples for training. '
                             'Enter -1 for random subsampling up to 5 percent of total slices.')

    # Network related arguments
    parser.add_argument('--net', type=str.lower, default='segcaps',
                        choices=['segcaps', 'segcapsbasic', 'unet', 'd_unet', 'tiramisu56',
                                 'tiramisu67', 'tiramisu103', 'phnn', 'd_phnn'],
                        help='Choose your network.')
    parser.add_argument('--weights_path', type=str, default='',
                        help='path/to/trained_model.hdf5 (relative to root_path). Set to "" for none.')

    parser.add_argument('--loss', type=str.lower, default='w_bce', choices=['bce', 'w_bce', 'dice', 'mar', 'w_mar'],
                        help='Which loss to use. "bce" and "w_bce": unweighted and weighted binary cross entropy'
                             '"dice": soft dice coefficient, "mar" and "w_mar": unweighted and weighted margin loss.')
    parser.add_argument('--batch_size_per_device', type=int, default=1,
                        help='Batch size for training/testing. For multi-GPU training this should be the batch size '
                             'per GPU. The total batch size will be the batch size per GPU times the number of GPUs. '
                             'The saved value in naming is the total batch size since this is the true batch size.')
    parser.add_argument('--initial_lr', type=float, default=0.0001,
                        help='Initial learning rate for Adam.')

    parser.add_argument('--thresh_level', type=float, default=0.,
                        help='Enter 0.0 for otsu thresholding, else set value')
    parser.add_argument('--r1', type=int, default=3,
                        help='Number of capsule routing iterations to perform when the spatial resolution stays the '
                             'same.')
    parser.add_argument('--r2', type=int, default=3,
                        help='Number of capsule routing iterations to perform when the spatial resolution changes.')
    parser.add_argument('--recon_wei', type=float, default=131.072,
                        help="If using SegCaps: The coefficient (weighting) for the loss of decoder")

    # Output/saving related arguments
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Prefix to append to saved CSV.')
    parser.add_argument('--save_raw', type=int, default=1, choices=[0, 1],
                        help='Enter 0 to not save, 1 to save.')
    parser.add_argument('--save_seg', type=int, default=1, choices=[0, 1],
                        help='Enter 0 to not save, 1 to save.')
    parser.add_argument('--compute_dice', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_jaccard', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_assd', type=int, default=0,
                        help='0 or 1')
    parser.add_argument('--compute_hd', type=int, default=1,
                        help='0 or 1')

    # Miscellaneous arguments
    parser.add_argument('--create_all_imgs', type=int, default=0, choices=[0, 1],
                        help='Whether to create all images up front or dynamically as they are needed.')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Set the verbose value for training. 0: Silent, 1: per iteration, 2: per epoch.')
    parser.add_argument('--which_gpus', type=str, default="-1",
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--rand_seed', type=int, default=5,
                        help='Random seed for reproducibility.')

    arguments = parser.parse_args()


    # Mask out the GPUs based on the user selected choices.
    if arguments.which_gpus == '-1':
        pass
    elif arguments.which_gpus == '-2':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        gpu_str = arguments.which_gpus.replace(' ', '')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    arguments.compute_resource_strategy = tf.distribute.MirroredStrategy()

    main(arguments)
