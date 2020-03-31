'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for manipulating the vectors of the final layer of capsules (the SegCaps or segmentation capsules).
This manipulation attempts to show what each dimension of these final vectors are storing (paying attention to),
in terms of information about the positive input class.
Please see the README for further details about how to use this file.
'''

from __future__ import print_function

import os
import SimpleITK as sitk
from tqdm import tqdm, trange
from PIL import Image
import numpy as np
import math

from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import print_summary


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def manip(args, test_list, manip_model, net_input_shape):
    if args.net != 'segcapsbasic':
        out_net_name = args.net + 'r{}r{}'.format(args.r1, args.r2)
    else:
        out_net_name = args.net

    if args.weights_path != '':
        manip_out_dir = os.path.join(args.data_root_dir, 'results', out_net_name, 'split_{}'.format(args.split_num),
                          os.path.basename(args.weights_path)[-24:-5], 'manip_output')
        try:
            manip_model.load_weights(args.weights_path)
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')
    else:
        manip_out_dir = os.path.join(args.data_root_dir, 'results', out_net_name, 'split_{}'.format(args.split_num),
                                     args.time, 'manip_output')
        try:
            manip_model.load_weights(os.path.join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5'))
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')

    try:
        os.makedirs(manip_out_dir)
    except:
        pass

    # Manipulating capsule vectors
    print('Running Manipulaiton of Capsule Vectors... This will take some time...')
    for i, img in enumerate(tqdm(test_list)):
        sitk_img = sitk.ReadImage(os.path.join(args.data_root_dir, 'imgs', img[0]))
        img_data = sitk.GetArrayFromImage(sitk_img)
        num_slices = img_data.shape[0]
        sitk_mask = sitk.ReadImage(os.path.join(args.data_root_dir, 'masks', img[0]))
        gt_data = sitk.GetArrayFromImage(sitk_mask)

        x, y = img_data[num_slices//2, :, :], gt_data[num_slices//2, :, :]
        x, y = np.expand_dims(np.expand_dims(x, -1), 0), np.expand_dims(np.expand_dims(y, -1), 0)

        noise = np.zeros([1, img_data.shape[1], img_data.shape[2], 1, 16])
        x_recons = []
        for dim in trange(16):
            for r in [-0.25, -0.125, 0, 0.125, 0.25]:
                tmp = np.copy(noise)
                tmp[:, :, :, :, dim] = r
                x_recon = manip_model.predict([x, y, tmp])
                x_recons.append(x_recon)

        x_recons = np.concatenate(x_recons)

        out_img = combine_images(x_recons, height=16)
        out_image = out_img * 4096
        out_image[out_image > 574] = 574
        out_image = out_image / 574 * 255

        Image.fromarray(out_image.astype(np.uint8)).save(os.path.join(manip_out_dir, img[0].split('.')[0] + '_manip_output.png'))

    print('Done.')
