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

from os.path import join
from os import makedirs
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


def manip(args, test_list, model_list, net_input_shape):
    if args.weights_path == '':
        weights_path = join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5')
    else:
        weights_path = join(args.data_root_dir, args.weights_path)

    output_dir = join(args.data_root_dir, 'results', args.net, 'split_' + str(args.split_num))
    manip_out_dir = join(output_dir, 'manip_output')
    try:
        makedirs(manip_out_dir)
    except:
        pass

    assert(len(model_list) == 3), "Must be using segcaps with the three models."
    manip_model = model_list[2]
    try:
        manip_model.load_weights(weights_path)
    except:
        print('Unable to find weights path. Testing with random weights.')
    print_summary(model=manip_model, positions=[.38, .65, .75, 1.])


    # Manipulating capsule vectors
    print('Testing... This will take some time...')

    for i, img in enumerate(tqdm(test_list)):
        sitk_img = sitk.ReadImage(join(args.data_root_dir, 'imgs', img[0]))
        img_data = sitk.GetArrayFromImage(sitk_img)
        num_slices = img_data.shape[0]
        sitk_mask = sitk.ReadImage(join(args.data_root_dir, 'masks', img[0]))
        gt_data = sitk.GetArrayFromImage(sitk_mask)

        x, y = img_data[num_slices//2, :, :], gt_data[num_slices//2, :, :]
        x, y = np.expand_dims(np.expand_dims(x, -1), 0), np.expand_dims(np.expand_dims(y, -1), 0)

        noise = np.zeros([1, 512, 512, 1, 16])
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

        Image.fromarray(out_image.astype(np.uint8)).save(join(manip_out_dir, img[0][:-4] + '_manip_output.png'))

    print('Done.')
