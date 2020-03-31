'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for predicting on new data which do not have GTs. Please see the README for details about predicting.
'''

from __future__ import print_function, division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from os.path import join, basename
from os import makedirs
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import scipy.ndimage.morphology
from skimage import measure, filters

from keras import backend as K
K.set_image_data_format('channels_last')

from utils.load_3D_data import generate_test_batches


def threshold_mask(raw_output, threshold):
    if threshold == 0:
        try:
            threshold = filters.threshold_otsu(raw_output)
        except:
            threshold = 0.5

    print('\tThreshold: {}'.format(threshold))

    raw_output[raw_output > threshold] = 1
    raw_output[raw_output < 1] = 0

    all_labels = measure.label(raw_output)
    props = measure.regionprops(all_labels)
    props.sort(key=lambda x: x.area, reverse=True)
    thresholded_mask = np.zeros(raw_output.shape)

    if len(props) >= 2:
        if props[0].area / props[1].area > 2:  # if the largest is way larger than the second largest
            thresholded_mask[all_labels == props[0].label] = 1  # only turn on the largest component
        else:
            thresholded_mask[all_labels == props[0].label] = 1  # turn on two largest components
            thresholded_mask[all_labels == props[1].label] = 1
    elif len(props):
        thresholded_mask[all_labels == props[0].label] = 1

    thresholded_mask = scipy.ndimage.morphology.binary_fill_holes(thresholded_mask).astype(np.uint8)

    return thresholded_mask


def predict(args, pred_list, model_list, net_input_shape):
    if len(model_list) > 1:
        eval_model = model_list[1]
    else:
        eval_model = model_list[0]

    if args.weights_path != '':
        output_dir = join(args.data_root_dir, 'results', args.net, 'split_{}'.format(args.split_num), basename(args.weights_path)[-24:-5])
        try:
            eval_model.load_weights(args.weights_path)
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')
    else:
        output_dir = join(args.data_root_dir, 'results', args.net, 'split_{}'.format(args.split_num), args.time)
        try:
            eval_model.load_weights(join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5'))
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')

    raw_out_dir = join(output_dir, 'pred_raw_output')
    fin_out_dir = join(output_dir, 'pred_final_output')
    fig_out_dir = join(output_dir, 'pred_qual_figs')
    try:
        makedirs(raw_out_dir)
    except:
        pass
    try:
        makedirs(fin_out_dir)
    except:
        pass
    try:
        makedirs(fig_out_dir)
    except:
        pass

    # Predicting on the new data
    print('Predicting... This will take some time...')

    for i, img in enumerate(tqdm(pred_list)):
        sitk_img = sitk.ReadImage(join(args.data_root_dir, 'imgs', img[0]))
        img_data = sitk.GetArrayFromImage(sitk_img)
        num_slices = img_data.shape[0]

        output_array = eval_model.predict_generator(generate_test_batches(args.data_root_dir, [img],
                                                                          net_input_shape,
                                                                          batchSize=args.batch_size,
                                                                          numSlices=args.slices,
                                                                          subSampAmt=0,
                                                                          stride=1),
                                                    steps=num_slices, max_queue_size=1, workers=1,
                                                    use_multiprocessing=False, verbose=1)

        if args.net.find('caps') != -1:
            output = output_array[0][:,:,:,0]
            #recon = output_array[1][:,:,:,0]
        else:
            output = output_array[:,:,:,0]

        h_pad = int(np.ceil(img_data.shape[1] / 2 ** 5)) * (2 ** 5) - img_data.shape[1]
        w_pad = int(np.ceil(img_data.shape[2] / 2 ** 5)) * (2 ** 5) - img_data.shape[2]
        if h_pad != 0 or w_pad != 0:
            if h_pad == 0:
                if int(np.floor(w_pad / 2.)) == 0:
                    output = output[:, :, int(np.ceil(w_pad / 2.)):]
                else:
                    output = output[:, :, int(np.ceil(w_pad / 2.)):-int(np.floor(w_pad / 2.))]
            elif w_pad == 0:
                if int(np.floor(h_pad / 2.)) == 0:
                    output = output[:, int(np.ceil(h_pad / 2.)):-int(np.floor(h_pad / 2.)),:]
                else:
                    output = output[:, int(np.ceil(h_pad / 2.)):-int(np.floor(h_pad / 2.)),:]
            else:
                if int(np.floor(w_pad / 2.)) == 0 and int(np.floor(h_pad / 2.)) == 0:
                    output = output[:, int(np.ceil(h_pad / 2.)):, int(np.ceil(w_pad / 2.)):]
                elif int(np.floor(w_pad / 2.)) == 0:
                    output = output[:, int(np.ceil(h_pad / 2.)):-int(np.floor(h_pad / 2.)), int(np.ceil(w_pad / 2.)):]
                elif int(np.floor(h_pad / 2.)) == 0:
                    output = output[:, int(np.ceil(h_pad / 2.)):, int(np.ceil(w_pad / 2.)):-int(np.floor(w_pad / 2.))]
                else:
                    output = output[:, int(np.ceil(h_pad / 2.)):-int(np.floor(h_pad / 2.)),
                             int(np.ceil(w_pad / 2.)):-int(np.floor(w_pad / 2.))]

        output_img = sitk.GetImageFromArray(output)
        print('Segmenting Output')
        output_bin = threshold_mask(output, args.thresh_level)
        output_mask = sitk.GetImageFromArray(output_bin)

        output_img.CopyInformation(sitk_img)
        output_mask.CopyInformation(sitk_img)

        print('Saving Output')
        sitk.WriteImage(output_img, join(raw_out_dir, img[0][:-4] + '_raw_output' + img[0][-4:]))
        sitk.WriteImage(output_mask, join(fin_out_dir, img[0][:-4] + '_final_output' + img[0][-4:]))

        # Plot Qual Figure
        print('Creating Qualitative Figure for Quick Reference')
        f, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(img_data[img_data.shape[0] // 3, :, :], alpha=1, cmap='gray')
        ax[0].imshow(output_bin[img_data.shape[0] // 3, :, :], alpha=0.5, cmap='viridis')
        ax[0].set_title('Slice {}/{}'.format(img_data.shape[0] // 3, img_data.shape[0]))
        ax[0].axis('off')

        ax[1].imshow(img_data[img_data.shape[0] // 2, :, :], alpha=1, cmap='gray')
        ax[1].imshow(output_bin[img_data.shape[0] // 2, :, :], alpha=0.5, cmap='viridis')
        ax[1].set_title('Slice {}/{}'.format(img_data.shape[0] // 2, img_data.shape[0]))
        ax[1].axis('off')

        ax[2].imshow(img_data[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=1, cmap='gray')
        ax[2].imshow(output_bin[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=0.5,
                     cmap='viridis')
        ax[2].set_title(
            'Slice {}/{}'.format(img_data.shape[0] // 2 + img_data.shape[0] // 4, img_data.shape[0]))
        ax[2].axis('off')

        fig = plt.gcf()
        fig.suptitle(img[0][:-4])

        plt.savefig(join(fig_out_dir, img[0][:-4] + '_qual_fig' + '.png'),
                    format='png', bbox_inches='tight')
        plt.close('all')

    print('Done.')
