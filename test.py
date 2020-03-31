'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for testing models. Please see the README for details about testing.
'''

from __future__ import print_function, division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from os.path import join, basename
from os import makedirs
import csv
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import scipy.ndimage.morphology
from skimage import measure, filters
from utils.metrics import dc, jc, assd, hd

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
        if props[0].area / props[1].area > 5:  # if the largest is way larger than the second largest
            thresholded_mask[all_labels == props[0].label] = 1  # only turn on the largest component
        else:
            thresholded_mask[all_labels == props[0].label] = 1  # turn on two largest components
            thresholded_mask[all_labels == props[1].label] = 1
    elif len(props):
        thresholded_mask[all_labels == props[0].label] = 1

    thresholded_mask = scipy.ndimage.morphology.binary_fill_holes(thresholded_mask).astype(np.uint8)

    return thresholded_mask


def test(args, test_list, model_list, net_input_shape):
    if len(model_list) > 1:
        eval_model = model_list[1]
    else:
        eval_model = model_list[0]

    out_net_name = args.net
    if args.net.find('caps') != -1 and args.net != 'segcapsbasic':
        out_net_name = out_net_name + 'r{}r{}'.format(args.r1, args.r2)

    if args.weights_path != '':
        output_dir = join(args.data_root_dir, 'results', out_net_name, 'split_{}'.format(args.split_num),
                          basename(args.weights_path)[-24:-5])
        try:
            eval_model.load_weights(args.weights_path)
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')
    else:
        output_dir = join(args.data_root_dir, 'results', out_net_name, 'split_{}'.format(args.split_num), args.time)
        try:
            eval_model.load_weights(join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5'))
        except Exception as e:
            print(e)
            raise Exception('Failed to load weights from training.')

    raw_out_dir = join(output_dir, 'raw_output')
    fin_out_dir = join(output_dir, 'final_output')
    fig_out_dir = join(output_dir, 'qual_figs')
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

    # Set up placeholders
    outfile = ''
    if args.compute_dice:
        dice_arr = np.zeros((len(test_list)))
        outfile += 'dice_'
    if args.compute_jaccard:
        jacc_arr = np.zeros((len(test_list)))
        outfile += 'jacc_'
    if args.compute_assd:
        assd_arr = np.zeros((len(test_list)))
        outfile += 'assd_'
    if args.compute_hd:
        hd_arr = np.zeros((len(test_list)))
        outfile += 'hd_'

    # Testing the network
    print('Testing... This will take some time...')

    with open(join(output_dir, args.save_prefix + outfile + 'scores.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        row = ['Scan Name']
        if args.compute_dice:
            row.append('Dice Coefficient')
        if args.compute_jaccard:
            row.append('Jaccard Index')
        if args.compute_assd:
            row.append('Average Symmetric Surface Distance')
        if args.compute_hd:
            row.append('Hausdorff Distance')

        writer.writerow(row)

        for i, img in enumerate(tqdm(test_list)):
            sitk_img = sitk.ReadImage(join(args.data_root_dir, 'imgs', img[0]))
            img_data = sitk.GetArrayFromImage(sitk_img)
            temp1 = img[0].split('.')
            img_name = temp1[0]
            img_ext = '.'.join(temp1[1:])

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
            elif args.net.find('phnn') != -1:
                output = output_array[-1][:, :, :, 0]
            else:
                output = output_array[:,:,:,0]

            h_pad = int(np.ceil(img_data.shape[1] / 2 ** 5)) * (2 ** 5) - img_data.shape[1]
            w_pad = int(np.ceil(img_data.shape[2] / 2 ** 5)) * (2 ** 5) - img_data.shape[2]

            if h_pad == 1:
                output = output[:, 1:, :]
            elif h_pad > 1:
                output = output[:, int(np.ceil(h_pad / 2.)):-int(np.floor(h_pad / 2.)), :]

            if w_pad == 1:
                output = output[:, :,1:]
            elif w_pad > 1:
                output = output[:, :, int(np.ceil(w_pad / 2.)):-int(np.floor(w_pad / 2.))]

            output_img = sitk.GetImageFromArray(output)
            print('Segmenting Output')
            output_bin = threshold_mask(output, args.thresh_level)
            output_mask = sitk.GetImageFromArray(output_bin)

            output_img.CopyInformation(sitk_img)
            output_mask.CopyInformation(sitk_img)

            print('Saving Output')
            if args.save_raw:
                sitk.WriteImage(output_img, join(raw_out_dir, img_name + '_raw_output.' + img_ext))
            if args.save_seg:
                sitk.WriteImage(output_mask, join(fin_out_dir, img_name + '_final_output.' + img_ext))

            # Load gt mask
            sitk_mask = sitk.ReadImage(join(args.data_root_dir, 'masks', img[0]))
            gt_data = sitk.GetArrayFromImage(sitk_mask)

            # Plot Qual Figure
            print('Creating Qualitative Figure for Quick Reference')
            f, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(img_data[img_data.shape[0] // 3, :, :], alpha=1, cmap='gray')
            ax[0].imshow(output_bin[img_data.shape[0] // 3, :, :], alpha=0.5, cmap='Blues')
            ax[0].imshow(gt_data[img_data.shape[0] // 3, :, :], alpha=0.2, cmap='Reds')
            ax[0].set_title('Slice {}/{}'.format(img_data.shape[0] // 3, img_data.shape[0]))
            ax[0].axis('off')

            ax[1].imshow(img_data[img_data.shape[0] // 2, :, :], alpha=1, cmap='gray')
            ax[1].imshow(output_bin[img_data.shape[0] // 2, :, :], alpha=0.5, cmap='Blues')
            ax[1].imshow(gt_data[img_data.shape[0] // 2, :, :], alpha=0.2, cmap='Reds')
            ax[1].set_title('Slice {}/{}'.format(img_data.shape[0] // 2, img_data.shape[0]))
            ax[1].axis('off')

            ax[2].imshow(img_data[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=1, cmap='gray')
            ax[2].imshow(output_bin[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=0.5,
                         cmap='Blues')
            ax[2].imshow(gt_data[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=0.2,
                         cmap='Reds')
            ax[2].set_title(
                'Slice {}/{}'.format(img_data.shape[0] // 2 + img_data.shape[0] // 4, img_data.shape[0]))
            ax[2].axis('off')

            fig = plt.gcf()
            fig.suptitle(img_name)

            plt.savefig(join(fig_out_dir, img_name + '_qual_fig' + '.png'),
                        format='png', bbox_inches='tight')
            plt.close('all')

            row = [img_name]
            if args.compute_dice:
                print('Computing Dice')
                try:
                    dice_arr[i] = dc(output_bin, gt_data)
                except:
                    dice_arr[i] = 0
                print('\tDice: {}'.format(dice_arr[i]))
                row.append(dice_arr[i])
            if args.compute_jaccard:
                print('Computing Jaccard')
                try:
                    jacc_arr[i] = jc(output_bin, gt_data)
                except:
                    jacc_arr[i] = 0
                print('\tJaccard: {}'.format(jacc_arr[i]))
                row.append(jacc_arr[i])
            if args.compute_assd:
                print('Computing ASSD')
                try:
                    assd_arr[i] = assd(output_bin, gt_data, voxelspacing=sitk_img.GetSpacing(), connectivity=1)
                except:
                    assd_arr[i] = np.nan
                print('\tASSD: {}'.format(assd_arr[i]))
                row.append(assd_arr[i])
            if args.compute_hd:
                print('Computing HD')
                try:
                    hd_arr[i] = hd(output_bin, gt_data, voxelspacing=sitk_img.GetSpacing(), connectivity=1)
                except:
                    hd_arr[i] = np.nan
                print('\tHD: {}'.format(hd_arr[i]))
                row.append(hd_arr[i])

            writer.writerow(row)

        row = ['Average Scores']
        if args.compute_dice:
            row.append(np.nanmean(dice_arr))
        if args.compute_jaccard:
            row.append(np.nanmean(jacc_arr))
        if args.compute_assd:
            row.append(np.nanmean(assd_arr))
        if args.compute_hd:
            row.append(np.nanmean(hd_arr))
        writer.writerow(row)

    print('Done.')
