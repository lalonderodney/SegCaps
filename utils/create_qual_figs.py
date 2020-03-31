'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for creating qualitative figures.
'''

import os
import csv
from glob import glob
import errno

import SimpleITK as sitk
import numpy as np
from tqdm import trange
from skimage.measure import find_contours
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
params = {'figure.figsize': (15, 15),
          'legend.loc': 'lower left',
          'legend.fontsize': 18,
          'axes.labelweight': 'bold'}
pylab.rcParams.update(params)
legend_properties = {'weight':'bold'}

# Nets, datasets, paths, etc.
nets_list = ['unet', 'tiramisu103', 'phnn', 'segcapsr3r3']
colors_dict = {'segcapsr3r3': 'c', 'unet': 'y', 'tiramisu103': 'g', 'phnn': 'r'}
alphas_dict = {'segcapsr3r3': 0.75, 'unet': 0.5, 'tiramisu103': 0.5, 'phnn': 0.4}
nets_names = {'segcapsr3r3': 'SegCaps:        ', 'unet': 'U-Net:             ', 'tiramisu103': 'Tiramisu:        ', 'phnn': 'P-HNN:            '}
datasets_list = ['LIDC-IDRI-Seg', 'LTRC', 'ILD_HUG', 'Mice', 'Mice/JHU_Clofazimine']
datasets_dict = {'LIDC-IDRI-Seg': 'LIDC-IDRI', 'LTRC': 'LTRC', 'ILD_HUG': 'UHG', 'Mice': 'JHU-TBS', 'Mice/JHU_Clofazimine': 'JHU-DRTB'}
root_dir = '/home/rodney/mnt/hinton/MedicalProjects/Data/Lungs'

for dataset in datasets_list:
    img_list = glob(os.path.join(root_dir, dataset, 'results', 'segcapsr3r3', 'split_*', '*', 'final_output', '*.nii.gz'))
    img_list.extend(glob(os.path.join(root_dir, dataset, 'results', 'segcapsr3r3', 'split_*', '*', 'final_output', '*.hdr')))

    for img_path in img_list:
        # Grab image name without extension
        img_name = os.path.basename(img_path).split('_final_output')[0]
        out_dir = os.path.join(root_dir, 'qual_figs', '{}_{}'.format(datasets_dict[dataset], img_name))

        # Create the output directory in root_dir
        try:
            os.makedirs(out_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # raises the error again

        # Load the image and mask and convert them to numpy
        try:
            itk_img = sitk.ReadImage(glob(os.path.join(root_dir, dataset, 'imgs', '{}.nii.gz'.format(img_name)))[0])
        except:
            itk_img = sitk.ReadImage(glob(os.path.join(root_dir, dataset, 'imgs', '{}.hdr'.format(img_name)))[0])
        np_img = sitk.GetArrayFromImage(itk_img)
        np_img[np_img < -1024] = -1024
        np_img[np_img > 3072] = 3072
        try:
            itk_mask = sitk.ReadImage(glob(os.path.join(root_dir, dataset, 'masks', '{}.nii.gz'.format(img_name)))[0])
        except:
            itk_mask = sitk.ReadImage(glob(os.path.join(root_dir, dataset, 'masks', '{}.hdr'.format(img_name)))[0])
        np_mask = sitk.GetArrayFromImage(itk_mask)

        # Preload the results
        results_list = []
        for net in nets_list:
            itk_result = sitk.ReadImage(glob(os.path.join(root_dir, dataset, 'results', net, 'split_*', '*',
                                                          'final_output', os.path.basename(img_path)))[0])
            np_result = sitk.GetArrayFromImage(itk_result)
            results_list.append(np_result)

        # Loop over the slices of the scan
        for slice_idx in trange(np_img.shape[0]//3, np_img.shape[0] - np_img.shape[0]//3, 1,
                                desc='Looping over slices of {}'.format(datasets_dict[dataset])):
            # Create figure to be filled in
            fig, ax = plt.subplots()
            # fig.suptitle('{}'.format(datasets_dict[dataset]), y=0.1) # Just add this in PowerPoint
            ax.axis('off')

            img_slice = np_img[slice_idx, :, :]
            mask_slice = np_mask[slice_idx, :, :]
            if dataset == 'LTRC' or dataset == 'ILD_HUG':
                img_slice = np.flipud(img_slice)
                mask_slice = np.flipud(mask_slice)
            mask_outline = find_contours(mask_slice, 0.8)

            # Plot the image and the ground truth outline
            ax.imshow(img_slice, cmap='gray')
            for i, outline in enumerate(mask_outline):
                if i == 0:
                    ax.plot(outline[:, 1], outline[:, 0], linewidth=3, color='m', alpha=0.75,
                            label='Ground Truth: Dice %  IoU %   HD (mm)')
                else:
                    ax.plot(outline[:, 1], outline[:, 0], linewidth=3, color='m', alpha=0.75)

            # Loop over the network results and plot their outlines as well
            for idx, net in enumerate(nets_list):
                result_slice = results_list[idx][slice_idx, :, :]
                if dataset == 'LTRC' or dataset == 'ILD_HUG':
                    result_slice = np.flipud(result_slice)
                result_outline = find_contours(result_slice, 0.8)

                # Load the csv file for the results
                with open(glob(os.path.join(root_dir, dataset, 'results', net, 'split_*', '*',
                                            'qual_img_dice_jacc_hd_scores.csv'))[0], 'rb') as csvfile:
                    r = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for row in r:
                        row_vals = row[0].split(',')
                        if row_vals[0] == img_name:
                            result_scores = row_vals[1:]

                # Plot the result outline
                for i, outline in enumerate(result_outline):
                    if i == 0:
                        ax.plot(outline[:, 1], outline[:, 0], linewidth=2, color=colors_dict[net], alpha=alphas_dict[net],
                                label='{} {:.2f}    {:.2f}   {:.3f}'.format(
                                    nets_names[net], float(result_scores[0])*100,
                                    float(result_scores[1])*100, float(result_scores[2])))
                    else:
                        ax.plot(outline[:, 1], outline[:, 0], linewidth=3, color=colors_dict[net], alpha=alphas_dict[net])

            # Add the labels to the legend and save the plot
            ax.legend(prop=legend_properties)
            plt.savefig(os.path.join(out_dir, 'qual_fig_{}_{}_{}.png'.format(datasets_dict[dataset],
                                                                             img_name, slice_idx)), dpi=100)
            plt.close()
