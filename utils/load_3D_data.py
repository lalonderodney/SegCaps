'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for loading training, validation, and testing data into the models.
It is specifically designed to handle 3D single-channel medical data.
Modifications will be needed to train/test on normal 3-channel images.
'''

from __future__ import print_function, division

import threading
import os
import sys
from glob import glob
import csv

from tensorflow.keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom
from sklearn.model_selection import KFold
import numpy as np
from numpy.random import rand, shuffle
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from utils.custom_data_aug import elastic_transform, salt_pepper_noise
from utils.utils import safe_mkdir

debug = 0

def load_data(root, split=0, k_folds=4, val_split=0.1, rand_seed=5):
    # Main functionality of loading and spliting the data
    def _load_data():
        with open(os.path.join(root, 'split_lists', 'train_split_{}.csv'.format(split)), 'r') as f:
            reader = csv.reader(f)
            training_list = list(reader)
        with open(os.path.join(root, 'split_lists', 'test_split_{}.csv'.format(split)), 'r') as f:
            reader = csv.reader(f)
            test_list = list(reader)
        new_train_list, val_list = train_test_split(training_list, test_size=val_split, random_state=rand_seed)
        return new_train_list, val_list, test_list

    # Try-catch to handle calling split data before load only if files are not found.
    try:
        new_training_list, validation_list, testing_list = _load_data()
    except FileNotFoundError:
        # Create the training and test splits if not found
        split_data(root, num_splits=k_folds, rand_seed=rand_seed)
        try:
            new_training_list, validation_list, testing_list = _load_data()
        except FileNotFoundError as e:
            print('Failed to load data, see load_data in load_3D_data.py', e)
            raise

    return new_training_list, validation_list, testing_list

def compute_class_weights(root, train_data_list):
    '''
        We want to weight the the positive pixels by the ratio of negative to positive.
        Three scenarios:
            1. Equal classes. neg/pos ~ 1. Standard binary cross-entropy
            2. Many more negative examples. The network will learn to always output negative. In this way we want to
               increase the punishment for getting a positive wrong that way it will want to put positive more
            3. Many more positive examples. We weight the positive value less so that negatives have a chance.
    '''
    pos = 0.0
    neg = 0.0
    for img_name in tqdm(train_data_list):
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, 'masks', img_name[0])))
        for slic in img:
            if not np.any(slic):
                continue
            else:
                p = np.count_nonzero(slic)
                pos += p
                neg += (slic.size - p)

    return neg/pos

def load_class_weights(root, split):
    class_weight_filename = os.path.join(root, 'split_lists', 'train_split_{}_class_weights.npy'.format(split))
    try:
        return np.load(class_weight_filename)
    except:
        print('Class weight file {} not found.\nComputing class weights now. This may take '
              'some time.'.format(class_weight_filename))
        train_data_list, _, _ = load_data(root, str(split))
        value = compute_class_weights(root, train_data_list)
        np.save(class_weight_filename,value)
        print('Finished computing class weights. This value has been saved for this training split.')
        return value


def split_data(root_path, num_splits=4, rand_seed=5):
    mask_list = []
    for ext in ('*.mhd', '*.hdr', '*.nii', '*.nii.gz'):
        mask_list.extend(sorted(glob(os.path.join(root_path,'masks',ext))))

    assert len(mask_list) != 0, 'Unable to find any files in {}'.format(os.path.join(root_path,'masks'))

    outdir = os.path.join(root_path,'split_lists')
    safe_mkdir(outdir)

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=rand_seed)
    n = 0
    for train_index, test_index in kf.split(mask_list):
        with open(os.path.join(outdir,'train_split_{}.csv'.format(n)), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in train_index:
                writer.writerow([os.path.basename(mask_list[i])])
        with open(os.path.join(outdir,'test_split_{}.csv'.format(n)), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in test_index:
                writer.writerow([os.path.basename(mask_list[i])])
        n += 1


def convert_data_to_numpy(root_path, img_name, no_masks=False, overwrite=False):
    fname = img_name.split('.')[0]
    numpy_path = os.path.join(root_path, 'np_files')
    img_path = os.path.join(root_path, 'imgs')
    mask_path = os.path.join(root_path, 'masks')
    fig_path = os.path.join(root_path, 'figs')
    safe_mkdir(numpy_path)
    safe_mkdir(fig_path)

    ct_min = -1024
    ct_max = 3072

    if not overwrite:
        try:
            with np.load(os.path.join(numpy_path, fname + '.npz')) as data:
                return data['img'], data['mask']
        except:
            pass

    try:
        itk_img = sitk.ReadImage(os.path.join(img_path, img_name))
        img = sitk.GetArrayFromImage(itk_img)
        img = np.rollaxis(img, 0, 3)
        img = img.astype(np.float32)
        img[img > ct_max] = ct_max
        img[img < ct_min] = ct_min
        img += -np.min(img)
        img /= np.max(img)

        if not no_masks:
            itk_mask = sitk.ReadImage(os.path.join(mask_path, img_name))
            mask = sitk.GetArrayFromImage(itk_mask)
            mask = np.rollaxis(mask, 0, 3)
            mask[mask > 250] = 1 # In case using 255 instead of 1
            mask[mask > 4.5] = 0 # Trachea = 5
            mask[mask >= 1] = 1 # Left lung = 3, Right lung = 4
            mask[mask != 1] = 0 # Non-Lung/Background
            mask = mask.astype(np.uint8)

        try:
            f, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(img[:, :, img.shape[2] // 3], cmap='gray')
            if not no_masks:
                ax[0].imshow(mask[:, :, img.shape[2] // 3], alpha=0.15)
            ax[0].set_title('Slice {}/{}'.format(img.shape[2] // 3, img.shape[2]))
            ax[0].axis('off')

            ax[1].imshow(img[:, :, img.shape[2] // 2], cmap='gray')
            if not no_masks:
                ax[1].imshow(mask[:, :, img.shape[2] // 2], alpha=0.15)
            ax[1].set_title('Slice {}/{}'.format(img.shape[2] // 2, img.shape[2]))
            ax[1].axis('off')

            ax[2].imshow(img[:, :, img.shape[2] // 2 + img.shape[2] // 4], cmap='gray')
            if not no_masks:
                ax[2].imshow(mask[:, :, img.shape[2] // 2 + img.shape[2] // 4], alpha=0.15)
            ax[2].set_title('Slice {}/{}'.format(img.shape[2] // 2 + img.shape[2] // 4, img.shape[2]))
            ax[2].axis('off')

            fig = plt.gcf()
            fig.suptitle(fname)

            plt.savefig(os.path.join(fig_path, fname + '.png'), format='png', bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print('\n'+'-'*100)
            print('Error creating qualitative figure for {}'.format(fname))
            print(e)
            print('-'*100+'\n')

        h_pad = int(np.ceil(img.shape[0] / 2**5))*(2**5) - img.shape[0]
        w_pad = int(np.ceil(img.shape[1] / 2**5))*(2**5) - img.shape[1]

        if h_pad != 0 or w_pad != 0:
            img = np.pad(img, ((int(np.ceil(h_pad/2.)), int(np.floor(h_pad/2.))),
                               (int(np.ceil(w_pad/2.)), int(np.floor(w_pad/2.))),
                               (0,0)), 'edge')
            if not no_masks:
                mask = np.pad(mask, ((int(np.ceil(h_pad/2.)), int(np.floor(h_pad/2.))),
                                     (int(np.ceil(w_pad/2.)), int(np.floor(w_pad/2.))),
                                     (0,0)), 'edge')

        if not no_masks:
            np.savez_compressed(os.path.join(numpy_path, fname + '.npz'), img=img, mask=mask)
        else:
            np.savez_compressed(os.path.join(numpy_path, fname + '.npz'), img=img)

        if not no_masks:
            return img, mask
        else:
            return img

    except Exception as e:
        print('\n'+'-'*100)
        print('Unable to load img or masks for {}'.format(fname))
        print(e)
        print('Skipping file')
        print('-'*100+'\n')

        return np.zeros(1), np.zeros(1)

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def augmentImages(batch_of_images, batch_of_masks):
    for i in range(len(batch_of_images)):
        img_and_mask = np.concatenate((batch_of_images[i, ...], batch_of_masks[i,...]), axis=2)
        if img_and_mask.ndim == 4: # This assumes single channel data. For multi-channel you'll need
            # change this to put all channel in slices channel
            orig_shape = img_and_mask.shape
            img_and_mask = img_and_mask.reshape((img_and_mask.shape[0:3]))

        if np.random.randint(0,10) == 7:
            img_and_mask = random_rotation(img_and_mask, rg=45, row_axis=0, col_axis=1, channel_axis=2,
                                           fill_mode='constant', cval=0.)

        if np.random.randint(0, 5) == 3:
            img_and_mask = elastic_transform(img_and_mask, alpha=1000, sigma=80, alpha_affine=50)

        if np.random.randint(0, 10) == 7:
            img_and_mask = random_shift(img_and_mask, wrg=0.2, hrg=0.2, row_axis=0, col_axis=1, channel_axis=2,
                                        fill_mode='constant', cval=0.)

        if np.random.randint(0, 10) == 7:
            img_and_mask = random_shear(img_and_mask, intensity=16, row_axis=0, col_axis=1, channel_axis=2,
                         fill_mode='constant', cval=0.)

        if np.random.randint(0, 10) == 7:
            img_and_mask = random_zoom(img_and_mask, zoom_range=(0.75, 0.75), row_axis=0, col_axis=1, channel_axis=2,
                         fill_mode='constant', cval=0.)

        if np.random.randint(0, 10) == 7:
            img_and_mask = flip_axis(img_and_mask, axis=1)

        if np.random.randint(0, 10) == 7:
            img_and_mask = flip_axis(img_and_mask, axis=0)

        if np.random.randint(0, 10) == 7:
            salt_pepper_noise(img_and_mask, salt=0.2, amount=0.04)

        if batch_of_images.ndim == 4:
            batch_of_images[i, ...] = img_and_mask[...,0:img_and_mask.shape[2]//2]
            batch_of_masks[i,...] = img_and_mask[...,img_and_mask.shape[2]//2:]
        if batch_of_images.ndim == 5:
            img_and_mask = img_and_mask.reshape(orig_shape)
            batch_of_images[i, ...] = img_and_mask[...,0:img_and_mask.shape[2]//2, :]
            batch_of_masks[i,...] = img_and_mask[...,img_and_mask.shape[2]//2:, :]

        # Ensure the masks did not get any non-binary values.
        batch_of_masks[batch_of_masks > 0.5] = 1
        batch_of_masks[batch_of_masks <= 0.5] = 0

    return(batch_of_images, batch_of_masks)


''' Make the generators threadsafe in case of multiple threads '''
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generate_train_batches(root_path, train_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=1, aug_data=1):

    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                file_name = scan_name[0]
                scan_name = os.path.basename(file_name).split('.')[0]
                path_to_np = os.path.join(root_path,'np_files', scan_name+'.npz')
                with np.load(path_to_np) as data:
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name))
                train_img, train_mask = convert_data_to_numpy(root_path, file_name)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1:
                np.random.seed(None)
                subSampAmt = int(rand(1)*(train_img.shape[2]*0.05))

            # Create placeholders for training
            img_batch = np.zeros(((batchSize, train_img.shape[0], train_img.shape[1], net_input_shape[2])), dtype=np.float32)
            mask_batch = np.zeros(((batchSize, train_img.shape[0], train_img.shape[1], net_input_shape[2])), dtype=np.float32)

            indicies = np.arange(0, train_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
            if shuff:
                shuffle(indicies)

            for j in indicies:
                if not np.any(train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    continue
                if img_batch.ndim == 4:
                    img_batch[count, :, :, :] = train_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :] = train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[count, :, :, :, 0] = train_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :, 0] = train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                else:
                    print('Error this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if aug_data:
                        img_batch, mask_batch = augmentImages(img_batch, mask_batch)
                    if debug:
                        if img_batch.ndim == 4:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0]), alpha=0.15)
                        elif img_batch.ndim == 5:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0, 0]), alpha=0.15)
                        plt.savefig(os.path.join(root_path, 'logs', 'ex_train.png'), format='png', bbox_inches='tight')
                        plt.close()
                    if net.find('caps') != -1:
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch*img_batch])
                    elif net.find('phnn') != -1:
                        yield (img_batch, [mask_batch, mask_batch, mask_batch, mask_batch, mask_batch])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            if aug_data:
                img_batch[:count,...], mask_batch[:count,...] = augmentImages(img_batch[:count,...],
                                                                              mask_batch[:count,...])
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            elif net.find('phnn') != -1:
                yield (img_batch[:count, ...], [mask_batch[:count, ...], mask_batch[:count, ...],
                                                mask_batch[:count, ...], mask_batch[:count, ...],
                                                mask_batch[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator
def generate_val_batches(root_path, val_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                         stride=1, downSampAmt=1, shuff=1):

    while True:
        if shuff:
            shuffle(val_list)
        count = 0
        for i, scan_name in enumerate(val_list):
            try:
                file_name = scan_name[0]
                scan_name = os.path.basename(file_name).split('.')[0]
                path_to_np = os.path.join(root_path,'np_files',scan_name+'.npz')
                with np.load(path_to_np) as data:
                    val_img = data['img']
                    val_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name))
                val_img, val_mask = convert_data_to_numpy(root_path, file_name)
                if np.array_equal(val_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1:
                np.random.seed(None)
                subSampAmt = int(rand(1)*(val_img.shape[2]*0.05))

            # Create placeholders for training
            img_batch = np.zeros(((batchSize, val_img.shape[0], val_img.shape[1], net_input_shape[2])), dtype=np.float32)
            mask_batch = np.zeros(((batchSize, val_img.shape[0], val_img.shape[1], net_input_shape[2])), dtype=np.float32)

            indicies = np.arange(0, val_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
            if shuff:
                shuffle(indicies)

            for j in indicies:
                if not np.any(val_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    continue
                if img_batch.ndim == 4:
                    img_batch[count, :, :, :] = val_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :] = val_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[count, :, :, :, 0] = val_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :, 0] = val_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                else:
                    print('Error this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if net.find('caps') != -1:
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch * img_batch])
                    elif net.find('phnn') != -1:
                        yield (img_batch, [mask_batch, mask_batch, mask_batch, mask_batch, mask_batch])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            elif net.find('phnn') != -1:
                yield (img_batch[:count, ...], [mask_batch[:count, ...], mask_batch[:count, ...],
                                                mask_batch[:count, ...], mask_batch[:count, ...],
                                                mask_batch[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator
def generate_test_batches(root_path, test_list, net_input_shape, batchSize=1, numSlices=1, subSampAmt=0,
                          stride=1, downSampAmt=1):
    # Create placeholders for testing
    count = 0
    for i, scan_name in enumerate(test_list):
        try:
            file_name = scan_name[0]
            scan_name = os.path.basename(file_name).split('.')[0]
            path_to_np = os.path.join(root_path,'np_files',scan_name+'.npz')
            with np.load(path_to_np) as data:
                test_img = data['img']
        except:
            print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name))
            test_img = convert_data_to_numpy(root_path, file_name, no_masks=True)
            if np.array_equal(test_img,np.zeros(1)):
                continue
            else:
                print('\nFinished making npz file.')

        if numSlices == 1:
            subSampAmt = 0
        elif subSampAmt == -1 and numSlices > 1:
            np.random.seed(None)
            subSampAmt = int(rand(1)*(test_img.shape[2]*0.05))

        # Create placeholders for training
        img_batch = np.zeros(((batchSize, test_img.shape[0], test_img.shape[1], net_input_shape[2])), dtype=np.float32)

        indicies = np.arange(0, test_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
        for j in indicies:
            if img_batch.ndim == 4:
                img_batch[count, :, :, :] = test_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
            elif img_batch.ndim == 5:
                # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                img_batch[count, :, :, :, 0] = test_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
            else:
                print('Error this function currently only supports 2D and 3D data.')
                exit(0)

            count += 1
            if count % batchSize == 0:
                count = 0
                yield (img_batch)

    if count != 0:
        yield (img_batch[:count,:,:,:])
