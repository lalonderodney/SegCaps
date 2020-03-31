'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is simple utils file for commonly used simple functions.
Please see the README for detailed instructions for this project.
'''

import os
import sys
import errno


def safe_mkdir(dir_to_make: str) -> None:
    '''
    Attempts to make a directory following the Pythonic EAFP strategy which prevents race conditions.

    :param dir_to_make: The directory path to attempt to make.
    :return: None
    '''
    try:
        os.makedirs(dir_to_make)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('ERROR: Unable to create directory: {}'.format(dir_to_make), e)
            raise
