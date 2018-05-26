#!/usr/bin/env python

'''
'''

# Python 2/3 compatibility
from __future__ import print_function
import glob
import os
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

# file names config
category_folders = glob.glob('img/*')
g_category_id = None
g_image_filenames = []
train_txt = 'train.txt'
obj_data = 'obj.data'
obj_names = 'obj.names'

def update_category(category_id):
    global g_image_filenames, g_category_id
    g_category_id = category_id
    g_image_filenames = glob.glob(category_folders[category_id] + '/*.jpg')

def main():
    print(__doc__)

    cwd = os.getcwd()

    with open(obj_data, 'w') as obj_data_fp:
        obj_data_fp.write('classes=%d\n' % len(category_folders))
        obj_data_fp.write('train  =%s/train.txt\n' % (cwd))
        obj_data_fp.write('valid  =%s/train.txt\n' % (cwd))
        obj_data_fp.write('labels =%s/obj.names\n' % (cwd))
        obj_data_fp.write('names =%s/obj.names\n' % (cwd))
        obj_data_fp.write('backup =%s/weights/\n' % (cwd))

    train_txt_fp = open(train_txt, 'w')
    obj_names_fp = open(obj_names, 'w')

    for category in category_folders:
        obj_names_fp.write(category)
        obj_names_fp.write('\n')
        image_filenames = glob.glob(category + '/*.jpg')

        for image_filename in image_filenames:
            train_txt_fp.write(os.path.abspath(image_filename))
            train_txt_fp.write('\n')

if __name__ == '__main__':
    main()
