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
import configparser
from collections import OrderedDict

# file names config
category_folders = glob.glob('img/*')
train_txt = 'train.txt'
valid_txt = 'valid.txt'
obj_data = 'obj.data'
obj_names = 'obj.names'
obj_cfg = 'obj.cfg'

def main():
    print(__doc__)

    cwd = '.' #os.getcwd()

    num_classes = len(category_folders)

    with open(obj_data, 'w') as obj_data_fp:
        obj_data_fp.write('classes=%d\n' % num_classes)
        obj_data_fp.write('train=%s/%s\n' % (cwd, train_txt))
        obj_data_fp.write('valid=%s/%s\n' % (cwd, valid_txt))
        obj_data_fp.write('labels=%s/obj.names\n' % (cwd))
        obj_data_fp.write('names=%s/obj.names\n' % (cwd))
        obj_data_fp.write('backup=%s/weights/\n' % (cwd))
        obj_data_fp.write('top=5\n')

    train_txt_fp = open(train_txt, 'w')
    valid_txt_fp = open(valid_txt, 'w')
    obj_names_fp = open(obj_names, 'w')

    idx = 0
    for category in category_folders:
        obj_names_fp.write(category)
        obj_names_fp.write('\n')
        image_filenames = glob.glob(category + '/**/*.jpg', recursive=True)
        image_filenames.extend(glob.glob(category + '/**/*.png', recursive=True))

        for image_filename in image_filenames:
            if idx % 100 < 1: # 1% for validation
                valid_txt_fp.write(os.path.abspath(image_filename))
                valid_txt_fp.write('\n')
            else:
                train_txt_fp.write(os.path.abspath(image_filename))
                train_txt_fp.write('\n')
            idx = idx + 1

    class multidict(OrderedDict):
        _unique = 0   # class variable

        def __setitem__(self, key, val):
            if isinstance(val, dict):
                self._unique += 1
                val['original_key'] = key
                key += str(self._unique)
            OrderedDict.__setitem__(self, key, val)
    config = configparser.ConfigParser(
        defaults=None, dict_type=multidict, strict=False)
    config.read(obj_cfg)
    print(config.sections())
    config._sections['net1']['batch'] = 32
    config._sections['net1']['subdivisions'] = 4
    config._sections['net1']['max_batches'] = 10000
    config._sections['net1']['learning_rate'] = 0.001

    for key in reversed(config._sections):
        # find the last CONV layer
        if 'convolutional' in key:
            # modify its filters field
            config._sections[key]['filters'] = num_classes
            break

    with open(obj_cfg, 'w') as configfile:
        for key in config._sections:
            section = config._sections[key]
            configfile.write('[%s]\n' % section['original_key'])
            del section['original_key']
            for field in section:
                configfile.write('%s=%s\n' % (field, section[field]))
            configfile.write('\n')

if __name__ == '__main__':
    main()
