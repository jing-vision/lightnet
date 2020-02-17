#!/usr/bin/env python

'''
'''

# Python 2/3 compatibility
from __future__ import print_function
from collections import OrderedDict
import configparser
import cv2 as cv
import numpy as np
import glob
import os
import sys
import random
from pathlib import Path

# file names config
category_folders = glob.glob('img/*')
train_txt = 'train.txt'
valid_txt = 'valid.txt'
obj_data = 'obj.data'
obj_names = 'obj.names'
obj_sku = 'obj.sku'
obj_cfg = 'obj.cfg'


def main():
    print(__doc__)

    cwd = '.'  # os.getcwd()

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
    obj_sku_fp = open(obj_sku, 'w')

    obj_sku_set = set()

    idx = 0
    for category in category_folders:
        obj_names_fp.write(category)
        obj_names_fp.write('\n')
        image_filenames = glob.glob(category + '/**/*.jpg', recursive=True)
        image_filenames.extend(
            glob.glob(category + '/**/*.png', recursive=True))

        for image_filename in image_filenames:
            path = Path(image_filename)
            obj_sku_set.add(os.path.basename(path.parent))

            if random.random() < 0.1:  # 10% for validation
                valid_txt_fp.write(os.path.abspath(image_filename))
                valid_txt_fp.write('\n')
            else:
                train_txt_fp.write(os.path.abspath(image_filename))
                train_txt_fp.write('\n')
            idx = idx + 1

    for sku in obj_sku_set:
        obj_sku_fp.write(str(sku))
        obj_sku_fp.write('\n')

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
    config._sections['net1']['batch'] = 64
    config._sections['net1']['subdivisions'] = 4
    config._sections['net1']['max_batches'] = 5000
    config._sections['net1']['learning_rate'] = 0.001

    conv_reverse_id = 0
    for key in reversed(config._sections):
        if 'convolutional' in key:
            if conv_reverse_id is 0:
                # The last CONV layer specify filters numbers
                config._sections[key]['filters'] = num_classes
            elif conv_reverse_id is 1:
                # Freeze penultimate CONV layer for fine-tuning
                # https://github.com/AlexeyAB/darknet/issues/1061#issuecomment-399083012
                config._sections[key]['stopbackward'] = 1

            conv_reverse_id += 1

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
