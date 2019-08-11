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

    with open(obj_names, 'r') as obj_names_fp:
        num_classes = 2

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

    network_name = 'N/A'
    for key in reversed(config._sections):
        if 'region' in key:
            network_name = 'yolov2'
            config._sections['net1']['subdivisions'] = 8 # workaround for GTX 1060 6G
            break
        if 'yolo' in key:
            network_name = 'yolov3'
            config._sections['net1']['subdivisions'] = 16 # workaround for GTX 1060 6G
            break

    if network_name == 'yolov2':
        for key in reversed(config._sections):
            # set number of classes (objects) in obj.cfg#L230
            if 'region' in key:
                config._sections[key]['classes'] = num_classes

            # set `filter`-value equal to `(classes + 5)*5` in obj.cfg#L224
            if 'convolutional' in key:
                network_name = 'yolov3'
                config._sections[key]['filters'] = (num_classes + 5) * 5
                break
    elif network_name == 'yolov3':
        seen_yolo_layer = False
        for key in reversed(config._sections):
            # change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers:
            if 'yolo' in key:
                config._sections[key]['classes'] = num_classes
                seen_yolo_layer = True

            # change [`filters=255`] to filters=(classes + 5)x3 in the 3 `[convolutional]` before each `[yolo]` layer
            if seen_yolo_layer and 'convolutional' in key:
                seen_yolo_layer = False
                network_name = 'yolov3'
                config._sections[key]['filters'] = (num_classes + 5) * 3

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
