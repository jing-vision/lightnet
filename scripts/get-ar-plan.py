import requests
import os
import shutil,os

from os.path import join

import numpy as np
import cv2 as cv
import configparser
from collections import OrderedDict

this_script_path = os.path.realpath(__file__) 
scripts_folder = os.path.dirname(this_script_path)
lightnet_folder = os.path.dirname(scripts_folder)

def copytree2(source,dest):
    os.mkdir(dest)
    dest_dir = join(dest,os.path.basename(source))
    shutil.copytree(source,dest_dir)

def mkdir2(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

url = 'http://localhost:6666/ar/plan'

response = requests.get(url)
result = response.json()

for plan in result:
    mkdir2(plan['plan_name'])
    for group in plan['groups']:
        group_path = join(plan['plan_name'], group['group_name'])
        shutil.copytree(join(lightnet_folder, '__template-darknet19_448'), group_path)

        # file names config
        train_txt = 'train.txt'
        valid_txt = 'valid.txt'
        obj_data = 'obj.data'
        obj_names = 'obj.names'
        obj_cfg = 'obj.cfg'

        train_txt_fp = open(join(group_path, train_txt), 'w')
        valid_txt_fp = open(join(group_path, valid_txt), 'w')
        obj_names_fp = open(join(group_path, obj_names), 'w')

        label_dict = {}

        for sku in group['skus']:
            label_dict[sku['sku_code']] = ''
            train_txt_fp.write(sku['image_path'])
            train_txt_fp.write('\n')

            # same files as train_txt
            # since we have limited # of images now....
            valid_txt_fp.write(sku['image_path'])
            valid_txt_fp.write('\n')

        num_classes = len(label_dict.keys()) 
        for label in label_dict.keys():
            obj_names_fp.write(label)
            obj_names_fp.write('\n')
        

        with open(join(group_path, obj_data), 'w') as obj_data_fp:
            obj_data_fp.write('classes=%d\n' % num_classes)
            obj_data_fp.write('train=%s\n' % join('.', train_txt))
            obj_data_fp.write('valid=%s\n' % join('.', valid_txt))
            obj_data_fp.write('labels=%s\n' % join('.', obj_names))
            obj_data_fp.write('names=%s\n' % join('.', obj_names))
            obj_data_fp.write('backup=%s/weights/\n' % '.')
            obj_data_fp.write('top=5\n')

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
        config.read(join(group_path, obj_cfg))
        print(config.sections())
        config._sections['net1']['batch'] = 32
        config._sections['net1']['subdivisions'] = 4
        config._sections['net1']['max_batches'] = 4000
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

        obj_cfg_fp = open(join(group_path, obj_cfg), 'w')
        for key in config._sections:
            section = config._sections[key]
            obj_cfg_fp.write('[%s]\n' % section['original_key'])
            del section['original_key']
            for field in section:
                obj_cfg_fp.write('%s=%s\n' % (field, section[field]))
            obj_cfg_fp.write('\n')        