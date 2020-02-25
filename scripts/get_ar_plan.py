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

def test():
    url = 'http://localhost:6666/ar/plan'

    response = requests.get(url)
    plan_json = response.json()
    prepare_training_folders(plan_json)

def get_training_metas(plan_json):
    training_metas = []

    # global_train_bat_fp = open(join(lightnet_folder, train_bat), 'w')
    for plan in plan_json:

        for group in plan['groups']:
            abs_group_path = join(lightnet_folder, plan['plan_name'], group['group_name'])
            training_metas.append({
                'group': group['group_name'],
                'plan': plan['plan_name'],
                'folder': abs_group_path
            })
    return training_metas

# training_folders: an array of training folders
def prepare_training_folders(plan_json, subdivisions = 8, max_batches = 1500):
    training_folders = []
    train_bat = 'train.bat'

    # global_train_bat_fp = open(join(lightnet_folder, train_bat), 'w')
    for plan in plan_json:
        mkdir2(join(lightnet_folder, plan['plan_name']))
        # global_train_bat_fp.write('pushd ' + plan['plan_name'])
        # global_train_bat_fp.write('\n')
        # global_train_bat_fp.write('call ' + train_bat)
        # global_train_bat_fp.write('\n')
        # global_train_bat_fp.write('popd')
        # global_train_bat_fp.write('\n')

        plan_train_bat_fp = open(join(lightnet_folder, plan['plan_name'], train_bat), 'w')
        for group in plan['groups']:
            plan_train_bat_fp.write('pushd ' + group['group_name'])
            plan_train_bat_fp.write('\n')
            plan_train_bat_fp.write('call ' + train_bat)
            plan_train_bat_fp.write('\n')
            plan_train_bat_fp.write('popd')
            plan_train_bat_fp.write('\n')
            abs_group_path = join(lightnet_folder, plan['plan_name'], group['group_name'])
            training_folders.append(abs_group_path)

            shutil.copytree(join(scripts_folder, 'template-darknet19_448'), abs_group_path)

            # file names config
            train_txt = 'train.txt'
            valid_txt = 'valid.txt'
            obj_data = 'obj.data'
            obj_names = 'obj.names'
            obj_cfg = 'obj.cfg'

            train_txt_fp = open(join(abs_group_path, train_txt), 'w')
            valid_txt_fp = open(join(abs_group_path, valid_txt), 'w')
            obj_names_fp = open(join(abs_group_path, obj_names), 'w')

            label_dict = {}

            idx = 0
            for sku in group['skus']:
                label_dict[sku['sku_code']] = ''

                if idx % 10 == 0:
                    # 10%
                    valid_txt_fp.write(sku['image_path'])
                    valid_txt_fp.write('\n')
                else:
                    train_txt_fp.write(sku['image_path'])
                    train_txt_fp.write('\n')

                idx += 1

            num_classes = len(label_dict.keys()) 
            for label in label_dict.keys():
                obj_names_fp.write(label)
                obj_names_fp.write('\n')
            

            with open(join(abs_group_path, obj_data), 'w') as obj_data_fp:
                obj_data_fp.write('classes=%d\n' % num_classes)
                obj_data_fp.write('train=%s\n' % join('.', train_txt))
                obj_data_fp.write('valid=%s\n' % join('.', valid_txt))
                obj_data_fp.write('labels=%s\n' % join('.', obj_names))
                obj_data_fp.write('names=%s\n' % join('.', obj_names))
                obj_data_fp.write('backup=%s/weights/\n' % '.')
                obj_data_fp.write('top=3\n')

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
            config.read(join(abs_group_path, obj_cfg))
            print(config.sections())
            config._sections['net1']['batch'] = 64
            config._sections['net1']['subdivisions'] = subdivisions
            config._sections['net1']['max_batches'] = max_batches
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

            obj_cfg_fp = open(join(abs_group_path, obj_cfg), 'w')
            for key in config._sections:
                section = config._sections[key]
                obj_cfg_fp.write('[%s]\n' % section['original_key'])
                del section['original_key']
                for field in section:
                    obj_cfg_fp.write('%s=%s\n' % (field, section[field]))
                obj_cfg_fp.write('\n') 
    return training_folders