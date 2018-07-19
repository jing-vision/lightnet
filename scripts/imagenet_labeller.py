import xml.etree.ElementTree as ET
# import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["tennis racket"]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_id):
    in_file = open('annotation/%s' % (image_id))
    out_file = open('labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    objs = root.findall('object')
    num_objs = len(objs)
    print len(objs)
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        print obj
        bndbox = obj.find('bndbox')
        print bndbox
        x1 = float(bndbox[0].text)
        y1 = float(bndbox[1].text)
        x2 = float(bndbox[2].text)
        y2 = float(bndbox[3].text)
        b = (x1, x2, y1, y2)
        bb = convert((w, h), b)
        out_file.write("0" + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

if not os.path.exists('labels'):
    os.makedirs('labels/')
image_ids = open('imageSet.txt').read().strip().split()
list_file = open('train_test.txt', 'w')
for image_id in image_ids:
    list_file.write('%s/images/%s.JPEG\n'%(wd, image_id))
    convert_annotation(image_id)
list_file.close()