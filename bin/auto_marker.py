#!/usr/bin/env python

'''
This program illustrates the use of findContours and drawContours.
The original image is put up along with the image of drawn contours.

Usage:
    contours.py
A trackbar is put up which controls the contour level from -3 to 3
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
g_image_filenames = []
train_txt = 'train.txt'
obj_names_txt = 'obj.names'

# runtime config
contours = None
hierarchy = None
img = None
gray = None

cv.namedWindow('marker')

def update_category(category_id):
    global g_image_filenames
    g_image_filenames = glob.glob(category_folders[category_id] + '/*.jpg')
    cv.createTrackbar("image", 'marker', 0,
                      len(g_image_filenames) - 1, update_image)
    cv.setTrackbarMax('image', 'marker', len(g_image_filenames) - 1)
    update_image(0)

def update_threshold(*arg):
    update_image(cv.getTrackbarPos('image', 'marker'))

cv.createTrackbar("category", 'marker', 0, len(
    category_folders) - 1, update_category)

CANNY_MODE = False
if CANNY_MODE:
    cv.createTrackbar('thrs1', 'marker', 2000, 5000, update_threshold)
    cv.createTrackbar('thrs2', 'marker', 4000, 5000, update_threshold)


def update_image(image_id, category_id = 0, image_filenames=[], enable_vis=True, enable_marker_dump=False):
    try:
        global contours, hierarchy, img, gray, g_image_filenames
        if len(image_filenames) > 0:
            g_image_filenames=image_filenames
        img=cv.imread(g_image_filenames[image_id])

        cv.setTrackbarPos('image', 'marker', image_id)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if enable_vis:
            cv.imshow('gray', gray)

        if CANNY_MODE:
            thrs1 = cv.getTrackbarPos('thrs1', 'marker')
            thrs2 = cv.getTrackbarPos('thrs2', 'marker')
            bin = cv.Canny(gray, thrs1, thrs2, apertureSize=5)
        else:
            bin = cv.adaptiveThreshold(
                gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 10)

        if enable_vis:
            cv.imshow('bin', bin)

        _, contours0, hierarchy = cv.findContours(
            bin.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours0 if cv.contourArea(cnt) > 200]

        if enable_vis:
            cv.imshow('image', img)
        update_contour(category_id, image_id, enable_vis, enable_marker_dump)
    except Exception:
        import traceback
        traceback.print_exc()
        raise


def update_contour(category_id, image_id, enable_vis, enable_marker_dump):
    h, w = img.shape[:2]
    vis = np.zeros((h, w, 3), np.uint8)
    bboxes = [cv.boundingRect(cnt) for cnt in contours]
    cv.drawContours(vis, contours, -1, 255, -1)
    # boxes = [[bbox.x, bbox[1], bbox.x + bbox.w, bbox.y+bbox.h] for bbox in bboxes]
    # cv.drawContours(vis, bboxes, -1, (0, 0, 255), 2)
    left = w
    right = 0
    top = h
    bottom = 0
    for bbox in bboxes:
        new_left = bbox[0]
        new_right = bbox[0] + bbox[2]
        new_top = bbox[1]
        new_bottom = bbox[1] + bbox[3]
        if new_left < left:
            left = new_left
        if new_right > right:
            right = new_right
        if new_top < top:
            top = new_top
        if new_bottom > bottom:
            bottom = new_bottom
        if enable_vis:
            cv.rectangle(vis, (new_left, new_top),
                     (new_right, new_bottom), (0, 255, 255), 1)
    if enable_vis:
        cv.rectangle(vis, (left, top), (right, bottom), (0, 0, 255), 3)

    if enable_vis:
        cv.imshow('marker', vis)

    if enable_marker_dump:
        txt_filename = g_image_filenames[image_id].replace('.jpg', '.txt')
        with open(txt_filename, 'w') as fp:
            fp.write("%d %f %f %f %f\n" % (category_id, (left + right) / 2 / w,
                                           (top + bottom) / 2 / h, (right - left) / w, (bottom - top) / h))
    
def main():
    print(__doc__)

    update_category(0)

    cv.waitKey()

    train_txt_fp = open(train_txt, 'w')

    from multiprocessing import Pool
    from itertools import repeat
    from functools import partial

    with Pool(4) as pool:
        for category_id in xrange(len(category_folders)):
            category = category_folders[category_id]
            image_filenames = glob.glob(category + '/*.jpg')

            for image_filename in image_filenames:
                train_txt_fp.write(os.path.abspath(image_filename))
                train_txt_fp.write('\n')

            pool.map(partial(update_image, category_id=category_id,image_filenames=image_filenames, enable_vis=False,
                            enable_marker_dump=True), xrange(len(image_filenames)))

if __name__ == '__main__':
    main()
