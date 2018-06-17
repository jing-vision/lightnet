#!/usr/bin/env python

'''
Video capture sample.

Sample shows how VideoCapture class can be used to acquire video
frames from a camera of a movie file. Also the sample provides
an example of procedural video generation by an object, mimicking
the VideoCapture interface (see Chess class).

Keys:
    ESC    - exit
    SPACE  - save current frame to <shot path> directory

'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import datetime

# built-in modules
from time import clock

def create_capture(args):
    cap = cv.VideoCapture(args.source)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.h)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', args.source)
    return cap

if __name__ == '__main__':
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Dump image from camera device')
    parser.add_argument(
        '--source', type=int, default=0, help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--w', type=int, default=1024)
    parser.add_argument('--h', type=int, default=768)
    parser.add_argument('--shotdir', default='')
    parser.add_argument('--prefix', default='shot')
    args = parser.parse_args()

    if not os.path.exists(args.shotdir):
        os.mkdir(args.shotdir)

    cap = create_capture(args)
    prev_datetime = datetime.datetime.now()
    while True:
        ret, img = cap.read()
        cv.imshow('capture', img)
        ch = cv.waitKey(1)
        if ch == 27:
            break
        now = datetime.datetime.now()
        if (now - prev_datetime).microseconds > 200000:
            now_string = now.strftime("%Y-%h-%d-%H-%M-%S-%f")
            prev_datetime = now
            fn = '%s/cam%d_%s_%s.jpg' % (args.shotdir, args.source, args.prefix, now_string)
            cv.imwrite(fn, img)
            print(fn, 'saved')
    cv.destroyAllWindows()
