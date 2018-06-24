#!/usr/bin/env python

'''
1. 安装 Python 3.6 https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
2. ctrl + r -> cmd 进命令行，输入 pip install opencv-python
3. 运行目录下的 run.bat

run.bat 中的内容解释
python video2image.py --source 0 --w 1024 --h 768 --shotdir img

--source 0 摄像头设备的编号，从零开始
--w 1024 画面宽度，不需要修改
--h 768 画面高度，不需要修改
--shotdir img 输出的文件夹名字，建议一类物体配一个文件夹

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
    parser.add_argument('--shotdir', default='img')
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
