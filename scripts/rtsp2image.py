#!/usr/bin/env python

'''
1. 安装 Python 3.6 https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
2. ctrl + r -> cmd 进命令行，输入 pip install opencv-python
3. 运行范例代码，从默认摄像头读取画面，存到 img 文件夹内
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
    else:
        print('created capture')
    return cap

if __name__ == '__main__':
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Dump image from camera device')
    parser.add_argument(
        '--source', type=int, default=0, help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--w', type=int, default=640)
    parser.add_argument('--h', type=int, default=480)
    parser.add_argument('--shotdir')
    args = parser.parse_args()

    args.source = 'rtsp://admin:admin@192.168.1.200:8554/live'

    interactive_mode = args.shotdir is None
    needs_new_name = True

    if interactive_mode:
        print("Press SPACE in Capture window to input new folder name")

    cap = create_capture(args)
    prev_datetime = datetime.datetime.now()
    while True:
        ret, img = cap.read()
        if not ret: 
            cap = create_capture(args)
            continue
        cv.imshow('Capture', img)
        ch = cv.waitKey(1)
        if ch == 32:
            # SPACE
            needs_new_name = True
        if ch == 27:
            # ESC
            break
        now = datetime.datetime.now()
        if (now - prev_datetime).microseconds > 200 * 1000:
            if interactive_mode and needs_new_name:
                args.shotdir = input("Input folder name: ")
                needs_new_name = False
            if not os.path.exists(args.shotdir):
                os.mkdir(args.shotdir)

            now_string = now.strftime("%Y-%h-%d-%H-%M-%S-%f")
            prev_datetime = now
            fn = '%s/cam_%s.jpg' % (args.shotdir, now_string)
            cv.imwrite(fn, img)
            print(fn, 'saved')
    cv.destroyAllWindows()
