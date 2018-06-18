import sys
import os
import cv2 as cv
import argparse

import lightnet
import darknet

if __name__ == "__main__":
    # lightnet.set_cwd(dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='obj.cfg')
    parser.add_argument('--weights', default='weights/obj_200.weights')
    parser.add_argument('--data', default='obj.data')
    parser.add_argument('--image')
    parser.add_argument('--video')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--min_confidence', type=float, default=0.95)
    args = parser.parse_args()

    net, meta = lightnet.load_network_meta(
        args.config, args.weights, args.data)

    if args.image is not None:
        if True:
            frame = cv.imread(lightnet.to_str(
                'test.jpg'))
            im, arr = darknet.array_to_image(frame)
            darknet.rgbgr_image(im)
        else:
            im = darknet.load_image(lightnet.to_str(
                'test.jpg', True), 0, 0)

        r = darknet.classify(net, meta, im)
        print(r)
    else:
        cap = cv.VideoCapture(args.camera)
        if not cap.isOpened():
            raise Exception('Fail to open %s' % (0))
        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv.waitKey()
                break

            im, arr = darknet.array_to_image(frame)
            darknet.rgbgr_image(im)
            r = darknet.classify(net, meta, im)
            # print(r[0])

            for rank in range(0, args.top_k):
                left = 10
                top = 20 + rank * 20
                (label, score) = r[rank]

                if score > args.min_confidence:
                    label = '%s %.2f%%' % (label[4:], score * 100)
                else:
                    label = '%s' % (label[4:])
                labelSize, baseLine = cv.getTextSize(
                    label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                back_clr = (222, 222, 222)
                if score > args.min_confidence:
                    back_clr = (122, 122, 255)
                cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[
                    0], top + baseLine), back_clr, cv.FILLED)

                cv.putText(frame, label, (left, top),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            cv.imshow("output", frame)
            if cv.waitKey(1) != -1:
                break
