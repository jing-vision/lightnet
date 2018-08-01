import sys
import os
import cv2 as cv
import argparse

import lightnet
import darknet

if __name__ == "__main__":
    # lightnet.set_cwd(dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='obj.cfg')
    parser.add_argument('--weights', default='weights/obj_200.weights')
    parser.add_argument('--data', default='obj.data')
    parser.add_argument('--image')
    parser.add_argument('--video')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gold_confidence', type=float, default=0.95)
    parser.add_argument('--display_confidence', type=float, default=0.5)
    args = parser.parse_args()

    net, meta = lightnet.load_network_meta(
        args.cfg, args.weights, args.data)

    if args.video or args.image:
        media = args.image
        if not media:
            media = args.video
        cap = cv.VideoCapture(media)
        if not cap.isOpened():
            raise Exception('Fail to open %s' % media)
    else:
        cap = cv.VideoCapture(args.camera)
        if not cap.isOpened():
            raise Exception('Fail to open camera %d' % args.camera)
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        im, arr = darknet.array_to_image(frame)
        darknet.rgbgr_image(im)
        results = darknet.classify(net, meta, im)

        top_k = args.top_k
        if top_k >= len(results):
            top_k = len(results)
        # print(r[0])

        if results[0][1] > args.display_confidence:
            for rank in range(0, top_k):
                left = 10
                top = 20 + rank * 20
                (label, score) = results[rank]

                label = '%s %.2f%%' % (label[4:], score * 100)
                labelSize, baseLine = cv.getTextSize(
                    label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                back_clr = (222, 222, 222)
                if score > args.gold_confidence:
                    back_clr = (122, 122, 255)
                cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[
                    0], top + baseLine), back_clr, cv.FILLED)

                cv.putText(frame, label, (left, top),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("output", frame)
        if cv.waitKey(1) != -1:
            break
