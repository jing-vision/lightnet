import sys
import os
import cv2 as cv

dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir, '../../bin'))

import lightnet
import darknet

IMAGE_MODE = False

if __name__ == "__main__":
    lightnet.set_cwd(dir)

    net, meta = lightnet.load_network_meta(
        "obj.cfg", "weights/obj_200.weights", "obj.data")

    if IMAGE_MODE:
        if True:
            frame = cv.imread(lightnet.to_str(
                'test.jpg'))
            im, arr = darknet.array_to_image(frame)
            darknet.rgbgr_image(im)
        else:
            im = darknet.load_image(lightnet.to_str(
                'test.jpg').encode("ascii"), 0, 0)

        r = darknet.classify(net, meta, im)
        print(r)
    else:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise Exception('Fail to open %s' % (0))
        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv.waitKey()
                break
            # cv.imwrite('test.jpg', frame)
            cols = frame.shape[1]
            rows = frame.shape[0]

            im, arr = darknet.array_to_image(frame)
            darknet.rgbgr_image(im)
            r = darknet.classify(net, meta, im)
            print(r[0])

            left = 10
            top = 20
            label = '%.2f%% %s' % (r[0][1] * 100, r[0][0])
            labelSize, baseLine = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            back_clr = (222, 222, 222)
            if r[0][1] > 0.95:
                back_clr = (0, 0, 255)
            cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[
                0], top + baseLine), back_clr, cv.FILLED)


            cv.putText(frame, label, (left, top),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            
            cv.imshow("output", frame)
            if cv.waitKey(1) != -1:
                break
