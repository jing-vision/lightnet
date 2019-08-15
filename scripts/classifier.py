'''
pip install flask gevent requests pillow

https://github.com/jrosebr1/simple-keras-rest-api

https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594

'''

''' Usage
python ..\scripts\classifier.py --socket=5000 --weights=weights\obj_last.weights
curl -X POST -F image=@dog.png http://localhost:5000/predict
'''


import threading
import time
import csv
import datetime
import flask
import sys
import os
import cv2 as cv
import argparse
import lightnet
import darknet
import socket
import logging
logger = logging.getLogger(__name__)

app = flask.Flask(__name__)

args = None
nets = []
metas = []
yolo_net = None
yolo_meta = None
csv_file = None
csv_writer = None

gpu_lock = threading.Lock()

host_ip = 'localhost'


def get_Host_name_IP():
    try:
        global host_ip
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("baidu.com", 80))
        host_ip, _ = s.getsockname()
        print("http://%s:5000" % host_ip)
    except:
        print("Unable to get Hostname and IP")


@app.route("/", methods=["GET"])
def index_get():
    data = vars(args)
    data['usage'] = "curl -X POST -F image=@dog.png http://%s:5000/predict" % (
        host_ip)
    return flask.jsonify(data)


@app.route("/predict", methods=["POST"])
def predict_post():
    import numpy as np

    # initialize the data dictionary that will be returned from the
    # view

    logger.info("/predict start")

    data = []

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method != "POST":
        return '[]'
    image = flask.request.files.get("image")
    if not image:
        return '[]'

    try:
        # read the image in PIL format
        image = flask.request.files["image"].read()
        logger.info("|flask.request")
        # convert string of image data to uint8
        nparr = np.fromstring(image, np.uint8)

        # decode image
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

        logger.info("|cv.imdecode")
        results = slave_labor(frame)
        logger.info(results)
    except:
        logger.error('|exception', exc_info=True)
        return "[]"

    # loop over the results and add them to the list of
    # returned predictions
    for (label, prob) in results:
        r = {"label": label, "score": float(prob)}
        data.append(r)
    logger.info("\predict end")
    # return the data dictionary as a JSON response
    return flask.jsonify(data)


def cvDrawBoxes(detections, img):
    roi_array = []
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = lightnet.convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        roi_array.append((xmin, ymin, xmax, ymax))

        if args.debug:
            cv.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv.putText(img,
                       # detection[0] +
                       " [" + str(round(detection[1] * 100, 2)) + "]",
                       (pt1[0], pt1[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       [0, 255, 0], 2)
    return roi_array


def slave_labor(frame):
    h, w, _ = frame.shape
    roi_array = []
    full_im, _ = darknet.array_to_image(frame)
    darknet.rgbgr_image(full_im)

    gpu_lock.acquire()
    if args.yolo:
        #
        r = lightnet.detect_from_memory(
            yolo_net, yolo_meta, full_im, thresh=0.75, debug=False)
        logger.debug(r)
        roi_array = cvDrawBoxes(r, frame)

        if args.debug:
            logger.info("|yolo")

    if not roi_array:
        roi_array = [(0, 0, w, h)]

    results_hier = []
    results_flat = []

    frame_rois = []
    im_rois = []

    for i, _ in enumerate(nets):
        results = []
        for roi in roi_array:
            if args.yolo:
                # print(roi)
                frame_roi = frame[roi[1]: roi[3], roi[0]:roi[2]]
                frame_rois.append(frame_roi)
                if not args.socket and not args.interactive:
                    cv.imshow("frame_roi", frame_roi)
            else:
                frame_roi = frame
            im, _ = darknet.array_to_image(frame_roi)
            darknet.rgbgr_image(im)
            r = darknet.classify(nets[i], metas[i], im)

            results.extend(r)
            results_flat.extend(r)
            # results = sorted(results, key=lambda x: -x[1])
        results_hier.append(results)
    logger.info("|darknet.classify")
    gpu_lock.release()

    results_flat = sorted(results_flat, key=lambda x: -x[1])
    top_k = args.top_k
    if top_k >= len(results_flat):
        top_k = len(results_flat)

    preds = []
    for rank in range(0, top_k):
        left = 10
        top = 20 + rank * 20
        (label, score) = results_flat[rank]
        if score >= args.threshold:
            preds.append((label[4:], score))

        text = '%s %.2f%%' % (label, score * 100)
        labelSize, baseLine = cv.getTextSize(
            text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        back_clr = (222, 222, 222)
        if score > args.gold_confidence:
            back_clr = (122, 122, 255)
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[
            0], top + baseLine), back_clr, cv.FILLED)

        cv.putText(frame, text, (left, top),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    if args.socket:
        if args.debug:
            now = datetime.datetime.now()
            now_string = now.strftime("%Y-%h-%d-%H-%M-%S-%f")
            image_name = 'socket_debug' + '/' + now_string + '.jpg'
            cv.imwrite(image_name, frame)
            csv_file.write(image_name)
            for results in results_hier:
                top_k = 3
                for rank in range(0, top_k):
                    (label, score) = results[rank]
                    csv_file.write(',%s,%.3f' % (label[4:], score))
            csv_file.write('\n')
            csv_file.flush()

            logger.info("|csv_file.write")

    elif args.interactive:
        pass
    else:
        cv.imshow("output", frame)

    return preds


def interactive_run():
    while True:
        filename = input("Input image path:")
        if not filename:
            continue
        frame = cv.imread(filename)
        results = slave_labor(frame)
        for r in results:
            logger.debug("%s: %.3f" % (r[0], r[1]))
        # key = cv.waitKey(1)


def local_app_run():
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        slave_labor(frame)
        key = cv.waitKey(1)
        if key == 27:
            break


def main():
    # lightnet.set_cwd(dir)
    global nets, metas, args, yolo_net, yolo_meta

    def add_bool_arg(parser, name, default=False):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no-' + name, dest=name, action='store_false')
        parser.set_defaults(**{name: default})

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='obj.cfg')
    parser.add_argument('--weights', default='weights/obj_last.weights')
    parser.add_argument('--names', default='obj.names')
    parser.add_argument('--image')
    parser.add_argument('--video')
    parser.add_argument('--socket', type=int)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gold_confidence', type=float, default=0.95)
    parser.add_argument('--threshold', type=float, default=0.5)
    add_bool_arg(parser, 'debug')
    add_bool_arg(parser, 'yolo')
    add_bool_arg(parser, 'interactive')
    parser.add_argument('--yolo_cfg', default='yolo/obj.cfg')
    parser.add_argument('--yolo_weights', default='yolo/obj_last.weights')

    args = parser.parse_args()
    args_cfgs = args.cfg.split(',')
    args_weights = args.weights.split(',')
    args_names = args.names.split(',')
    for i, _ in enumerate(args_cfgs):
        net, meta = lightnet.load_network_meta(
            args_cfgs[i], args_weights[i], args_names[i])
        nets.append(net)
        metas.append(meta)

    if args.yolo:
        yolo_net, yolo_meta = lightnet.load_network_meta(
            args.yolo_cfg, args.yolo_weights)

    logging.basicConfig(level=logging.INFO)

    if args.debug:
        folder_name = 'socket_debug'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        # create a file handler
        handler = logging.FileHandler('socket_debug/debug.log')
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(thread)d - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        global csv_file
        now = datetime.datetime.now()
        now_string = now.strftime("%Y-%h-%d-%H-%M-%S-%f")
        csv_name = 'socket_debug' + '/' + now_string + '.csv'
        csv_file = open(csv_name, 'w')
        csv_file.write('image')
        for i, _ in enumerate(args_cfgs):
            csv_file.write(
                ',%d_top1,score,%d_top2,score,%d_top3,score' % (i + 1, i + 1, i + 1))
        csv_file.flush()
        csv_file.write('\n')

    if args.socket:

        # flask routine
        print('=========================================')
        get_Host_name_IP()
        print('=========================================')
        app.run(host='0.0.0.0', port=args.socket, threaded=True)
        exit(0)

    if args.interactive:
        interactive_run()
    elif args.video or args.image:
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

    local_app_run()


if __name__ == "__main__":
    main()
