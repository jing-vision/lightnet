'''
pip install flask gevent requests pillow

https://github.com/jrosebr1/simple-keras-rest-api

https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594

'''

''' Usage
python ..\scripts\classifier.py --socket=5000 --weights=weights\obj_last.weights
curl -X POST -F image=@dog.png http://localhost:5000/predict
'''

import sys
import os
import cv2 as cv
import argparse

import lightnet
import darknet
import socket 
import flask
app = flask.Flask(__name__)

host_ip = 'localhost'
def get_Host_name_IP(): 
    try:
        global host_ip
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print("Hostname:",host_name)
        print("IP:",host_ip)
    except: 
        print("Unable to get Hostname and IP")

@app.route("/", methods=["GET"])
def index_get():
    data = vars(args)
    data['usage'] = "curl -X POST -F image=@dog.png http://%s:5000/predict" % (host_ip)
    return flask.jsonify(data)

@app.route("/predict", methods=["POST"])
def predict_post():
    import numpy as np
    import io

    # initialize the data dictionary that will be returned from the
    # view
    data = []

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            # convert string of image data to uint8
            nparr = np.fromstring(image, np.uint8)
            # decode image
            frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            results = slave_labor(frame)
            print(results)

            # loop over the results and add them to the list of
            # returned predictions
            for (label, prob) in results:
                r = {"label": label, "score": float(prob)}
                data.append(r)

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    if xmin < 0: xmin = 0
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    if ymin < 0: ymin = 0
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    roi_array = []
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
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
    if args.yolo:
        #
        r = lightnet.detect_from_memory(
            yolo_net, yolo_meta, full_im, thresh=0.75, debug=False)
        if args.debug:
            print(r)
        roi_array = cvDrawBoxes(r, frame)

    if not roi_array:
        roi_array = [(0,0, w, h)]

    results = []
    for roi in roi_array:
        frame_roi = frame[roi[1] : roi[3], roi[0]:roi[2]]
        if not args.interactive:
            cv.imshow("frame_roi", frame_roi)
        im, _ = darknet.array_to_image(frame_roi)
        darknet.rgbgr_image(im)
        r = darknet.classify(net, meta, im)
        results.extend(r)

    results = sorted(results, key=lambda x: -x[1])
    top_k = args.top_k
    if top_k >= len(results):
        top_k = len(results)
    # print(r[0])

    preds = []
    if results[0][1] > args.threshold:
        for rank in range(0, top_k):
            left = 10
            top = 20 + rank * 20
            (label, score) = results[rank]
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
            cv.imwrite("socket_output.jpg", frame)
    elif args.interactive:
        pass
    else:
        cv.imshow("output", frame)

    return preds

def interactive_run():
    while True:
        filename = input("Input image path:")
        if not filename: continue
        frame = cv.imread(filename)
        results = slave_labor(frame)
        for r in results:
            print("%s: %.3f" % (r[0], r[1]))
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

if __name__ == "__main__":
    # lightnet.set_cwd(dir)

    def add_bool_arg(parser, name, default=False):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no-' + name, dest=name, action='store_false')
        parser.set_defaults(**{name:default})

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='obj.cfg')
    parser.add_argument('--weights', default='weights/obj_last.weights')
    parser.add_argument('--data', default='obj.data')
    parser.add_argument('--image')
    parser.add_argument('--video')
    parser.add_argument('--socket', type=int)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gold_confidence', type=float, default=0.95)
    parser.add_argument('--threshold', type=float, default=0.001)
    add_bool_arg(parser, 'debug')
    add_bool_arg(parser, 'yolo')
    add_bool_arg(parser, 'interactive')
    parser.add_argument('--yolo_cfg', default='yolo/obj.cfg')
    parser.add_argument('--yolo_weights', default='yolo/obj_last.weights')

    args = parser.parse_args()
    net, meta = lightnet.load_network_meta(args.cfg, args.weights, args.data)

    if args.yolo:
        yolo_net, yolo_meta = lightnet.load_network_meta(args.yolo_cfg, args.yolo_weights)

    if args.socket:
        # flask routine
        print('=========================================')
        get_Host_name_IP()
        print('=========================================')
        app.run(host='0.0.0.0', port=args.socket, debug=args.debug)
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
