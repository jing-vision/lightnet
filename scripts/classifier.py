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

def slave_labor(frame):
    im, arr = darknet.array_to_image(frame)
    darknet.rgbgr_image(im)
    results = darknet.classify(net, meta, im)

    top_k = args.top_k
    if top_k >= len(results):
        top_k = len(results)
    # print(r[0])

    preds = []
    if results[0][1] > args.display_confidence:
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

    cv.imshow("output", frame)
    cv.imwrite("socket_output.jpg", frame)

    return preds

def local_app_run():
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        slave_labor(frame)
        if cv.waitKey(1) != -1:
            break

if __name__ == "__main__":
    # lightnet.set_cwd(dir)

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
    parser.add_argument('--display_confidence', type=float, default=0.5)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    net, meta = lightnet.load_network_meta(
        args.cfg, args.weights, args.data)

    if args.socket:
        # flask routine
        print('=========================================')
        get_Host_name_IP()
        print('=========================================')
        app.run(host='0.0.0.0', port=args.socket, debug=args.debug)
        exit(0)

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

    local_app_run()
