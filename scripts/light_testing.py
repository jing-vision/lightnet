'''
pip install flask gevent requests pillow

https://github.com/jrosebr1/simple-keras-rest-api

https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594

'''

''' Usage
python ..\scripts\classifier.py --socket=5001 --weights=weights\obj_last.weights
curl -X POST -F image=@dog.png http://localhost:5001/training/begin?plan=testplan
'''

import threading
import time
import csv
import datetime
import flask
import sys
import os

os.environ["FORCE_CPU"] = '1'

import cv2 as cv
import argparse
import lightnet
import darknet
import socket
import requests
import get_ar_plan
import logging
logger = logging.getLogger(__name__)
app = flask.Flask(__name__)
from os.path import join
import traceback

gpu_lock = threading.Lock()

host_ip = 'localhost'

#
server_state_idle = 0
server_state_testing_loaded = 2

server_state = None

server_testing_internal = {}

def get_Host_name_IP():
    try:
        global host_ip
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("baidu.com", 80))
        host_ip, _ = s.getsockname()
        print("http://%s:5001" % host_ip)
    except:
        print("Unable to get Hostname and IP")

@app.route("/", methods=["GET"])
def index_get():
    data = vars(args)
    data['usage'] = "curl -X POST -F image=@dog.png http://%s:5001/predict" % (
        host_ip)
    return flask.jsonify(data)

@app.route("/testing/load", methods=["GET"])
def testing_load():
    global server_testing_internal
    plan = flask.request.args.get("plan")
    print(plan)
 
    server_testing_internal = {
        'plans': [],
        'groups': [],
        'nets': [],
        'metas': []
    }

    try:
        url = 'http://localhost:8800/api/Training/plan?plan=%s' % plan
        response = requests.get(url)
        plan_json = response.json()
        training_metas = get_ar_plan.get_training_metas(plan_json)

        for training_meta in training_metas:
            folder = training_meta['folder']
            obj_cfg = join(folder, 'obj.cfg')
            obj_weights = join(folder, 'weights/obj_final.weights')
            obj_names = join(folder, 'obj.names')
            if not os.path.exists(obj_cfg):
                raise Exception('%s missing' % obj_cfg)
            if not os.path.exists(obj_weights):
                raise Exception('%s missing' % obj_weights)
            if not os.path.exists(obj_names):
                raise Exception('%s missing' % obj_names)

            net, meta = lightnet.load_network_meta(obj_cfg, obj_weights, obj_names)
            server_testing_internal['plans'].append(training_meta['plan'])
            server_testing_internal['groups'].append(training_meta['group'])
            server_testing_internal['nets'].append(net)
            server_testing_internal['metas'].append(meta)
        result = {
                'errCode': 'OK', # or 'Error'
                'errMsg': ''
            }
    except:
        error_callstack = traceback.format_exc()
        print(error_callstack)
        result = {
            'errCode': 'Error', # or 'Error'
            'errMsg': error_callstack
        }
    
    return flask.jsonify(result)

@app.route("/predict", methods=["POST"])
def predict_post():
    import numpy as np

    # initialize the data dictionary that will be returned from the
    # view

    logger.info("/predict start")

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

    logger.info("\predict end")
    # return the data dictionary as a JSON response
    return flask.jsonify(results)

def slave_labor(frame):
    h, w, _ = frame.shape
    roi_array = []
    full_im, _ = darknet.array_to_image(frame)
    darknet.rgbgr_image(full_im)

    gpu_lock.acquire()
    if args.yolo:
        if w < h:
            spacing = int((h - w) / 2)
            roi_array = [(0, spacing, w, h - spacing)]
        else:
            spacing = int((w - h) / 2)
            roi_array = [(spacing, 0, w - spacing, h)]

    if not roi_array:
        roi_array = [(0, 0, w, h)]

    # TODO: remove frame_rois
    frame_rois = []

    nets = server_testing_internal['nets']
    plans = server_testing_internal['plans']
    groups = server_testing_internal['groups']
    metas = server_testing_internal['metas']

    results = []

    for i, _ in enumerate(nets):
        roi = roi_array[0]
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
        r = lightnet.classify(nets[i], metas[i], im)

        top_k = args.top_k
        if top_k >= len(r):
            top_k = len(r)

        for rank in range(0, top_k):
            (label, score) = r[rank]
            results.append({
                'plan': plans[i], 
                'group': groups[i], 
                'predicate_sku': label,
                'score': score,
            })
    logger.info("|lightnet.classify")
    gpu_lock.release()

    return results

def main():
    # lightnet.set_cwd(dir)
    global args, server_state
    server_state = server_state_idle

    def add_bool_arg(parser, name, default=False):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no-' + name, dest=name, action='store_false')
        parser.set_defaults(**{name: default})

    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', type=int, default=5001)
    parser.add_argument('--top_k', type=int, default=3)
    add_bool_arg(parser, 'yolo')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # flask routine
    print('=========================================')
    get_Host_name_IP()
    print('=========================================')
    app.run(host='0.0.0.0', port=args.socket, threaded=True)

if __name__ == "__main__":
    main()
