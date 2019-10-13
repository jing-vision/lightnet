'''
pip install flask gevent requests pillow

https://github.com/jrosebr1/simple-keras-rest-api

https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594

'''

''' Usage
python ..\scripts\classifier.py --socket=5000 --weights=weights\obj_last.weights
curl -X POST -F image=@dog.png http://localhost:5000/training/begin?plan=testplan
'''


import threading
import time
import csv
import datetime
import flask
import traceback
import sys
import os
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

args = None
nets = []
metas = []
args_groups = []
csv_file = None
csv_writer = None
cap = None

gpu_lock = threading.Lock()

host_ip = 'localhost'

#
server_state_idle = 0
server_state_training = 1

server_state = None

server_training_status = {
    'plan_name': '',
    'percentage': 0,
}
server_training_status_internal = {
    'folders': [],
}

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

@app.route("/training/status", methods=["GET"])
def training_status():
    return flask.jsonify(server_training_status)

def training_thread_function(training_folders):
    global server_state, server_training_status, server_training_status_internal
    server_training_status_internal['folders'] = training_folders

    import subprocess
    idx = 1 # start from 1
    for folder in training_folders:
        bat_file = join(folder, 'train.bat')
        logging.info("%s: starting", bat_file)
        p = subprocess.Popen(bat_file, shell=True, stdout = subprocess.PIPE)
        stdout, stderr = p.communicate()
        print(p.returncode) # is 0 if success    
        logging.info("%s: finishing", bat_file)
        server_training_status['percentage'] = idx * 100 / len(training_folders)
        idx += 1
    server_state = server_state_idle
    server_training_status['plan_name'] = ''
    server_training_status['percentage'] = 0
    server_training_status_internal['folders'] = []

@app.route("/training/begin", methods=["GET"])
def training_begin():
    global server_state, server_training_status
    if server_state != server_state_idle:
        result = {
            'errCode': 'Busy', # 'OK/Busy/Error'
            'errMsg': 'Server is busy training %s' % server_training_status['plan_name']
        }
        return flask.jsonify(result)

    try:
        server_state = server_state_training

        plan = flask.request.args.get("plan")
        print(plan)
        server_training_status['plan_name'] = plan
        server_training_status['percentage'] = 0

        url = 'http://localhost:8800/api/Training/plan?plan=%s' % plan
        response = requests.get(url)
        plan_json = response.json()
        # return flask.jsonify(result)
        training_folders = get_ar_plan.prepare_training_folders(plan_json, max_batches=1000)

        x = threading.Thread(target=training_thread_function, args=(training_folders,))
        x.start()

        result = {
            'errCode': 'OK', # 'OK/Busy/Error'
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

def main():
    # lightnet.set_cwd(dir)
    global nets, metas, args, cap, args_groups
    global server_state
    server_state = server_state_idle

    def add_bool_arg(parser, name, default=False):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no-' + name, dest=name, action='store_false')
        parser.set_defaults(**{name: default})

    parser = argparse.ArgumentParser()
    parser.add_argument('--group', default='default')
    parser.add_argument('--cfg', default='obj.cfg')
    parser.add_argument('--weights', default='weights/obj_last.weights')
    parser.add_argument('--names', default='obj.names')
    parser.add_argument('--socket', type=int, default=5000)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gold_confidence', type=float, default=0.95)
    parser.add_argument('--threshold', type=float, default=0.5)
    add_bool_arg(parser, 'debug')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # flask routine
    print('=========================================')
    get_Host_name_IP()
    print('=========================================')
    app.run(host='0.0.0.0', port=args.socket, threaded=True)

if __name__ == "__main__":
    main()
