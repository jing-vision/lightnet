#pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import sys
import os
import darknet

cwd = os.getcwd()

def set_cwd(path):
    global cwd
    cwd = path

def to_str(path, feed_to_darknet = False):
    path =  os.path.join(cwd, path)
    if feed_to_darknet:
        path = path.encode('ascii')
    return path

def load_name_list(metaPath):
    # In Python 3, the metafile default access craps out on Windows (but not Linux)
    # Read the names file and create a list to feed to detect
    try:
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            import re
            match = re.search("names *= *(.*)$", metaContents,
                                re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
                        return altNames
            except TypeError:
                pass
    except Exception:
        pass
    return []


def detect_from_file(net, meta, image_path, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    #pylint: disable= C0321
    if False:
        im = darknet.load_image(to_str(image_path, True), 0, 0)
    else:
        import cv2
        custom_image = cv2.imread(to_str(image_path))
        im, arr = darknet.array_to_image(custom_image)
    if debug:
        print("Loaded image")
    det = detect_from_memory(net, meta, im, thresh, hier_thresh, nms, debug)
    if False:
        darknet.free_image(im)
        if debug:
            print("freed image")

    return det

def detect_from_memory(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the meat of the detection
    """

    num = c_int(0)
    if debug:
        print("Assigned num")
    pnum = pointer(num)
    if debug:
        print("Assigned pnum")
    darknet.predict_image(net, im)
    if debug:
        print("did prediction")
    dets = darknet.get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum, 1)
    if debug:
        print("Got dets")
    num = pnum[0]
    if debug:
        print("got zeroth index of pnum")
    if nms:
        darknet.do_nms_sort(dets, num, meta.classes, nms)
    if debug:
        print("did sort")
    res = []
    if debug:
        print("about to range")
    for j in range(num):
        if debug:
            print("Ranging on " + str(j) + " of " + str(num))
        if debug:
            print("Classes: " + str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug:
                print("Class-ranging on " + str(i) + " of " +
                      str(meta.classes) + "= " + str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if darknet.altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = darknet.altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug:
        print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug:
        print("did sort")
    darknet.free_detections(dets, num)
    if debug:
        print("freed detections")
    return res

def load_network_meta(cfg_path, weights_path, meta_path):
    net = darknet.load_net_custom(
        to_str(cfg_path, True), to_str(weights_path, True), 0, 1)  # batch size = 1
    meta = darknet.load_meta(to_str(meta_path, True))
    darknet.altNames = load_name_list(to_str(meta_path))

    return net, meta

if __name__ == "__main__":
    net, meta = load_network_meta(
        "yolo-obj.cfg", "weights/yolo-obj_200.weights", "obj.data")

    r = detect_from_file(net, meta, 
        "img/air_vapor/air_vapor_original_shot_2018-May-07-22-12-34-237301_3fd9a907-034f-45c0-9a84-f4431945fa7b.jpg", thresh=0.25, debug=False)
    print(r)
