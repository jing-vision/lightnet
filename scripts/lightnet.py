# pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import sys
import os
import darknet

cwd = os.getcwd()

predict_image_v2 = darknet.lib.network_predict_image_v2
predict_image_v2.argtypes = [c_void_p, darknet.IMAGE]
predict_image_v2.restype = POINTER(c_float)


def set_cwd(path):
    global cwd
    cwd = path


def classify(net, meta, im):
    out = darknet.predict_image(net, im)
    res = []
    for i in range(meta.classes):
        nameTag = meta.names[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    print("v1")
    print(res[0])

    out = predict_image_v2(net, im)
    res = []
    for i in range(meta.classes):
        nameTag = meta.names[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    print("v2")
    print(res[0])
    return res


def to_str(path, feed_to_darknet=False):
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    if feed_to_darknet:
        path = path.encode('ascii')
    return path


def load_name_list(names_path):
    if os.path.exists(names_path):
        with open(names_path) as namesFH:
            namesList = namesFH.read().strip().split("\n")
            altNames = [x.strip() for x in namesList]
            return altNames
    return []


USING_DARKNET_IMAGE_IO = True


def detect_from_file(net, meta, image_path, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    #pylint: disable= C0321
    if USING_DARKNET_IMAGE_IO:
        im = darknet.load_image(to_str(image_path, True), 0, 0)
    else:
        import cv2
        custom_image = cv2.imread(to_str(image_path))
        im, arr = darknet.array_to_image(custom_image)
    if debug:
        print("Loaded image")
    det = detect_from_memory(net, meta, im, thresh, hier_thresh, nms, debug)
    if USING_DARKNET_IMAGE_IO:
        darknet.free_image(im)
        if debug:
            print("freed image")

    return det


def convertBack(x, y, w, h):
    xmin = round(x - (w / 2))
    if xmin < 0:
        xmin = 0
    xmax = round(x + (w / 2))
    if xmax > w - 1:
        xmax = round(w - 1)
    ymin = round(y - (h / 2))
    if ymin < 0:
        ymin = 0
    ymax = round(y + (h / 2))
    if ymax > h - 1:
        ymax = round(h - 1)
    return xmin, ymin, xmax, ymax


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
                nameTag = meta.names[i]
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


def load_network_meta(cfg_path, weights_path, meta_path=None):
    class PyMeta:

        def __init__(self):
            self.classes = 0
            self.names = []
    py_meta = PyMeta()

    if meta_path:
        names = load_name_list(meta_path)
        py_meta.classes = len(names)
        py_meta.names = names
    else:
        py_meta.classes = 1
        py_meta.names = ['obj']

    net = darknet.load_net_custom(
        to_str(cfg_path, True), to_str(weights_path, True), 0, 1)  # batch size = 1

    return net, py_meta


if __name__ == "__main__":
    net, meta = load_network_meta(
        "obj.cfg", "weights/obj_last.weights", "obj.names")

    r = detect_from_file(net, meta,
                         "img/air_vapor/air_vapor_original_shot_2018-May-07-22-12-34-237301_3fd9a907-034f-45c0-9a84-f4431945fa7b.jpg", thresh=0.25, debug=False)
    print(r)
