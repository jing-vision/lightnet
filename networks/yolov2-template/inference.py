import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

import darknet

net = darknet.load_net(b"yolo-obj.cfg", b"weights/yolo-obj_200.weights", 0)
meta = darknet.load_meta(b"obj.data")
r = darknet.detect(net, meta, b"img/female_005.jpg", 0.25)
print(r)
