import sys
import os

# dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(dir, '../bin'))

import lightnet

if __name__ == "__main__":
    # lightnet.set_cwd(dir)
    net, meta = lightnet.load_network_meta(
        "obj.cfg", "weights/obj_last.weights", "obj.data")

    r = lightnet.detect_from_file(
        net, meta, "kiki/20190805130743.jpg", thresh=0.25, debug=False)
    print(r)
