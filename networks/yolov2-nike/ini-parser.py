import configparser
from collections import OrderedDict

num_classes = 10

class multidict(OrderedDict):
    _unique = 0   # class variable

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            self._unique += 1
            key += str(self._unique)
        OrderedDict.__setitem__(self, key, val)

file = 'D:/__svn_pool/yolo-studio/networks/yolov2-nike/yolo-obj.cfg'
config = configparser.ConfigParser(
    defaults=None, dict_type=multidict, strict=False)
config.read(file)
print(config.sections())
config._sections['convolutional31']['filters'] = (num_classes + 5) * 5
config._sections['region32']['classes'] = num_classes

with open(file + '.cfg', 'w') as configfile:
    config.write(configfile)
