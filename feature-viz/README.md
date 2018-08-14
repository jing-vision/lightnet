How to build
====

- Execute `build.bat` in the root folder, make sure there is no error.
- Download `premake` from https://premake.github.io/download.html
- Enter `feature-viz` folder, execute `premake5 vs2015`
- Open `feature-viz\vs2015\feature-viz.sln` and build
- Read `feature-viz\README.md` for usages

How to use
====
```
feature-viz.exe -cfg=cfg\alexnet.cfg -weights=alexnet.weights
feature-viz.exe -cfg=cfg\darknet.cfg -weights=darknet.weights
feature-viz.exe -cfg=cfg\darknet19.cfg -weights=darknet19.weights
feature-viz.exe -cfg=cfg\darknet19_448.cfg -weights=darknet19_448.weights
feature-viz.exe -cfg=cfg\extraction.cfg -weights=extraction.weights
feature-viz.exe -cfg=cfg\vgg-conv.cfg -weights=vgg-conv.weights
feature-viz.exe -cfg=cfg\vgg-16.cfg -weights=vgg-16.weights
feature-viz.exe -cfg=..\jing-pose\openpose.cfg -weights=openpose.weight
feature-viz.exe -cfg=cfg\yolo9000.cfg -weights=yolo9000.weights -names=cfg\9k.names
feature-viz.exe -cfg=cfg\yolov2.cfg -weights=yolov2.weights
feature-viz.exe -cfg=cfg\yolov3.cfg -weights=yolov3.weights -names=cfg\coco.names
feature-viz.exe -cfg=..\..\openpose\models\pose\body_25\body_25.cfg -weights=..\..\openpose\models\pose\body_25\body_25.weights
feature-viz.exe -cfg=..\==nike_darknet19_448\obj.cfg -weights=..\==nike_darknet19_448\weights\obj_300.weights ..\==nike_darknet19_448\test.jpg
```


![](https://github.com/jing-vision/lightnet/raw/master/feature-viz/doc/yolo9000-viz.jpg)

