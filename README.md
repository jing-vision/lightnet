lightnet
====
lightnet is a turnkey solution to real world problems accelerated with deep learning AI technology, including but not limited to object detection, image classification and human pose estimation.

* **[The source code](#how-to-read-the-source-code)**
* **[Build instructions](#how-to-build-from-visual-studio-2015)**
* **[End2end Object Detection](#object-detection---inference-w-pre-trained-weights)**
* **[End2end Image Classification](#image-classification---inference-w-pre-trained-weights)**
* **[Human Pose Estimation](#human-pose-estimation---inference-w-pre-trained-weights)**
* **[FAQ](#faq)**


How to read the source code
====

This project is dependent on a few open-source projects:
- modules/darknet - the main engine for training & inferencing.
- modules/Yolo_mark - the toolkit to prepare training data for object detection.
- modules/yolo2_light - lightweighted inferencing engine [optional].
- modules/cvui - lightweighted GUI based purely on OpenCV.
- moudles/pytorch-caffe-darknet-convert - DL framework model converter
- modules/minitrace - library to generate tracing logs for Chrome "about:tracing"
- modules/readerwriterqueue - single-producer, single-consumer lock-free queue for C++
- modules/bhtsne - Barnes-Hut implementation of the t-SNE algorithm

How to build from Visual Studio 2015
====

Install NVIDIA SDK
----

- Download [CUDA 10.0](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
- Download [cuDNN v7.x] (https://developer.nvidia.com/rdp/cudnn-download)

    -  Extract to the same folder as CUDA SDK
    -  e.g. `c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\`

Install OpenCV
----
- OpenCV 3.4.0: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.0/opencv-3.4.0-vc14_vc15.exe/download
- Extract to `d:\opencv\`

Build it
----

Execute the batch file
> build.bat


Object Detection - inference w/ pre-trained weights
====

First you need to download the weights. You can read more details on [darknet website](https://pjreddie.com/darknet/yolo/).

cfg|weights
---|-------
cfg/yolov2.cfg|https://pjreddie.com/media/files/yolov2.weights
cfg/yolov2-tiny.cfg|https://pjreddie.com/media/files/yolov2-tiny.weights
cfg/yolo9000.cfg|http://pjreddie.com/media/files/yolo9000.weights
cfg/yolov3.cfg|https://pjreddie.com/media/files/yolov3.weights
cfg/yolov3-tiny.cfg|https://pjreddie.com/media/files/yolov3-tiny.weights

Syntax for object detection
----
```
darknet.exe detector demo <data> <cfg> <weights> -c <camera_idx> -i <gpu_idx>
darknet.exe detector demo <data> <cfg> <weights> <video_filename> -i <gpu_idx>
darknet.exe detector test <data> <cfg> <weights> <img_filename> -i <gpu_idx>
```

Default launch device combination is `-i 0 -c 0`.

Run yolov3
----
```
darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights
```

Run yolo9000 on camera #0
----
```
darknet.exe detector demo cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights
```

Run yolo9000 CPU on camera #0
----
```
darknet_no_gpu.exe detector demo cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights
```

Object Detection - label images manually
====

 - delete all files from directory `my-yolo-net/img` and put your `.jpg`-images in
 - change numer of classes (objects for detection) in file `my-yolo-net/obj.data`: https://github.com/jing-vision/yolo-studio/blob/master/networks/yolo-template/obj.data#L1
 - put names of objects, one for each line in file `my-yolo-net/obj.names`: https://github.com/jing-vision/yolo-studio/blob/master/networks/yolo-template/obj.names
 - Run file: `my-yolo-net/yolo_mark.cmd`

Object Detection - train yolo v2 network
====

0. Fork `__template-yolov2` to `my-yolo-net`

1. Download pre-trained weights for the convolutional layers: http://pjreddie.com/media/files/darknet19_448.conv.23 to `bin/darknet19_448.conv.23`

2. To training for your custom objects, you should change 2 lines in file `obj.cfg`:

 - change `classes` in obj.data#L1
 - set number of classes (objects) in obj.cfg#L230
 - set `filter`-value equal to `(classes + 5)*5` in obj.cfg#L224

3. Run `my-yolo-net/train.cmd`

Object Detection - train yolo v3 network
====

0. Fork `__template-yolov3` to `my-yolo-net`

1. Download pre-trained weights for the convolutional layers: http://pjreddie.com/media/files/darknet53.conv.74 to `bin/darknet53.conv.74`

2. Create file `obj.cfg` with the same content as in `yolov3.cfg` (or copy `yolov3.cfg` to `obj.cfg)` and:

  * change line batch to [`batch=64`](obj.cfg#L3)
  * change line subdivisions to [`subdivisions=8`](obj.cfg#L4)
  * change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers:
      * obj.cfg#L610
      * obj.cfg#L696
      * obj.cfg#L783
  * change [`filters=255`] to filters=(classes + 5)x3 in the 3 `[convolutional]` before each `[yolo]` layer
      * obj.cfg#L603
      * obj.cfg#L689
      * obj.cfg#L776

  So if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`.
  
  **(Do not write in the cfg-file: filters=(classes + 5)x3)**
  
  (Generally `filters` depends on the `classes`, `coords` and number of `mask`s, i.e. filters=`(classes + coords + 1)*<number of mask>`, where `mask` is indices of anchors. If `mask` is absence, then filters=`(classes + coords + 1)*num`)

  So for example, for 2 objects, your file `obj.cfg` should differ from `yolov3.cfg` in such lines in each of **3** [yolo]-layers:

  ```
  [convolutional]
  filters=21

  [region]
  classes=2
  ```

Image Classification - inference w/ pre-trained weights
====

Again, you need download weights first. You can read more details on [darknet website](https://pjreddie.com/darknet/imagenet/).

cfg|weights
---|-------
cfg/alexnet.cfg|https://pjreddie.com/media/files/alexnet.weights
cfg/vgg-16.cfg|https://pjreddie.com/media/files/vgg-16.weights
cfg/extraction.cfg|https://pjreddie.com/media/files/extraction.weights
cfg/darknet.cfg|https://pjreddie.com/media/files/darknet.weights
cfg/darknet19.cfg|https://pjreddie.com/media/files/darknet19.weights
cfg/darknet19_448.cfg|https://pjreddie.com/media/files/darknet19_448.weights
cfg/resnet50.cfg|https://pjreddie.com/media/files/resnet50.weights
cfg/resnet152.cfg|https://pjreddie.com/media/files/resnet152.weights
cfg/densenet201.cfg|https://pjreddie.com/media/files/densenet201.weights

Image Classification - train darknet19_448 network
====

0. Fork `__template-darknet19_448` to `my-darknet19-net`

1. Download pre-trained weights for the convolutional layers: http://pjreddie.com/media/files/darknet19_448.conv.23 to `bin/darknet19_448.conv.23`

2. Create file `obj.cfg` with the same content as in `darknet19_448.cfg` (or copy `darknet19_448.cfg` to `obj.cfg)` and:

  * set `batch` to `128` or `64` or `32` depends on your GPU memory in darknet19-classify.cfg#L4
  * change line to [`subdivisions=4`](darknet19-classify.cfg#L5)
  * set `filter`-value equal to `classes` in darknet19-classify.cfg#L189


Human Pose Estimation - inference w/ pre-trained weights
====

This project lives in [DancingGaga](https://github.com/jing-interactive/DancingGaga)

For more details, please check the README there.

<b>[Weight file] (darknet version openpose.weight)</b><p>
https://drive.google.com/open?id=1BfY0Hx2d2nm3I4JFh0W1cK2aHD1FSGea
  

FAQ
====

How to fix `CUDA Error: no kernel image is available for execution on the device`?

```
# Tesla V100
# ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]

# GeForce RTX 2080 Ti, RTX 2080, RTX 2070, Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000, Tesla T4, XNOR Tensor Cores
# ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]

# Jetson XAVIER
# ARCH= -gencode arch=compute_72,code=[sm_72,compute_72]

# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

# GP100/Tesla P100 - DGX-1
# ARCH= -gencode arch=compute_60,code=sm_60

# For Jetson TX1, Tegra X1, DRIVE CX, DRIVE PX - uncomment:
# ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

# For Jetson Tx2 or Drive-PX2 uncomment:
# ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]
```

How to fine tune a existing network?
----

https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection

> darknet.exe partial cfg/darknet.cfg darknet.weights darknet.conv.13 13
> darknet.exe partial cfg/extraction.cfg extraction.weights extraction.conv.23 23
> darknet.exe partial cfg/darknet19.cfg darknet19.weights darknet19.conv.23 23
> darknet.exe partial cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.23 23

Explanation of yolo training output
----

https://github.com/rafaelpadilla/darknet#faq_yolo


CFG Parameters
----
- [in net section](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section)
- [in layer section](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers)