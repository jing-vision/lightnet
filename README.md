# yolo-studio
A turnkey solution to train and deploy your own object detection network, contains:

- modules/darknet - the main engine for training & inferencing.
- modules/Augmentor - image augmentation library in Python.
- modules/Yolo_mark - the toolkit to prepare training data.
- modules/yolo2_light - lightweighted inferencing engine [optional].
- modules/cvui - lightweighted GUI based purely on OpenCV.
- jing-pose - Openpose implementation using darknet framework.

# How to build

## Install CUDA

- Uninstall Geforce Experience and current driver
- CUDA 9.1: https://developer.nvidia.com/cuda-downloads
- cuDNN v7.x for CUDA: https://developer.nvidia.com/rdp/cudnn-download

    -  Extract to the same folder as CUDA SDK
    -  e.g. `c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\`

## Install OpenCV
- OpenCV 3.4.0: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.0/opencv-3.4.0-vc14_vc15.exe/download
- Extract to `d:\opencv\`

## Build from source

Execute the batch file
> build.bat

# Fine tune a existing network

https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection

darknet.exe partial cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.23 23

# Object Detection - yolo
## How to mark labelled images

 - delete all files from directory `my-yolo-net/img` and put your `.jpg`-images in
 - change numer of classes (objects for detection) in file `my-yolo-net/obj.data`: https://github.com/jing-vision/yolo-studio/blob/master/networks/yolo-template/obj.data#L1
 - put names of objects, one for each line in file `my-yolo-net/obj.names`: https://github.com/jing-vision/yolo-studio/blob/master/networks/yolo-template/obj.names
 - Run file: `my-yolo-net/yolo_mark.cmd`

## Train yolo v2

0. Fork `__template-yolov2` to `my-yolo-net`

1. Download pre-trained weights for the convolutional layers: http://pjreddie.com/media/files/darknet19_448.conv.23 to `bin/darknet19_448.conv.23`

2. To training for your custom objects, you should change 2 lines in file `obj.cfg`:

 - change `classes` in obj.data#L1
 - set number of classes (objects) in obj.cfg#L230
 - set `filter`-value equal to `(classes + 5)*5` in obj.cfg#L224

3. Run `my-yolo-net/train.cmd`

## Train yolo v3

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

## How to inference

## Pre-trained models for different cfg-files can be downloaded from (smaller -> faster & lower quality):

cfg|weights
---|-------
cfg/yolov2.cfg|https://pjreddie.com/media/files/yolov2.weights
cfg/yolov2-tiny.cfg|https://pjreddie.com/media/files/yolov2-tiny.weights
cfg/yolo9000.cfg|http://pjreddie.com/media/files/yolo9000.weights
cfg/yolov3.cfg|https://pjreddie.com/media/files/yolov3.weights
cfg/yolov3-tiny.cfg|https://pjreddie.com/media/files/yolov3-tiny.weights

## Run Darknet

### General

```
darknet.exe detector demo <data> <cfg> <weights> -c <camera_idx>
darknet.exe detector demo <data> <cfg> <weights> <video_filename>
darknet.exe detector test <data> <cfg> <weights> <img_filename>
```

Default launch device combination is `-i 0 -c 0`.

## Run from bin/ folder

### yolov3
```
darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights
```

### yolo9000 on camera #0
```
darknet.exe detector demo cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights
```

### yolo9000 CPU on camera #0
```
darknet_no_gpu.exe detector demo cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights
```

# Image Classification
## Download weights

https://pjreddie.com/darknet/imagenet/

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

## Train custom darknet19_448 network

0. Fork `__template-darknet19_448` to `my-darknet19-net`

1. Download pre-trained weights for the convolutional layers: http://pjreddie.com/media/files/darknet19_448.conv.23 to `bin/darknet19_448.conv.23`

2. Create file `obj.cfg` with the same content as in `darknet19_448.cfg` (or copy `darknet19_448.cfg` to `obj.cfg)` and:

  * set `batch` to `128` or `64` or `32` depends on your GPU memory in darknet19-classify.cfg#L4
  * change line to [`subdivisions=4`](darknet19-classify.cfg#L5)
  * set `filter`-value equal to `classes` in darknet19-classify.cfg#L189


# Build cvui

> mkdir vs2015
> cd vs2015
> cmake -DOpenCV_DIR=d:\opencv\build -G "Visual Studio 14 2015 Win64" ..