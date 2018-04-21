# yolo-studio

# How to install

## Install CUDA

* Uninstall Geforce Experience and current driver
* CUDA 9.1: https://developer.nvidia.com/cuda-downloads
* cuDNN v7.x for CUDA https://developer.nvidia.com/rdp/cudnn-download
** Extract to the same folder as CUDA SDK

## Install OpenCV
* OpenCV 3.4.0: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.0/opencv-3.4.0-vc14_vc15.exe/download
* Extract to `d:\opencv\`
* Make symbolic link from `c:\opencv_3.0\opencv\` to `d:\opencv\`

# Run

## Pre-trained models for different cfg-files can be downloaded from (smaller -> faster & lower quality):
* `yolo.cfg` (256 MB COCO-model) - require 4 GB GPU-RAM: http://pjreddie.com/media/files/yolo.weights
* `yolo-voc.cfg` (256 MB VOC-model) - require 4 GB GPU-RAM: http://pjreddie.com/media/files/yolo-voc.weights
* `tiny-yolo.cfg` (60 MB COCO-model) - require 1 GB GPU-RAM: http://pjreddie.com/media/files/tiny-yolo.weights
* `tiny-yolo-voc.cfg` (60 MB VOC-model) - require 1 GB GPU-RAM: http://pjreddie.com/media/files/tiny-yolo-voc.weights
* `yolo9000.cfg` (186 MB Yolo9000-model) - require 4 GB GPU-RAM: http://pjreddie.com/media/files/yolo9000.weights

network|data|cfg|weights
-------|----|---|-------
YOLO on COCO|data/coco.data|cfg/yolo.cfg|weights/yolo.weights
YOLO on VOC|data/voc.data|cfg/yolo-voc.cfg|weights/yolo-voc.weights
Tiny-YOLO on COCO|data/coco.data|cfg/tiny-yolo.cfg|weights/tiny-yolo.weights
Tiny-YOLO on VOC|data/voc.data|cfg/tiny-yolo-voc.cfg|weights/tiny-yolo-voc.weights
YOLO9000 on COCO|data/combine9k.data|cfg/yolo9000.cfg|weights/yolo9000.weights

## Run Darknet

### General

```
darknet.exe detector demo <data> <cfg> <weights> -i gpu_idx -c <camera_idx>
darknet.exe detector test <data> <cfg> <weights> -i gpu_idx <img_filename>
darknet.exe detector demo <data> <cfg> <weights> -i gpu_idx <video_filename>
```

Default launch device combination is `-i 0 -c 0`.

## Run from assets/ folder

### train_voc
```
Download pre-trained weights for the convolutional layers (76 MB): http://pjreddie.com/media/files/darknet19_448.conv.23
..\bin\darknet.exe detector train data/voc.data cfg/yolo-voc.cfg weights/darknet19_448.conv.2
```

### test_voc
```
..\bin\darknet.exe detector test data/voc.data cfg/yolo-voc.cfg weights/yolo-voc.weights -thresh 0.2
```

### yolo9000 on camera #0
```
..\bin\darknet.exe detector demo data/combine9k.data cfg/yolo9000.cfg weights/yolo9000.weights
```

### yolo9000 CPU on camera #0
```
..\bin\darknet-cpu.exe detector demo data/combine9k.data cfg/yolo9000.cfg weights/yolo9000.weights
```

## Train custom objects (my_awesome_net) for Yolo v2

1. Build `yolo_mark.exe` from `Yolo_mark/yolo_mark.sln`

2. To use for labeling your custom images:

 * Clone `example_yolo_net` directory to `my_awesome_net`
 * delete all files from directory `my_awesome_net/img` and put your `.jpg`-images in
 * change numer of classes (objects for detection) in file `my_awesome_net/obj.data`: https://github.com/jing-vision/Cinder-Darknet/blob/master/example_yolo_net/obj.data#L1
 * put names of objects, one for each line in file `my_awesome_net/obj.names`: https://github.com/jing-vision/Cinder-Darknet/blob/master/example_yolo_net/obj.names
 * Run file: `my_awesome_net/yolo_mark.cmd`

3. To training for your custom objects, you should change 2 lines in file `yolo-obj.cfg`:

 * set number of classes (objects): https://github.com/AlexeyAB/Yolo_mark/blob/master/yolo-obj.cfg#L230
 * set `filter`-value equal to `(classes + 5)*5`: https://github.com/AlexeyAB/Yolo_mark/blob/master/yolo-obj.cfg#L224


4. Download pre-trained weights for the convolutional layers (76 MB): http://pjreddie.com/media/files/darknet19_448.conv.23 to `assets/weights/darknet19_448.conv.23`
 
5. Run `my_awesome_net/train.cmd` or `my_awesome_net/train_cpu.cmd`