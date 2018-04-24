# yolo-studio
A turnkey solution to train and deploy your own object detection network, contains:

- Augmentor - image augmentation library in Python.
- Yolo_mark - the toolkit to prepare training data.
- darknet - the main engine for training & inferencing.
- yolo2_light - lightweighted inferencing engine, optional.

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
- Make symbolic link from `c:\opencv_3.0\opencv\` to `d:\opencv\`

## Build from source

Execute the batch file
> build.bat

Or build the componets from Visual Studio
- darknet: `darknet\build\darknet\darknet.sln`, x64|Release -> `darknet\build\darknet\x64\darknet.exe`
- Yolo_mark: `Yolo_mark\yolo_mark.sln`, x64|Release -> `Yolo_mark\x64\Release\yolo_mark.exe`
- yolo2_light: `yolo2_light\yolo_gpu.sln`, Release -> `yolo2_light\bin\yolo_gpu.exe`

# How to train `my-yolo-net`

1. Build `yolo_mark.exe` from `Yolo_mark/yolo_mark.sln`

2. To use for labeling your custom images:

 - Clone `yolo-template` directory to `my-yolo-net`
 - delete all files from directory `my-yolo-net/img` and put your `.jpg`-images in
 - change numer of classes (objects for detection) in file `my-yolo-net/obj.data`: https://github.com/jing-vision/yolo-studio/blob/master/networks/yolo-template/obj.data#L1
 - put names of objects, one for each line in file `my-yolo-net/obj.names`: https://github.com/jing-vision/yolo-studio/blob/master/networks/yolo-template/obj.names
 - Run file: `my-yolo-net/yolo_mark.cmd`

3. To training for your custom objects, you should change 2 lines in file `yolo-obj.cfg`:

 - change `classes` in obj.data#L1
 - set number of classes (objects) in yolo-obj.cfg#L230
 - set `filter`-value equal to `(classes + 5)*5` in yolo-obj.cfg#L224


4. Download pre-trained weights for the convolutional layers (76 MB): http://pjreddie.com/media/files/darknet19_448.conv.23 to `assets/darknet19_448.conv.23`
 
5. Run `my-yolo-net/train.cmd` or `my-yolo-net/train_cpu.cmd`

# How to inference

## Pre-trained models for different cfg-files can be downloaded from (smaller -> faster & lower quality):

cfg|weights
---|-------
cfg/yolov2.cfg|https://pjreddie.com/media/files/yolov2.weights
cfg/yolov2-tiny.cfg|https://pjreddie.com/media/files/yolov2-tiny.weights
cfg/yolo9000.cfg|http://pjreddie.com/media/files/yolo9000.weights
cfg/yolov3.cfg|https://pjreddie.com/media/files/yolov3.weights

## Run Darknet

### General

```
darknet.exe detector demo <data> <cfg> <weights> -c <camera_idx>
darknet.exe detector demo <data> <cfg> <weights> <video_filename>
darknet.exe detector test <data> <cfg> <weights> <img_filename>
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
