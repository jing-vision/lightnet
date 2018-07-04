# facenet-darknet-inference
Face recognition using facenet

**1. Intro**

[Facenet](https://github.com/davidsandberg/facenet) is developed by Google in 2015, the result of the net is the Euclidean embedding of human face. 

By careful defined triplet loss function, facenet achieves high accuracy on LFW(0.9963) and FacesDB(0.9512).

[Darknet](https://github.com/pjreddie/darknet) is a fast, easy to read DL framework. [Yolo](https://pjreddie.com/darknet/yolo/) is running based on it.

**2. Dependencies**

[OpenCV](https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html) for video i/o, face detection, image resizing, warping, and 3D pose estimation.

[Dlib](https://github.com/davisking/dlib) for facial landmark detection.

[NNPACK](https://github.com/digitalbrain79/NNPACK-darknet) for faster neural network computations.

Zenity for text input.

**3. Installation and run**
```
sudo apt-get install zenity
cd facenet-darknet-inference
#edit makefile
#specify your OPENCV_HEADER_DIR, OPENCV_LIBRARY_DIR, DLIB_HEADER_DIR, DLIB_LIBRARY_DIR, NNPACK_HEADER_DIR, NNPACK_LIBRARY_DIR
make
mkdir data
cd data
touch name
cd ..
mkdir model
```

download [weights](https://drive.google.com/open?id=1ATzb5ZEQo424wlSY-cdlT54FUWlIry8V) and extract in facenet-darknet-inference folder

```
cd facenet-darknet-inference
./facenet-darknet-inference
```

**4. Note**

OpenCV VJ face + Dlib landmark detection is used rather than MTCNN. VJ method is faster, but the unstable cropping may slightly influence recognition accuracy.

KNN is the final classification method, but it is suffered for openset problem. The 1792-d feature before bottleneck layer with normalization is used for KNN, because it has better result in openset than original facenet model, but you can still try the original network configure yourself just replacing *facenet.cfg* to *facenet_full.cfg*

The *facenet.weight* is converted from [facenet inception-resnet v1 20180402-114759 model](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view)

**5. Result**

![peek 2018-04-19 14-11](https://user-images.githubusercontent.com/16308037/38980107-89460dd4-43ee-11e8-997d-5ceafd226f43.gif)

