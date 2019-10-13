python ../scripts/ocvdnn/object_detection.py ^
    --config ../bin/cfg/yolov3.cfg ^
    --model ../bin/yolov3.weights ^
    --scale 0.007843 ^
    --classes ../bin/cfg/coco.names ^
    --rgb ^
    --input ../bin/data/dog.jpg ^
    --width 448 ^
    --height 448