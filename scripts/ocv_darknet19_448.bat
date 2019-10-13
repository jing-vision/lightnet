python ../scripts/ocvdnn/classification.py ^
    --config ../bin/cfg/darknet19_448_ocv.cfg ^
    --model ../bin/darknet19_448.weights ^
    --scale 0.007843 ^
    --mean 0.5 0.5 0.5 ^
    --classes ../bin/cfg/coco.names ^
    --rgb ^
    --input ../bin/data/giraffe.jpg ^
    --width 448 ^
    --height 448