set CWD=%~dp0
cd ..\..\bin
call darknet.exe detector demo cfg\coco.data cfg\yolov3.cfg %CWD%\yolov3.weights
cd %CWD%