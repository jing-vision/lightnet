REM set CUDA_VISIBLE_DEVICES=-1
set PATH=%PATH%;%~dp0\..\bin
REM set DARKNET_FORCE_CPU=
python ..\scripts\classifier.py --weights=weights\obj_last.weights %* --yolo --interactive --threshold 0.01