REM set CUDA_VISIBLE_DEVICES=-1
set PATH=%PATH%;%~dp0\..\bin
REM set DARKNET_FORCE_CPU=
python ..\scripts\roi_extractor.py --images=img --yolo_cfg=yolo\obj.cfg --yolo_weights=yolo\obj_last.weights 