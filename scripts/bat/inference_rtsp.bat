REM set CUDA_VISIBLE_DEVICES=-1
set PATH=%PATH%;%~dp0\..\bin
REM set DARKNET_FORCE_CPU=
python ..\scripts\classifier.py --video=rtsp://admin:admin@10.23.146.226:8554/live --weights=weights\obj_last.weights