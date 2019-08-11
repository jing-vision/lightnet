REM set CUDA_VISIBLE_DEVICES=-1
set PATH=%PATH%;%~dp0\..\bin
python ..\scripts\detector.py --camera=0 --weights=weights\obj_last.weights