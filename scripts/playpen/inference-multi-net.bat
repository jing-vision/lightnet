REM set CUDA_VISIBLE_DEVICES=-1
set PATH=%PATH%;%~dp0\..\..\bin
python ..\..\scripts\classifier.py ^
    --socket=5000 ^
    --cfg=G0\obj.cfg,G1\obj.cfg,G2\obj.cfg,G3\obj.cfg ^
    --weights=G0\weights\obj_last.weights,G1\weights\obj_last.weights,G2\weights\obj_last.weights,G3\weights\obj_last.weights ^
    --names=G0\obj.names,G1\obj.names,G2\obj.names,G3\obj.names ^
    --group=G0,G1,G2,G3 ^
    --top_k=3 ^
    --threshold=0.1 %*