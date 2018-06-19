REM =========================
REM Build
REM =========================

msbuild darknet_vs\darknet.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m
REM msbuild darknet_vs\darknet.sln /p:Configuration=Release-CPU /p:Platform=x64 /v:minimal /m

msbuild darknet_vs\yolo_cpp_dll.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m
REM msbuild darknet_vs\yolo_cpp_dll.sln /p:Configuration=Debug /p:Platform=x64 /v:minimal /m
REM msbuild darknet_vs\yolo_cpp_dll.sln /p:Configuration=Release-CPU /p:Platform=x64 /v:minimal /m

msbuild modules\Yolo_mark\yolo_mark.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m
REM msbuild modules\yolo2_light\yolo_gpu.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m

REM =========================
REM Deploy to bin/
REM =========================

robocopy darknet_vs\bin\ bin\ *.exe
robocopy darknet_vs\bin\ bin\ *.lib

robocopy modules\darknet\build\darknet\x64\ bin\ pthreadVC2.dll

robocopy modules\darknet\cfg bin\cfg /E
robocopy modules\darknet\data bin\data /E
robocopy modules\Yolo_mark\x64\Release\ bin\ yolo_mark.exe
REM robocopy modules\yolo2_light\bin\ bin\ yolo_gpu.exe
robocopy D:\opencv\build\x64\vc14\bin\ bin\ *.dll

REM =========================
REM Deploy to scripts/
REM =========================
robocopy darknet_vs\bin\ scripts\ *.dll
robocopy modules\darknet\ scripts\ darknet.py