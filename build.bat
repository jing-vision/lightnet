REM Build

msbuild darknet_vs\darknet.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m
REM msbuild darknet_vs\darknet.sln /p:Configuration=Release-CPU /p:Platform=x64 /v:minimal /m

msbuild darknet_vs\yolo_cpp_dll.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m
msbuild darknet_vs\yolo_cpp_dll.sln /p:Configuration=Debug /p:Platform=x64 /v:minimal /m
REM msbuild darknet_vs\yolo_cpp_dll.sln /p:Configuration=Release-CPU /p:Platform=x64 /v:minimal /m

msbuild modules\Yolo_mark\yolo_mark.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m
msbuild modules\yolo2_light\yolo_gpu.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m

REM Deploy to bin/

robocopy darknet_vs\bin\ bin\ *.exe
robocopy darknet_vs\bin\ bin\ *.dll
robocopy darknet_vs\bin\ bin\ *.lib

robocopy darknet\build\darknet\x64\ bin\ pthreadVC2.dll
robocopy darknet\ bin\ darknet.py
robocopy darknet\cfg bin\cfg /E
robocopy darknet\data bin\data /E
robocopy modules\Yolo_mark\x64\Release\ bin\ yolo_mark.exe
robocopy modules\yolo2_light\bin\ bin\ yolo_gpu.exe
robocopy D:\opencv\build\x64\vc14\bin\ bin\ *.dll