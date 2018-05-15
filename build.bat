REM Build
msbuild vs2015\darknet.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m
msbuild vs2015\darknet.sln /p:Configuration=Release-CPU /p:Platform=x64 /v:minimal /m
msbuild darknet\build\darknet\yolo_cpp_dll.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m
msbuild Yolo_mark\yolo_mark.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m
msbuild yolo2_light\yolo_gpu.sln /p:Configuration=Release /p:Platform=x64 /v:minimal /m

REM Deploy to bin/
REM robocopy darknet\build\darknet\x64\ bin\ darknet.exe
REM robocopy darknet\build\darknet\x64\ bin\ darknet_no_gpu.exe
robocopy darknet\build\darknet\x64\ bin\ yolo_cpp_dll.dll
robocopy darknet\build\darknet\x64\ bin\ pthreadVC2.dll
robocopy darknet\cfg bin\cfg /E
robocopy darknet\data bin\data /E
robocopy Yolo_mark\x64\Release\ bin\ yolo_mark.exe
robocopy yolo2_light\bin\ bin\ yolo_gpu.exe
robocopy C:\opencv_3.0\opencv\build\x64\vc14\bin\ bin\ opencv_world340.dll
