set msbuild_exe="C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
rem set msbuild_exe="C:\Program Files (x86)\MSBuild\14.0\Bin\amd64\MSBuild.exe"
rem =========================
rem Build
rem =========================

%msbuild_exe% darknet_vs\darknet.sln /p:Configuration=Release /p:Platform=x64 /v:quiet /m
rem %msbuild_exe% darknet_vs\darknet.sln /p:Configuration=Release-CPU /p:Platform=x64 /v:quiet /m

%msbuild_exe% darknet_vs\yolo_cpp_dll.sln /p:Configuration=Release /p:Platform=x64 /v:quiet /m
%msbuild_exe% darknet_vs\yolo_cpp_dll.sln /p:Configuration=Debug /p:Platform=x64 /v:quiet /m
%msbuild_exe% darknet_vs\yolo_cpp_dll.sln /p:Configuration=Release-CPU /p:Platform=x64 /v:quiet /m

rem %msbuild_exe% modules\Yolo_mark\yolo_mark.sln /p:Configuration=Release /p:Platform=x64 /v:quiet /m
rem %msbuild_exe% modules\yolo2_light\yolo_gpu.sln /p:Configuration=Release /p:Platform=x64 /v:quiet /m

rem =========================
rem Deploy to bin/
rem =========================

rem robocopy darknet_vs\bin\ bin\ *.exe
rem robocopy darknet_vs\bin\ bin\ *.lib

robocopy modules\darknet\3rdparty\pthreads\bin\ bin\ pthreadVC2.dll

robocopy modules\darknet\cfg bin\cfg /E
robocopy modules\darknet\data bin\data /E
robocopy modules\Yolo_mark\x64\Release\ bin\ yolo_mark.exe
rem robocopy modules\yolo2_light\bin\ bin\ yolo_gpu.exe
robocopy C:\opencv\build\x64\vc14\bin\ bin\ *.dll

rem =========================
rem Deploy to scripts/
rem =========================

rem robocopy darknet_vs\bin\ bin\ *.dll
robocopy bin\ scripts\ *.dll
robocopy modules\darknet\ scripts\ darknet.py