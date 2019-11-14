set DATE=%date:/=%
set mydate=%date:/=%
set TIMESTAMP=%mydate: =_%
set OUTPUT=lightnet_%TIMESTAMP%
rmdir /S /Q %OUTPUT%
mkdir %OUTPUT%
mkdir %OUTPUT%/bin

robocopy bin %OUTPUT%/bin opencv_world411.dll
robocopy bin %OUTPUT%/bin pthreadVC2.dll
robocopy bin %OUTPUT%/bin yolo_cpp_dll.dll
robocopy bin %OUTPUT%/bin feature-viz.exe
robocopy bin %OUTPUT%/bin darknet.exe
REM robocopy bin %OUTPUT%/bin darknet.conv.13
REM robocopy bin %OUTPUT%/bin darknet19.conv.23
robocopy bin %OUTPUT%/bin darknet19_448.conv.23
robocopy bin/cfg %OUTPUT%/bin/cfg
robocopy bin/data %OUTPUT%/bin/data

robocopy . %OUTPUT%/ testing-server-5001.bat
robocopy . %OUTPUT%/ training-server-5000.bat

robocopy scripts %OUTPUT%/scripts *.py
robocopy bin %OUTPUT%/scripts yolo_cpp_dll.dll
robocopy bin %OUTPUT%/scripts yolo_cpp_dll_nogpu.dll
robocopy scripts/bat %OUTPUT%/scripts/bat
robocopy scripts/template-darknet19_448 %OUTPUT%/scripts/template-darknet19_448

robocopy __template-darknet19_448 %OUTPUT%/__template-darknet19_448 /MIR
