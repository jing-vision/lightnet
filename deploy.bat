set DATE=%date:/=%
set mydate=%date:/=%
set TIMESTAMP=%mydate: =_%
set OUTPUT=lightnet_%TIMESTAMP%
rmdir /S /Q %OUTPUT%
mkdir %OUTPUT%
mkdir %OUTPUT%/bin

robocopy bin %OUTPUT%/bin opencv_world340.dll
robocopy bin %OUTPUT%/bin pthreadVC2.dll
robocopy bin %OUTPUT%/bin yolo_cpp_dll.dll
robocopy bin %OUTPUT%/bin darknet.exe
robocopy bin %OUTPUT%/bin darknet19_448.conv.23
robocopy bin/cfg %OUTPUT%/bin/cfg
robocopy bin/data %OUTPUT%/bin/data

robocopy scripts %OUTPUT%/scripts *.py
robocopy scripts %OUTPUT%/scripts pthreadVC2.dll
robocopy scripts %OUTPUT%/scripts yolo_cpp_dll.dll

robocopy __template-darknet19_448 %OUTPUT%/__template-darknet19_448 /MIR
