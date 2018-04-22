set CWD=%~dp0
cd ..\..\bin
call darknet.exe detector demo cfg\combine9k.data cfg\yolo9000.cfg %CWD%\yolo9000.weights
cd %CWD%