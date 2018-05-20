set CWD=%~dp0
set BIN=%CWD%..\..\bin
set PATH=%PATH%;%BIN%
cd %BIN%
call darknet.py
cd %CWD%