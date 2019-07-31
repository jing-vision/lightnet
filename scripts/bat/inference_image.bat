set PATH=%PATH%;%~dp0\..\bin
..\bin\darknet.exe classifier predict obj.data obj.cfg weights\obj_last.weights %*