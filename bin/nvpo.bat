set mydate=%date:/=%
set mytime=%time::=%
set TIMESTAMP=%mydate: =_%_%mytime:.=_%
call nvprof -o nvprof-%1-%TIMESTAMP%.prof %*