set mydate=%date:/=%
set mytime=%time::=%
set TIMESTAMP=%mydate: =_%_%mytime:.=_%
call nvprof --print-gpu-trace --demangling off --csv --log-file nvprof-%1-%TIMESTAMP%.csv %*