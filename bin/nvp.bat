set TIMESTAMP=%DATE:/=-%-%TIME::=-%
call nvprof --print-gpu-trace --demangling off --csv --log-file nvprof-%1-%TIMESTAMP%.csv %*