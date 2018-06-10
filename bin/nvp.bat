set TIMESTAMP=%DATE:/=-%-%TIME::=-%
call nvprof --print-gpu-trace --csv --log-file nvprof-%1-%TIMESTAMP%.csv %*