network layout
----

```
layer     filters    size              input                output
   0 conv     32  3 x 3 / 1   448 x 448 x   3   ->   448 x 448 x  32 0.347 BF
   1 max          2 x 2 / 2   448 x 448 x  32   ->   224 x 224 x  32 0.006 BF
   2 conv     64  3 x 3 / 1   224 x 224 x  32   ->   224 x 224 x  64 1.850 BF
   3 max          2 x 2 / 2   224 x 224 x  64   ->   112 x 112 x  64 0.003 BF
   4 conv    128  3 x 3 / 1   112 x 112 x  64   ->   112 x 112 x 128 1.850 BF
   5 conv     64  1 x 1 / 1   112 x 112 x 128   ->   112 x 112 x  64 0.206 BF
   6 conv    128  3 x 3 / 1   112 x 112 x  64   ->   112 x 112 x 128 1.850 BF
   7 max          2 x 2 / 2   112 x 112 x 128   ->    56 x  56 x 128 0.002 BF
   8 conv    256  3 x 3 / 1    56 x  56 x 128   ->    56 x  56 x 256 1.850 BF
   9 conv    128  1 x 1 / 1    56 x  56 x 256   ->    56 x  56 x 128 0.206 BF
  10 conv    256  3 x 3 / 1    56 x  56 x 128   ->    56 x  56 x 256 1.850 BF
  11 max          2 x 2 / 2    56 x  56 x 256   ->    28 x  28 x 256 0.001 BF
  12 conv    512  3 x 3 / 1    28 x  28 x 256   ->    28 x  28 x 512 1.850 BF
  13 conv    256  1 x 1 / 1    28 x  28 x 512   ->    28 x  28 x 256 0.206 BF
  14 conv    512  3 x 3 / 1    28 x  28 x 256   ->    28 x  28 x 512 1.850 BF
  15 conv    256  1 x 1 / 1    28 x  28 x 512   ->    28 x  28 x 256 0.206 BF
  16 conv    512  3 x 3 / 1    28 x  28 x 256   ->    28 x  28 x 512 1.850 BF
  17 max          2 x 2 / 2    28 x  28 x 512   ->    14 x  14 x 512 0.000 BF
  18 conv   1024  3 x 3 / 1    14 x  14 x 512   ->    14 x  14 x1024 1.850 BF
  19 conv    512  1 x 1 / 1    14 x  14 x1024   ->    14 x  14 x 512 0.206 BF
  20 conv   1024  3 x 3 / 1    14 x  14 x 512   ->    14 x  14 x1024 1.850 BF
  21 conv    512  1 x 1 / 1    14 x  14 x1024   ->    14 x  14 x 512 0.206 BF
  22 conv   1024  3 x 3 / 1    14 x  14 x 512   ->    14 x  14 x1024 1.850 BF
  23 conv     10  1 x 1 / 1    14 x  14 x1024   ->    14 x  14 x  10 0.004 BF
  24 avg                       14 x  14 x  10   ->    10
  25 softmax                                          10
  26 cost                                             10
Total BFLOPS 21.943
```

> mklink /D img /path/to/real/img

