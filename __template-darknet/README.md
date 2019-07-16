network layout
----

```
   layer   filters  size/strd(dil)      input                output
   0 conv     16      3 x 3/ 1    224 x 224 x   3  ->  224 x 224 x  16 0.043 BF
   1 max              2 x 2/ 2    224 x 224 x  16  ->  112 x 112 x  16 0.001 BF
   2 conv     32      3 x 3/ 1    112 x 112 x  16  ->  112 x 112 x  32 0.116 BF
   3 max              2 x 2/ 2    112 x 112 x  32  ->   56 x  56 x  32 0.000 BF
   4 conv     64      3 x 3/ 1     56 x  56 x  32  ->   56 x  56 x  64 0.116 BF
   5 max              2 x 2/ 2     56 x  56 x  64  ->   28 x  28 x  64 0.000 BF
   6 conv    128      3 x 3/ 1     28 x  28 x  64  ->   28 x  28 x 128 0.116 BF
   7 max              2 x 2/ 2     28 x  28 x 128  ->   14 x  14 x 128 0.000 BF
   8 conv    256      3 x 3/ 1     14 x  14 x 128  ->   14 x  14 x 256 0.116 BF
   9 max              2 x 2/ 2     14 x  14 x 256  ->    7 x   7 x 256 0.000 BF
  10 conv    512      3 x 3/ 1      7 x   7 x 256  ->    7 x   7 x 512 0.116 BF
  11 max              2 x 2/ 2      7 x   7 x 512  ->    4 x   4 x 512 0.000 BF
  12 conv   1024      3 x 3/ 1      4 x   4 x 512  ->    4 x   4 x1024 0.151 BF
  13 conv   1000      1 x 1/ 1      4 x   4 x1024  ->    4 x   4 x1000 0.033 BF
  14 avg                            4 x   4 x1000  ->   1000
  15 softmax                                            1000
  16 cost                                               1000
Total BFLOPS 0.807
```

> mklink /D img /path/to/real/img
