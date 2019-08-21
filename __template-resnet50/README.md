network layout
----

```
   layer   filters  size/strd(dil)      input                output
   0 conv     64      7 x 7/ 2    256 x 256 x   3  ->  128 x 128 x  64 0.308 BF
   1 max              2 x 2/ 2    128 x 128 x  64 ->   64 x  64 x  64 0.001 BF
   2 conv     64      1 x 1/ 1     64 x  64 x  64  ->   64 x  64 x  64 0.034 BF
   3 conv     64      3 x 3/ 1     64 x  64 x  64  ->   64 x  64 x  64 0.302 BF
   4 conv    256      1 x 1/ 1     64 x  64 x  64  ->   64 x  64 x 256 0.134 BF
   5 Shortcut Layer: 1
   6 conv     64      1 x 1/ 1     64 x  64 x 256  ->   64 x  64 x  64 0.134 BF
   7 conv     64      3 x 3/ 1     64 x  64 x  64  ->   64 x  64 x  64 0.302 BF
   8 conv    256      1 x 1/ 1     64 x  64 x  64  ->   64 x  64 x 256 0.134 BF
   9 Shortcut Layer: 5
  10 conv     64      1 x 1/ 1     64 x  64 x 256  ->   64 x  64 x  64 0.134 BF
  11 conv     64      3 x 3/ 1     64 x  64 x  64  ->   64 x  64 x  64 0.302 BF
  12 conv    256      1 x 1/ 1     64 x  64 x  64  ->   64 x  64 x 256 0.134 BF
  13 Shortcut Layer: 9
  14 conv    128      1 x 1/ 1     64 x  64 x 256  ->   64 x  64 x 128 0.268 BF
  15 conv    128      3 x 3/ 2     64 x  64 x 128  ->   32 x  32 x 128 0.302 BF
  16 conv    512      1 x 1/ 1     32 x  32 x 128  ->   32 x  32 x 512 0.134 BF
  17 Shortcut Layer: 13
 w = 32, w2 = 64, h = 32, h2 = 64, c = 512, c2 = 256
  18 conv    128      1 x 1/ 1     32 x  32 x 512  ->   32 x  32 x 128 0.134 BF
  19 conv    128      3 x 3/ 1     32 x  32 x 128  ->   32 x  32 x 128 0.302 BF
  20 conv    512      1 x 1/ 1     32 x  32 x 128  ->   32 x  32 x 512 0.134 BF
  21 Shortcut Layer: 17
  22 conv    128      1 x 1/ 1     32 x  32 x 512  ->   32 x  32 x 128 0.134 BF
  23 conv    128      3 x 3/ 1     32 x  32 x 128  ->   32 x  32 x 128 0.302 BF
  24 conv    512      1 x 1/ 1     32 x  32 x 128  ->   32 x  32 x 512 0.134 BF
  25 Shortcut Layer: 21
  26 conv    128      1 x 1/ 1     32 x  32 x 512  ->   32 x  32 x 128 0.134 BF
  27 conv    128      3 x 3/ 1     32 x  32 x 128  ->   32 x  32 x 128 0.302 BF
  28 conv    512      1 x 1/ 1     32 x  32 x 128  ->   32 x  32 x 512 0.134 BF
  29 Shortcut Layer: 25
  30 conv    256      1 x 1/ 1     32 x  32 x 512  ->   32 x  32 x 256 0.268 BF
  31 conv    256      3 x 3/ 2     32 x  32 x 256  ->   16 x  16 x 256 0.302 BF
  32 conv   1024      1 x 1/ 1     16 x  16 x 256  ->   16 x  16 x1024 0.134 BF
  33 Shortcut Layer: 29
 w = 16, w2 = 32, h = 16, h2 = 32, c = 1024, c2 = 512
  34 conv    256      1 x 1/ 1     16 x  16 x1024  ->   16 x  16 x 256 0.134 BF
  35 conv    256      3 x 3/ 1     16 x  16 x 256  ->   16 x  16 x 256 0.302 BF
  36 conv   1024      1 x 1/ 1     16 x  16 x 256  ->   16 x  16 x1024 0.134 BF
  37 Shortcut Layer: 33
  38 conv    256      1 x 1/ 1     16 x  16 x1024  ->   16 x  16 x 256 0.134 BF
  39 conv    256      3 x 3/ 1     16 x  16 x 256  ->   16 x  16 x 256 0.302 BF
  40 conv   1024      1 x 1/ 1     16 x  16 x 256  ->   16 x  16 x1024 0.134 BF
  41 Shortcut Layer: 37
  42 conv    256      1 x 1/ 1     16 x  16 x1024  ->   16 x  16 x 256 0.134 BF
  43 conv    256      3 x 3/ 1     16 x  16 x 256  ->   16 x  16 x 256 0.302 BF
  44 conv   1024      1 x 1/ 1     16 x  16 x 256  ->   16 x  16 x1024 0.134 BF
  45 Shortcut Layer: 41
  46 conv    256      1 x 1/ 1     16 x  16 x1024  ->   16 x  16 x 256 0.134 BF
  47 conv    256      3 x 3/ 1     16 x  16 x 256  ->   16 x  16 x 256 0.302 BF
  48 conv   1024      1 x 1/ 1     16 x  16 x 256  ->   16 x  16 x1024 0.134 BF
  49 Shortcut Layer: 45
  50 conv    256      1 x 1/ 1     16 x  16 x1024  ->   16 x  16 x 256 0.134 BF
  51 conv    256      3 x 3/ 1     16 x  16 x 256  ->   16 x  16 x 256 0.302 BF
  52 conv   1024      1 x 1/ 1     16 x  16 x 256  ->   16 x  16 x1024 0.134 BF
  53 Shortcut Layer: 49
  54 conv    512      1 x 1/ 1     16 x  16 x1024  ->   16 x  16 x 512 0.268 BF
  55 conv    512      3 x 3/ 2     16 x  16 x 512  ->    8 x   8 x 512 0.302 BF
  56 conv   2048      1 x 1/ 1      8 x   8 x 512  ->    8 x   8 x2048 0.134 BF
  57 Shortcut Layer: 53
 w = 8, w2 = 16, h = 8, h2 = 16, c = 2048, c2 = 1024
  58 conv    512      1 x 1/ 1      8 x   8 x2048  ->    8 x   8 x 512 0.134 BF
  59 conv    512      3 x 3/ 1      8 x   8 x 512  ->    8 x   8 x 512 0.302 BF
  60 conv   2048      1 x 1/ 1      8 x   8 x 512  ->    8 x   8 x2048 0.134 BF
  61 Shortcut Layer: 57
  62 conv    512      1 x 1/ 1      8 x   8 x2048  ->    8 x   8 x 512 0.134 BF
  63 conv    512      3 x 3/ 1      8 x   8 x 512  ->    8 x   8 x 512 0.302 BF
  64 conv   2048      1 x 1/ 1      8 x   8 x 512  ->    8 x   8 x2048 0.134 BF
  65 Shortcut Layer: 61
  66 conv      2      1 x 1/ 1      8 x   8 x2048  ->    8 x   8 x   2 0.001 BF
  67 avg                            8 x   8 x   2  ->      2
  68 softmax                                           2
  69 cost                                              2
Total BFLOPS 9.739
```

> mklink /D img /path/to/real/img

