network layout

https://gist.github.com/cebolan/f4e7c9b1e81d74e8097d9c59c8de7d05

----

```
layer     filters    size              input                output
   0 conv     32  3 x 3 / 1   256 x 256 x   3   ->   256 x 256 x  32 0.113 BF
   1 conv     64  3 x 3 / 2   256 x 256 x  32   ->   128 x 128 x  64 0.604 BF
   2 conv     32  1 x 1 / 1   128 x 128 x  64   ->   128 x 128 x  32 0.067 BF
   3 conv     64  3 x 3 / 1   128 x 128 x  32   ->   128 x 128 x  64 0.604 BF
   4 Shortcut Layer: 1
   5 conv    128  3 x 3 / 2   128 x 128 x  64   ->    64 x  64 x 128 0.604 BF
   6 conv     64  1 x 1 / 1    64 x  64 x 128   ->    64 x  64 x  64 0.067 BF
   7 conv    128  3 x 3 / 1    64 x  64 x  64   ->    64 x  64 x 128 0.604 BF
   8 Shortcut Layer: 5
   9 conv     64  1 x 1 / 1    64 x  64 x 128   ->    64 x  64 x  64 0.067 BF
  10 conv    128  3 x 3 / 1    64 x  64 x  64   ->    64 x  64 x 128 0.604 BF
  11 Shortcut Layer: 8
  12 conv    256  3 x 3 / 2    64 x  64 x 128   ->    32 x  32 x 256 0.604 BF
  13 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128 0.067 BF
  14 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256 0.604 BF
  15 Shortcut Layer: 12
  16 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128 0.067 BF
  17 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256 0.604 BF
  18 Shortcut Layer: 15
  19 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128 0.067 BF
  20 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256 0.604 BF
  21 Shortcut Layer: 18
  22 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128 0.067 BF
  23 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256 0.604 BF
  24 Shortcut Layer: 21
  25 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128 0.067 BF
  26 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256 0.604 BF
  27 Shortcut Layer: 24
  28 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128 0.067 BF
  29 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256 0.604 BF
  30 Shortcut Layer: 27
  31 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128 0.067 BF
  32 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256 0.604 BF
  33 Shortcut Layer: 30
  34 conv    128  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 128 0.067 BF
  35 conv    256  3 x 3 / 1    32 x  32 x 128   ->    32 x  32 x 256 0.604 BF
  36 Shortcut Layer: 33
  37 conv    512  3 x 3 / 2    32 x  32 x 256   ->    16 x  16 x 512 0.604 BF
  38 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256 0.067 BF
  39 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512 0.604 BF
  40 Shortcut Layer: 37
  41 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256 0.067 BF
  42 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512 0.604 BF
  43 Shortcut Layer: 40
  44 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256 0.067 BF
  45 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512 0.604 BF
  46 Shortcut Layer: 43
  47 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256 0.067 BF
  48 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512 0.604 BF
  49 Shortcut Layer: 46
  50 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256 0.067 BF
  51 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512 0.604 BF
  52 Shortcut Layer: 49
  53 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256 0.067 BF
  54 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512 0.604 BF
  55 Shortcut Layer: 52
  56 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256 0.067 BF
  57 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512 0.604 BF
  58 Shortcut Layer: 55
  59 conv    256  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 256 0.067 BF
  60 conv    512  3 x 3 / 1    16 x  16 x 256   ->    16 x  16 x 512 0.604 BF
  61 Shortcut Layer: 58
  62 conv   1024  3 x 3 / 2    16 x  16 x 512   ->     8 x   8 x1024 0.604 BF
  63 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512 0.067 BF
  64 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024 0.604 BF
  65 Shortcut Layer: 62
  66 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512 0.067 BF
  67 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024 0.604 BF
  68 Shortcut Layer: 65
  69 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512 0.067 BF
  70 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024 0.604 BF
  71 Shortcut Layer: 68
  72 conv    512  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x 512 0.067 BF
  73 conv   1024  3 x 3 / 1     8 x   8 x 512   ->     8 x   8 x1024 0.604 BF
  74 Shortcut Layer: 71
  75 avg                        8 x   8 x1024   ->  1024
  76 conv      2  1 x 1 / 1     1 x   1 x1024   ->     1 x   1 x   2 0.000 BF
  77 softmax                                           2
  78 cost                                              2
Total BFLOPS 18.568
```

> mklink /D img /path/to/real/img
