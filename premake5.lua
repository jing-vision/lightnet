-- http://industriousone.com/scripting-reference

local action = _ACTION or ""

local CUDA_PATH = os.getenv("CUDA_PATH");
local OPENCV_PATH   = "d:/opencv/build"

solution "yolo-studio"
    location (action)
    configurations { "Debug", "Release" }
    platforms {"x64"}
    language "C"
    kind "StaticLib"

    filter "system:windows"
        defines { 
            "_CRT_SECURE_NO_WARNINGS",
            "WIN32",
            "_TIMESPEC_DEFINED",
            "OPENCV",
            "CV_IGNORE_DEBUG_BUILD_GUARD",
        }

    configuration "x86"
        libdirs {
            path.join(action, "x86")
        }
        targetdir (path.join(action, "x86"))

    configuration "x64"
        libdirs {
            path.join(action, "x64"),
        }
        targetdir (path.join(action, "x64"))

    filter "system:macosx"
        defines {
            "_MACOSX",
        }

    configuration "Debug"
        defines { "DEBUG" }
        symbols "On"
        targetsuffix "-d"

    configuration "Release"
        defines { "NDEBUG" }
        flags { "No64BitChecks" }
        editandcontinue "Off"
        optimize "Speed"
        optimize "On"
        editandcontinue "Off"

    project "darknet"
        buildcustomizations{ "BuildCustomizations/CUDA 9.1" }
        includedirs {
            "darknet/3rdparty/include",
            "darknet/src",
            path.join(OPENCV_PATH, "include")
        }
        files { 
            "darknet/src/*h",
            "darknet/src/*c",
            "darknet/src/*cu",
        }
        removefiles {
            -- "darknet/src/darknet.c",
            "darknet/src/classifier.c",
            "darknet/src/go.c",
            "darknet/src/darknet_demo.c",
            "darknet/src/rnn.c",
            "darknet/src/coco.c",
            "darknet/src/yolo.c",
            "darknet/src/captcha.c",
            "darknet/src/compare.c",
            "darknet/src/nightmare.c",
            "darknet/src/cifar.c",
            "darknet/src/demo.c",
            "darknet/src/rnn_vid.c",
            "darknet/src/voxel.c",
            "darknet/src/tag.c",
            "darknet/src/writing.c",
            "darknet/src/super.c",
            "darknet/src/dice.c",
            "darknet/src/swag.c",
            "darknet/src/art.c",
            "darknet/src/classifier.h",
        }

