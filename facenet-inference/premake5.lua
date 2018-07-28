-- http://industriousone.com/scripting-reference

local action = _ACTION or ""

local OPENCV_PATH   = "d:/opencv/build"

solution "facenet"
    location (action)
    configurations { "Debug", "Profile", "Release" }
    platforms {"x64"}
    language "C"
    kind "StaticLib"

    filter "system:windows"
        defines { 
            "_CRT_SECURE_NO_WARNINGS",
            "WIN32",
            "_TIMESPEC_DEFINED",
            "OPENCV",
            "GPU",
            -- "CV_IGNORE_DEBUG_BUILD_GUARD",
        }

    configuration "x64"
        libdirs {
            "../bin",
            "../modules/darknet/3rdparty/lib/x64",
            path.join(OPENCV_PATH, "x64/vc14/lib")
        }
        targetdir ("../bin/")

    configuration "Debug"
        defines { "DEBUG" }
        symbols "On"
        targetsuffix "-d"

    configuration "Profile"
        defines { "NDEBUG", "MTR_ENABLED" }
        flags { "No64BitChecks" }
        editandcontinue "Off"
        optimize "Speed"
        optimize "On"
        editandcontinue "Off"

    configuration "Release"
        defines { "NDEBUG" }
        flags { "No64BitChecks" }
        editandcontinue "Off"
        optimize "Speed"
        optimize "On"
        editandcontinue "Off"

    project "facenet"
        kind "ConsoleApp"
        debugdir "../bin"
        includedirs {
            "../modules",
            "../modules/darknet/3rdparty/include",
            "../modules/darknet/src",
            "src",
            "../src",
            path.join("$(CUDA_PATH)", "include"),
            path.join(OPENCV_PATH, "include")
        }
        files { 
            "src/**",
            "../src/**",
            "../modules/minitrace/**",
            "../modules/PDollar/**",
        }
        links {
            "pthreadVC2"
        }
        configuration "Debug"
            links {
                "opencv_world340d.lib",
                "yolo_cpp_dll-d.lib",
            }
        configuration "Profile"
            links {
                "opencv_world340.lib",
                "yolo_cpp_dll.lib",
            }
        configuration "Release"
            links {
                "opencv_world340.lib",
                "yolo_cpp_dll.lib",
            }
