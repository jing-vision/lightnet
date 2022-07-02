-- http://industriousone.com/scripting-reference

local action = _ACTION or ""

local OPENCV_PATH   = "C:/opencv/build"

solution "feature-viz"
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
        cppdialect "C++17"

    configuration "x64"
        libdirs {
            "../bin",
            path.join(OPENCV_PATH, "x64/vc15/lib")
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

    project "feature-viz"
        kind "ConsoleApp"
        includedirs {
            "../modules",
            "../modules/darknet/include",
            "../modules/darknet/3rdparty/pthreads/include",
            "../modules/darknet/3rdparty/stb/include",
            -- "../modules/darknet/src",
            "src",
            "../include",
            path.join("$(CUDA_PATH)", "include"),
            path.join(OPENCV_PATH, "include")
        }
        debugdir "../bin"
        files {
            "../include/**",
            "../src/**",
            "../modules/minitrace/**",
            "src/**",
        }
        configuration "Debug"
            links {
                "yolo_cpp_dll-d.lib",
            }
        configuration "Profile"
            links {
                "yolo_cpp_dll.lib",
            }
        configuration "Release"
            links {
                "yolo_cpp_dll.lib",
            }
