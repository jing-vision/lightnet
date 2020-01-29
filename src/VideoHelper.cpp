#include "VideoHelper.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

struct ImageFilesCapture : cv::VideoCapture
{
    ImageFilesCapture(const String& filenamesTxt)
    {
        std::ifstream file(filenamesTxt);
        if (!file) return;
        std::string line;
        while (std::getline(file, line))
        {
            imageNames.push_back(line);
        }
    }

    bool isOpened() const override
    {
        return !imageNames.empty();
    }

    VideoCapture& operator >> (Mat& image) override
    {
        if (currentIdx >= imageNames.size() - 1)
        {
            image = {};
            return *this;
        }

        image = imread(imageNames[currentIdx]);
        currentIdx++;
        return *this;
    }

    int currentIdx = 0;
    vector<String> imageNames;
};

std::shared_ptr<VideoCapture> safe_open_video(const CommandLineParser &parser, const String &source, bool *source_is_camera)
{
    char info[100];
    sprintf(info, "open: %s", source.c_str());
    std::shared_ptr<VideoCapture> cap;

    if (source.empty())
        return cap;

    if (source.find(".txt") != String::npos)
    {
        return make_shared<ImageFilesCapture>(source);
    }

    if (source.size() == 1 && isdigit(source[0]))
    {
        cap->open(source[0] - '0');
        if (source_is_camera)
            *source_is_camera = true;
    }
    else
    {
        cap->open(source);
        if (source_is_camera)
            *source_is_camera = false;
    }
    if (!cap->isOpened())
    {
        cout << "Failed to open: " << source << endl;
        return cap;
    }

    if (cap->isOpened())
    {
        if (parser.has("fps"))
        {
            auto fps = parser.get<int>("fps");
            if (fps > 0)
            {
                if (!cap->set(CAP_PROP_FPS, fps))
                    cout << "WARNING: Can't set fps" << endl;
            }
        }

        if (parser.has("video_pos"))
        {
            auto video_pos = parser.get<int>("video_pos");
            if (video_pos > 0)
            {
                if (!cap->set(CAP_PROP_POS_MSEC, video_pos))
                    cout << "WARNING: Can't set video_pos" << endl;
            }
        }

        if (parser.has("width"))
        {
            auto width = parser.get<int>("width");
            if (width > 0)
            {
                if (!cap->set(CAP_PROP_FRAME_WIDTH, width))
                    cout << "WARNING: Can't set width" << endl;
            }
        }

        if (parser.has("height"))
        {
            auto height = parser.get<int>("height");
            if (height > 0)
            {
                if (!cap->set(CAP_PROP_FRAME_HEIGHT, height))
                    cout << "WARNING: Can't set height" << endl;
            }
        }
    }
    return cap;
};


bool safe_grab_video(std::shared_ptr<VideoCapture> cap, const CommandLineParser &parser, Mat& frame, const String& source, bool source_is_camera)
{
    if (!cap->isOpened()) return true;

    char info[100];
    sprintf(info, "open: %s", source.c_str());
    //MTR_SCOPE_FUNC_C("open", source.c_str());

    *cap >> frame; // get a new frame from camera/video or read image
    if (source_is_camera)
    {
        //MTR_SCOPE(__FILE__, "flip");
        flip(frame, frame, 1);
    }

    if (frame.empty())
    {
        if (!parser.has("loop")) return false;
        if (!parser.get<bool>("loop")) return false;

        cap = safe_open_video(parser, source);
        *cap >> frame;
    }

    return true;
};