#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "run_darknet.h"
#include "MiniTraceHelper.h"

using namespace std;
using namespace cv;

const char* params =
"{ help ?       | false             | print usage          }"
"{ proto        | cfg/vgg-conv.cfg  | model configuration }"
"{ model        |vgg-conv.weights   | model weights }"
"{@source       |0                  | source for processing   }"
"{ w width      | 0                 | width of video or camera device}"
"{ h height     | 0                 | height of video or camera device}"
"{ g gui        | true              | show gui, press g to toggle }"
"{ f fullscreen | false             | show in fullscreen, press f to toggle }"
"{ fps          | 0                 | fps of video or camera device }"
"{ single_step  |                   | single step mode, press any key to move to next frame }"
"{ l loop       | true              | whether to loop the video}"
"{ video_pos    | 0                 | current position of the video file in milliseconds. }"
"{ player       | 1                 | current position for player, press p to toggle. }"
;

bool is_gui_visible = false;
bool is_fullscreen = false;

#define APP_NAME "app"
#define VER_MAJOR 0
#define VER_MINOR 1
#define VER_PATCH 0

#define TITLE APP_NAME " " CVAUX_STR(VER_MAJOR) "." CVAUX_STR(VER_MINOR) "." CVAUX_STR(VER_PATCH)

auto safe_open_video = [](const CommandLineParser& parser, const String& source, bool* source_is_camera = nullptr) -> VideoCapture {
    char info[100];
    sprintf(info, "open: %s", source.c_str());
    MTR_SCOPE(__FILE__, info);
    VideoCapture cap;

    if (source.empty()) return cap;

    if (source.size() == 1 && isdigit(source[0]))
    {
        cap.open(source[0] - '0');
        if (source_is_camera) *source_is_camera = true;
    }
    else
    {
        cap.open(source);
        if (source_is_camera) *source_is_camera = false;
    }
    if (!cap.isOpened())
    {
        cout << "Failed to open: " << source << endl;
        return -1;
    }

    if (cap.isOpened())
    {
        auto fps = parser.get<int>("fps");
        if (fps > 0)
        {
            if (!cap.set(CAP_PROP_FPS, fps)) cout << "WARNING: Can't set fps" << endl;
        }

        auto video_pos = parser.get<int>("video_pos");
        if (video_pos > 0)
        {
            if (!cap.set(CAP_PROP_POS_MSEC, video_pos)) cout << "WARNING: Can't set video_pos" << endl;
        }

        auto width = parser.get<int>("width");
        if (width > 0)
        {
            if (!cap.set(CAP_PROP_FRAME_WIDTH, width)) cout << "WARNING: Can't set width" << endl;
        }

        auto height = parser.get<int>("height");
        if (height > 0)
        {
            if (!cap.set(CAP_PROP_FRAME_HEIGHT, height)) cout << "WARNING: Can't set height" << endl;
        }
    }
    return cap;
};


int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, params);
    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    MiniTraceHelper _;

    auto cfg_path = parser.get<string>("proto");
    auto weight_path = parser.get<string>("model");

    Mat frame;

    // 1. read args
    is_gui_visible = parser.get<bool>("gui");
    is_fullscreen = parser.get<bool>("fullscreen");
    Mat upscale_frame;

    int player = parser.get<int>("player");

    String source = parser.get<String>("@source");

    bool source_is_camera = false;

    VideoCapture capture = safe_open_video(parser, source, &source_is_camera);

    bool is_running = true;

    // 2. initialize net
    int net_inw = 0;
    int net_inh = 0;
    int net_outw = 0;
    int net_outh = 0;
    {
        MTR_SCOPE(__FILE__, "init_net");
        init_net(cfg_path.c_str(), weight_path.c_str(), &net_inw, &net_inh, &net_outw, &net_outh);
    }

    float scale = 0.0f;

    vector<float> net_inputs(net_inw * net_inh * 3);

    auto safe_grab_video = [&parser, &is_running](VideoCapture& cap, Mat& frame, const String& source, bool source_is_camera) -> bool {
        if (!cap.isOpened()) return true;

        char info[100];
        sprintf(info, "open: %s", source.c_str());
        MTR_SCOPE_FUNC_C("open", source.c_str());

        cap >> frame; // get a new frame from camera/video or read image
        if (source_is_camera)
        {
            MTR_SCOPE(__FILE__, "flip");
            flip(frame, frame, 1);
        }

        if (frame.empty())
        {
            if (!parser.get<bool>("loop")) return false;

            cap = safe_open_video(parser, source);
        }

        return true;
    };

    int frame_count = 0;

    Mat frames[2];
    Mat netim;
    Mat netim_f32;
    vector<Mat> input_channels;

    while (is_running)
    {
        MTR_SCOPE_FUNC_I("frame", frame_count);

        {
            MTR_SCOPE(__FILE__, "pre process");

            if (!safe_grab_video(capture, frame, source, source_is_camera))
            {
                is_running = false;
            }

            // 4. normalized to float type
            netim.convertTo(netim_f32, CV_32F, 1 / 256.f, -0.5);

            {
                // 5. split channels
                MTR_SCOPE(__FILE__, "split");
                static bool init_input_channels = true;
                if (init_input_channels)
                {
                    init_input_channels = false;
                    float *netin_data_ptr = net_inputs.data();
                    for (int i = 0; i < 3; ++i)
                    {
                        Mat channel(net_inh, net_inw, CV_32FC1, netin_data_ptr);
                        input_channels.emplace_back(channel);
                        netin_data_ptr += (net_inw * net_inh);
                    }
                }
                split(netim_f32, input_channels);
            }
        }

        // 6. feed forward
        float *netoutdata = NULL;
        {
            MTR_SCOPE(__FILE__, "run_net");
            double time_begin = getTickCount();
            netoutdata = run_net(net_inputs.data());
            double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
            cout << "forward fee: " << fee_time << "ms" << endl;
        }

        // 11. show and save result
        {
            MTR_SCOPE(__FILE__, "imshow");
            imshow(TITLE, frame);

            {
                MTR_SCOPE(__FILE__, "waitkey");
                int key = waitKey(1);
                if (key == 27) break;
                if (key == 'f') is_fullscreen = !is_fullscreen;
                if (key == 'g') is_gui_visible = !is_gui_visible;
                if (key == 'p') player = 1 - player;
            }
        }

        frame_count++;
    }

    return 0;
}
