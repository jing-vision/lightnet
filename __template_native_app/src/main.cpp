#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lightnet.h"
#include "MiniTraceHelper.h"
#include "VideoHelper.h"

using namespace std;
using namespace cv;

const char* params =
"{ help ?       | false             | print usage          }"
"{ cfg          | cfg/vgg-conv.cfg  | model configuration }"
"{ weights      |vgg-conv.weights   | model weights }"
"{@source       |0                  | source for processing   }"
"{ w width      | 0                 | width of video or camera device}"
"{ h height     | 0                 | height of video or camera device}"
"{ g gui        | true              | show gui, press g to toggle }"
"{ f fullscreen | false             | show in fullscreen, press f to toggle }"
"{ fps          | 0                 | fps of video or camera device }"
"{ single_step  |                   | single step mode, press any key to move to next frame }"
"{ l loop       | true              | whether to loop the video}"
"{ video_pos    | 0                 | current position of the video file in milliseconds. }"
;

bool is_gui_visible = false;
bool is_fullscreen = false;

#define APP_NAME "app"
#define VER_MAJOR 0
#define VER_MINOR 1
#define VER_PATCH 0

#define TITLE APP_NAME " " CVAUX_STR(VER_MAJOR) "." CVAUX_STR(VER_MINOR) "." CVAUX_STR(VER_PATCH)

#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib") 

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

    auto cfg_path = parser.get<string>("cfg");
    auto weights_path = parser.get<string>("weights");

    Mat frame;

    is_gui_visible = parser.get<bool>("gui");
    is_fullscreen = parser.get<bool>("fullscreen");

    String source = parser.get<String>("@source");

    bool source_is_camera = false;

    VideoCapture capture = safe_open_video(parser, source, &source_is_camera);

    bool is_running = true;

    int net_inw = 0;
    int net_inh = 0;
    int net_outw = 0;
    int net_outh = 0;
    {
        MTR_SCOPE(__FILE__, "init_net");
        init_net(cfg_path.c_str(), weights_path.c_str(), &net_inw, &net_inh, &net_outw, &net_outh);
    }

    int frame_count = 0;
    Mat input_blob;

    while (is_running)
    {
        MTR_SCOPE_FUNC_I("frame", frame_count);

        {
            MTR_SCOPE(__FILE__, "pre process");

            if (!safe_grab_video(capture, parser, frame, source, source_is_camera))
            {
                is_running = false;
            }
            input_blob = dnn::blobFromImage(frame, 1.0f / 255, Size(net_inw, net_inh), Scalar(), false, true);
        }

        float *netoutdata = NULL;
        TickMeter tick;
        {
            MTR_SCOPE(__FILE__, "run_net");
            tick.start();
            netoutdata = run_net(net_inputs);
            tick.stop();
            cout << "forward fee: " << tick.getTimeMilli() << "ms" << endl;
        }

        {
            MTR_SCOPE(__FILE__, "viz");
            {
                MTR_SCOPE(__FILE__, "imshow");
                imshow(TITLE, frame);

                MTR_SCOPE(__FILE__, "waitkey");
                int key = waitKey(1);
                if (key == 27) break;
                if (key == 'f') is_fullscreen = !is_fullscreen;
                if (key == 'g') is_gui_visible = !is_gui_visible;
            }
        }

        frame_count++;
    }

    return 0;
}
