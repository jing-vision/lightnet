#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "run_darknet.h"
#include "MiniTraceHelper.h"
#include "VideoHelper.h"

using namespace std;
using namespace cv;

const char* params =
"{ help ?       | false             | print usage          }"
"{ proto        | cfg/darknet.cfg   | model configuration }"
"{ model        | darknet.weights   | model weights }"
"{ names        | obj.names         | list of the object names }"
"{@source       | 0                 | source for processing   }"
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

#define APP_NAME "feature-viz"
#define VER_MAJOR 0
#define VER_MINOR 1
#define VER_PATCH 0

#define TITLE APP_NAME " " CVAUX_STR(VER_MAJOR) "." CVAUX_STR(VER_MINOR) "." CVAUX_STR(VER_PATCH)

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

    is_gui_visible = parser.get<bool>("gui");
    is_fullscreen = parser.get<bool>("fullscreen");
    Mat upscale_frame;

    int player = parser.get<int>("player");

    String source = parser.get<String>("@source");

    bool source_is_camera = false;

    VideoCapture capture = safe_open_video(parser, source, &source_is_camera);

    bool is_running = true;

    int net_inw = 0;
    int net_inh = 0;
    int net_outw = 0;
    int net_outh = 0;
    int net_output_count = 0;
    {
        MTR_SCOPE(__FILE__, "init_net");
        init_net(cfg_path.c_str(), weight_path.c_str(), &net_inw, &net_inh, &net_outw, &net_outh, &net_output_count);
    }

    float scale = 0.0f;

    Mat input_blob;

    int frame_count = 0;
    vector<int> top_indices;

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
            netoutdata = run_net(input_blob);
            vector<float> result = { netoutdata, netoutdata + net_output_count };
            int top_k = 5;
            partial_sort(result.begin(), result.begin() + top_k, result.end(), [](float left, float right) -> bool {
                return left > right;
            });
            tick.stop();
            cout << "forward fee: " << tick.getTimeMilli() << "ms" << endl;
        }

        {
            MTR_SCOPE(__FILE__, "viz");
            {
                MTR_SCOPE(__FILE__, "imshow");
                Mat out = float_to_mat(1, net_output_count, 1, netoutdata);
                imshow(TITLE, out);

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
