#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "lightnet.h"
#include "MiniTraceHelper.h"
#include "VideoHelper.h"

#define CVUI_IMPLEMENTATION
#include "cvui/cvui.h"

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
;

bool is_gui_visible = false;
bool is_fullscreen = false;

#define APP_NAME "feature-viz"
#define VER_MAJOR 0
#define VER_MINOR 1
#define VER_PATCH 1

#define TITLE APP_NAME " " CVAUX_STR(VER_MAJOR) "." CVAUX_STR(VER_MINOR) "." CVAUX_STR(VER_PATCH)


struct ControlPanel
{
    ControlPanel()
    {
        cvui::init(TITLE);
    }

    void update()
    {
        int x = 10;
        int y = 0;
        int dy_small = 16;
        int dy_large = 50;
        int width = 300;
        canvas = cv::Scalar(49, 52, 49);

        cvui::button(canvas, x, y += dy_large, "max_layer");
        cvui::trackbar(canvas, x, y += dy_small, width, &max_layer, 1, 13);

        cvui::imshow(TITLE, canvas);
    }

    const int width = 1800;
    const int height = 1080;
    cv::Mat canvas = cv::Mat(height, width, CV_8UC3);

    // param
    bool dream = false;
    int max_layer = 13;
    int iterations = 1;
    int octaves = 2;
    int range = 3;
    float thresh = 0.85;
    int norm = 1;
    float rate = 0.05;
    float blendAmt = 0.5;
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

    ControlPanel panel;
    MiniTraceHelper _;

    auto cfg_path = parser.get<string>("proto");
    auto weight_path = parser.get<string>("model");

    Mat frame;

    is_gui_visible = parser.get<bool>("gui");
    is_fullscreen = parser.get<bool>("fullscreen");
    Mat upscale_frame;

    String source = parser.get<String>("@source");

    bool source_is_camera = false;

    VideoCapture capture = safe_open_video(parser, source, &source_is_camera);

    bool is_running = true;

    vector<string> layer_names;
    int net_inw = 0;
    int net_inh = 0;
    int net_outw = 0;
    int net_outh = 0;
    int net_output_count = 0;
    {
        MTR_SCOPE(__FILE__, "init_net");
        init_net(cfg_path.c_str(), weight_path.c_str(), &net_inw, &net_inh, &net_outw, &net_outh, &net_output_count);
        layer_names = get_layer_names();
    }

    float scale = 0.0f;

    Mat input_blob;

    int frame_count = 0;
    vector<int> top_indices;
    int current_viz_layer = 0;

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

                int x = 30;
                int y = 30;
                int dy_small = 16;
                int dy_large = 50;
                int width = 300;
                panel.canvas = cv::Scalar(49, 52, 49);

                int layer = 0;
                const int btn_width = 100;
                for (auto layer_name : layer_names)
                {
                    if (cvui::button(panel.canvas, x + btn_width * layer, y, layer_name))
                    {
                        current_viz_layer = layer;
                    }
                    layer++;
                }
                circle(panel.canvas, { x + int(btn_width * (current_viz_layer + 0.5f)), y }, 5, { 0, 0, 255 }, -1);

                vector<Mat> tensor = get_layer_output_tensor(current_viz_layer);
                int channel_count = tensor.size();
                int tensor_cols = sqrt(channel_count);
                int tensor_rows = ceil(channel_count / (float)tensor_cols);
                const int cell_x0 = 30;
                const int cell_y0 = 100;
                const int cell_spc = 10;
                const int cell_w = (panel.width - cell_x0) * 0.9f / tensor_cols - cell_spc;
                const int cell_h = (panel.height - cell_y0) * 0.9f / tensor_rows - cell_spc;

                for (int i = 0; i < tensor.size(); i++)
                {
                    int cell_y = i / tensor_cols;
                    int cell_x = i % tensor_cols;

                    Rect dst_area(cell_x0 + cell_x * (cell_w + cell_spc), 
                        cell_y0 + cell_y * (cell_h + cell_spc),
                        cell_w,
                        cell_h);

                    Mat channel_8uc1 = tensor[i];
                    Mat channel_8uc3;
                    cvtColor(channel_8uc1, channel_8uc3, COLOR_GRAY2BGR);
                    resize(channel_8uc3, panel.canvas(dst_area), Size(cell_w, cell_h), 0, 0, INTER_NEAREST);
                    //imshow(title, channel);
                }

                //cvui::trackbar(canvas, x, y += dy_small, width, &max_layer, 1, 13);

                cvui::imshow(TITLE, panel.canvas);
            }

            {
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
