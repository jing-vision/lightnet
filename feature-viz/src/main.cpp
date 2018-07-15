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
"{ cfg          | cfg/darknet.cfg   | model configuration }"
"{ weights      | darknet.weights   | model weights }"
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
#define VER_PATCH 6

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

    auto cfg_path = parser.get<string>("cfg");
    auto weights_path = parser.get<string>("weights");

    Mat frame;

    is_gui_visible = parser.get<bool>("gui");
    is_fullscreen = parser.get<bool>("fullscreen");
    Mat upscale_frame;

    String source = parser.get<String>("@source");

    bool source_is_camera = false;

    VideoCapture capture = safe_open_video(parser, source, &source_is_camera);

    bool is_running = true;
    bool is_weights_mode = false;

    vector<LayerMeta> layer_metas;
    int net_inw = 0;
    int net_inh = 0;
    int net_outw = 0;
    int net_outh = 0;
    int net_output_count = 0;
    {
        MTR_SCOPE(__FILE__, "init_net");
        init_net(cfg_path.c_str(), weights_path.c_str(), &net_inw, &net_inh, &net_outw, &net_outh, &net_output_count);
        layer_metas = get_layer_metas();
    }

    float scale = 0.0f;

    Mat input_blob;

    int frame_count = 0;
    vector<int> top_indices;
    int current_layer_index = 0;
    int current_filter_index = 0;

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
            //cout << "forward fee: " << tick.getTimeMilli() << "ms" << endl;
        }

        {
            MTR_SCOPE(__FILE__, "viz");
            {
                MTR_SCOPE(__FILE__, "imshow");

                int x = 30;
                int y = 10;
                int dy_small = 18;
                int dy_large = 50;
                int width = 300;

                panel.canvas = cv::Scalar(49, 52, 49);
                const int btn_width = 100;
                const int btn_height = 30;
                const int btn_cols = (panel.width - x) / btn_width;

                {
                    // meta data
                    char info[100];
                    cvui::text(panel.canvas, x, y+= dy_small, cfg_path);
                    cvui::text(panel.canvas, x, y += dy_small, weights_path);
                    const auto& meta = layer_metas[current_layer_index];
                    sprintf(info, "filter: %d %d x %d x %d", meta.filter_count, meta.filter_dim[0], meta.filter_dim[1], meta.filter_dim[2]);
                    cvui::text(panel.canvas, x, y += dy_small, info);
                    sprintf(info, "output: %d x %d x %d", meta.output_dim[0], meta.output_dim[1], meta.output_dim[2]);
                    cvui::text(panel.canvas, x, y += dy_small, info);
                }

                y += dy_small * 2;

                // draw buttons
                int layer = 0;
                for (auto layer_meta : layer_metas)
                {
                    int btn_y = layer / btn_cols;
                    int btn_x = layer % btn_cols;
                    int xx = x + btn_width * btn_x;
                    int yy = y + btn_height * btn_y;
                    if (cvui::button(panel.canvas, xx, yy, layer_meta.name))
                    {
                        current_layer_index = layer;
                    }
                    layer++;
                }

                {
                    // draw red circle
                    int layer = current_layer_index;
                    int btn_y = layer / btn_cols;
                    int btn_x = layer % btn_cols;
                    int xx = x + btn_width * btn_x;
                    int yy = y + btn_height * btn_y;
                    circle(panel.canvas, { xx + int(btn_width * 0.5f), yy }, 5, { 0, 0, 255 }, -1);
                }
                bool is_weights_modes[] = { true, false };
                for (bool mode : is_weights_modes)
                {
                    vector<Mat> tensors;
                    if (mode)
                    {
                        tensors = get_layer_weights(current_layer_index);
                    }
                    else
                    {
                        tensors = get_layer_activations(current_layer_index);
                    }
                    int channel_count = tensors.size();
                    int tensor_cols = sqrt(channel_count);
                    int tensor_rows = ceil(channel_count / (float)tensor_cols);
                    int cell_x0 = 30;
                    if (!mode) cell_x0 += panel.width / 2;
                    const int cell_y0 = y + 30 + btn_height * ((layer_metas.size() - 1) / btn_cols + 1);

                    const int cell_spc = 10;
                    int cell_w = (panel.width / 2 - 30) * 0.9f / tensor_cols - cell_spc;
                    int cell_h = (panel.height - cell_y0) * 0.9f / tensor_rows - cell_spc;
                    if (cell_w > cell_h) cell_w = cell_h;
                    else cell_h = cell_w;

                    for (int i = 0; i < tensors.size(); i++)
                    {
                        int cell_y = i / tensor_cols;
                        int cell_x = i % tensor_cols;

                        Rect dst_area(cell_x0 + cell_x * (cell_w + cell_spc),
                            cell_y0 + cell_y * (cell_h + cell_spc),
                            cell_w,
                            cell_h);

                        Mat tensor_ = tensors[i];
                        Mat tensor_rgb;
                        Mat tensor_c0;
                        bool is_rgb = tensor_.channels() == 3;

                        if (mode && !is_rgb)
                        {
                            extractChannel(tensor_, tensor_c0, current_filter_index);
                        }
                        else
                        {
                            extractChannel(tensor_, tensor_c0, 0);
                        }

                        float convert_a = 255;
                        float convert_b = 0;

                        if (tensor_c0.cols != 1 && tensor_c0.rows != 1)
                        {
                            cv::Point min_loc, max_loc;
                            double min_value, max_value;
                            cv::minMaxLoc(tensor_c0, &min_value, &max_value, &min_loc, &max_loc);

                            convert_a = 255 / (max_value - min_value);
                            convert_b = -convert_a * min_value;
                        }

                        if (!is_rgb)
                        {
                            tensor_c0.convertTo(tensor_c0, CV_8UC1, convert_a, convert_b);
                            cvtColor(tensor_c0, tensor_rgb, COLOR_GRAY2BGR);
                        }
                        else
                        {
                            Mat tensor_c1, tensor_c2;
                            extractChannel(tensor_, tensor_c1, 1);
                            extractChannel(tensor_, tensor_c2, 2);

                            tensor_c0.convertTo(tensor_c0, CV_8UC1, convert_a, convert_b);
                            tensor_c1.convertTo(tensor_c1, CV_8UC1, convert_a, convert_b);
                            tensor_c2.convertTo(tensor_c2, CV_8UC1, convert_a, convert_b);

                            Mat channels[] = { tensor_c0, tensor_c1, tensor_c2 };
                            cv::merge(channels, 3, tensor_rgb);
                        }

                        resize(tensor_rgb, panel.canvas(dst_area), Size(cell_w, cell_h), 0, 0, INTER_NEAREST);
                    }
                }

                cvui::imshow(TITLE, panel.canvas);

                {
                    MTR_SCOPE(__FILE__, "waitkey");
                    int key = waitKey(1);
                    if (key == 27) break;
                    if (key == 'f') is_fullscreen = !is_fullscreen;
                    if (key == 'g') is_gui_visible = !is_gui_visible;

                    if (key == 'a')
                    {
                        current_layer_index--;
                        current_filter_index = 0;
                        if (current_layer_index < 0) current_layer_index = layer_metas.size() - 1;
                    }
                    else if (key == 'd')
                    {
                        current_layer_index++;
                        current_filter_index = 0;
                        if (current_layer_index > int(layer_metas.size() - 1)) current_layer_index = 0;
                    }

                    if (key == 'w')
                    {
                        current_filter_index--;
                        if (current_filter_index < 0) current_filter_index = layer_metas[current_layer_index].filter_dim[2] - 1;
                    }
                    else if (key == 's')
                    {
                        current_filter_index++;
                        if (current_filter_index > int(layer_metas[current_layer_index].filter_dim[2] - 1)) current_filter_index = 0;
                    }

                }
            }
        }

        frame_count++;
    }

    return 0;
}
