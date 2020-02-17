#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "lightnet.h"
#include "MiniTraceHelper.h"
#include "VideoHelper.h"
#include "os_hal.h"

#define USING_OCV_RESIZE_METHOD 0

#if (_MSC_VER >= 1923)
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif

#define CVUI_IMPLEMENTATION
#include "cvui/cvui.h"

using namespace std;
using namespace cv;

const char* params =
"{ h help ?     | false             | print usage          }"
"{ cfg          | obj.cfg           | file contains model configuration }"
"{ weights      | weights/obj_last.weights   | file contains model weights }"
"{ names        | obj.names         | file contains a list of label names, will be displayer in softmax layer }"
"{@source       | 0                 | source for processing   }"
"{ width        | 0                 | width of video or camera device}"
"{ height       | 0                 | height of video or camera device}"
"{ g gui        | true              | show gui, press g to toggle }"
"{ f fullscreen | false             | show in fullscreen, press f to toggle }"
"{ fps          | 0                 | fps of video or camera device }"
//"{ single_step  |                   | single step mode, press any key to move to next frame }"
"{ l loop       | true              | whether to loop the video}"
"{ video_pos    | 0                 | current position of the video file in milliseconds. }"
"{ offline      | false             | offline mode will produce <source>.csv. }"
"{ encoding     | true              | used with offline mode, will produce <source>.encoding. }"
;

bool is_gui_visible = false;
bool is_fullscreen = false;
bool is_paused = false;
bool is_offline = false;
bool is_encoding = false;

#define APP_NAME "feature-viz"
#define VER_MAJOR 0
#define VER_MINOR 2
#define VER_PATCH 0

#define TITLE APP_NAME " " CVAUX_STR(VER_MAJOR) "." CVAUX_STR(VER_MINOR) "." CVAUX_STR(VER_PATCH)

#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#ifdef DEBUG
#pragma comment(lib, "opencv_world" OPENCV_VERSION "d.lib") 
#else
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib") 
#endif

struct ControlPanel
{
    void setup()
    {
        cvui::init(TITLE);

        getScreenResolution(width, height);
        width -= 100;
        height -= 50;
        canvas = cv::Mat(height, width, CV_8UC3);

        params = imread("assets/params.jpg");
    }

    void update()
    {
        // DEPRECATED
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

    cv::Mat canvas;
    cv::Mat params;
    int width = 1024;
    int height = 768;

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

vector<string> obj_names;
string cfg_path;
string weights_path;
string names_path;
FILE* offline_output_fp = nullptr;
FILE* offline_encoding_fp = nullptr;
shared_ptr<VideoCapture> capture;

ControlPanel panel;
int current_layer_index = 0;
int current_filter_index = 0;
vector<LayerMeta> layer_metas;

void offline()
{
    static int encoding_layer_idx = -1; // encoding layer is the next to last layer before softmax, we will use it for offline debugging & tsne visualization
                                        // http://cs231n.github.io/understanding-cnn/
    static int softmax_layer_idx = -1;
    if (encoding_layer_idx == -1)
    {
        int idx = 0;
        for (auto& meta : layer_metas)
        {
            if (meta.name.find("Softmax") != string::npos)
            {
                softmax_layer_idx = idx;
                encoding_layer_idx = idx - 2;
                break;
            }
            idx++;
        }
    }
    auto name = get_current_image_name(capture);
    printf("%s\n", name.c_str());

    if (is_encoding)
    {
        auto enc_tensors = get_layer_activations(encoding_layer_idx);
        int channel_count = enc_tensors.size();
        fprintf(offline_encoding_fp, "%s", name.c_str());
        for (const auto& tensor : enc_tensors)
        {
            auto ptr = (float*)(tensor.data);
            for (int j = 0; j < tensor.rows; j++) {
                for (int i = 0; i < tensor.cols; i++) {
                    auto v = ptr[tensor.step * j + i];
                    fprintf(offline_encoding_fp, ",%f", v);
                }
            }
        }
        fprintf(offline_encoding_fp, "\n");
    }
    
    auto softmax_tensors = get_layer_activations(softmax_layer_idx);
    int correct_result_idx = -1;
    {
        int channel_count = softmax_tensors.size();
        vector<float> scores(channel_count);
        for (int i = 0; i < channel_count; i++)
        {
            scores[i] = softmax_tensors[i].at<float>(0);
        }
        int K = min(channel_count, 5);
        auto top_indices = top_k_indices(scores.data(), channel_count, K);
        for (int i = 0; i < K; i++)
        {
            auto a = obj_names[top_indices[i]].c_str();
            auto b = name.c_str();
            if (strstr(b, a))
            {
                correct_result_idx = i;
                break;
            }
        }
        fprintf(offline_output_fp, "%s,", name.c_str());
        fprintf(offline_output_fp, "%d,%.3f,", correct_result_idx, correct_result_idx >= 0 ? scores[top_indices[correct_result_idx]] : 0);

        for (int i = 0; i < K; i++)
        {
            if (obj_names.empty() || K > obj_names.size())
            {
                fprintf(offline_output_fp, "#%d,%.3f,", top_indices[i], scores[top_indices[i]]);
            }
            else
            {
                fprintf(offline_output_fp, "%s,%.3f,", obj_names[top_indices[i]].c_str(), scores[top_indices[i]]);
            }
        }
    }

    fprintf(offline_output_fp, "\n");
}

void viz()
{
    {
        MTR_SCOPE(__FILE__, "viz");
        {
            MTR_SCOPE(__FILE__, "imshow");

            int x = 30;
            int y = 10;
            int dy_small = 20;
            int dy_large = 50;
            int width = 300;

            panel.canvas = cv::Scalar(49, 52, 49);
            const int btn_width = 100;
            const int btn_height = 30;
            const int btn_cols = (panel.width - x) / btn_width;

            if (!panel.params.empty())
                cvui::image(panel.canvas, panel.width / 2, 2, panel.params);
            const auto& meta = layer_metas[current_layer_index];
            {
                // meta data
                char info[100];
                cvui::text(panel.canvas, x, y += dy_small, cfg_path);
                cvui::text(panel.canvas, x, y += dy_small, weights_path);
                sprintf(info, "input tensor: %d x %d x %d", meta.input_dim[0], meta.input_dim[1], meta.input_dim[2]);
                cvui::text(panel.canvas, x, y += dy_small, info);
                sprintf(info, "%d filter(s) : %d x %d x %d", meta.filter_count, meta.filter_dim[0], meta.filter_dim[1], meta.filter_dim[2]);
                cvui::text(panel.canvas, x, y += dy_small, info);
                sprintf(info, "output tensor: %d x %d x %d", meta.output_dim[0], meta.output_dim[1], meta.output_dim[2]);
                cvui::text(panel.canvas, x, y += dy_small, info);
            }

            y += dy_small * 4;

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

                int cell_w = (panel.width / 2 - 30) * 0.9f / tensor_cols;
                int cell_h = (panel.height - cell_y0) * 0.9f / tensor_rows;
                if (cell_w > cell_h) cell_w = cell_h;
                else cell_h = cell_w;
                const int cell_spc = min(5, cell_w / 4);

                if (!mode && meta.name.find("Softmax") != string::npos)
                {
                    vector<float> scores(channel_count);
                    for (int i = 0; i < channel_count; i++)
                    {
                        scores[i] = tensors[i].at<float>(0);
                    }
                    int K = min(channel_count, 10);
                    auto top_indices = top_k_indices(scores.data(), channel_count, K);

                    char info[100];
                    for (int i = 0; i < K; i++)
                    {
                        if (obj_names.empty() || K > obj_names.size())
                        {
                            sprintf(info, "#%d: %.2f", top_indices[i], scores[top_indices[i]]);
                        }
                        else
                        {
                            sprintf(info, "%s: %.2f", obj_names[top_indices[i]].c_str(), scores[top_indices[i]]);
                        }
                        cvui::text(panel.canvas, 100, cell_y0 + 20 * i, info);
                    }
                }

                for (int i = 0; i < channel_count; i++)
                {
                    int cell_y = i / tensor_cols;
                    int cell_x = i % tensor_cols;

                    Rect dst_area(cell_x0 + cell_x * (cell_w + 0),
                        cell_y0 + cell_y * (cell_h + 0),
                        cell_w - cell_spc,
                        cell_h - cell_spc);

                    Mat tensor_ = tensors[i];
                    Mat tensor_rgb;
                    Mat tensor_c0;
                    bool is_rgb = tensor_.channels() == 3;

                    if (mode && !is_rgb && current_filter_index < tensor_.channels())
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
                    resize(tensor_rgb, panel.canvas(dst_area), dst_area.size(), 0, 0, INTER_NEAREST);
                }
            }

            cvui::imshow(TITLE, panel.canvas);
        }
    }
}

int main(int argc, char** argv)
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

    cfg_path = parser.get<string>("cfg");
    weights_path = parser.get<string>("weights");
    names_path = parser.get<string>("names");

    if (!fs::exists(cfg_path))
    {
        cout << cfg_path << " can't be found." << endl;
        parser.printMessage();
        return 0;
    }
    if (!fs::exists(weights_path))
    {
        cout << weights_path << " can't be found." << endl;
        parser.printMessage();
        return 0;
    }

    if (fs::exists(names_path))
    {
        ifstream infile(names_path);
        string line;
        while (infile >> line)
        {
            obj_names.push_back(line);
        }
    }

    Mat frame;

    is_gui_visible = parser.get<bool>("gui");
    is_fullscreen = parser.get<bool>("fullscreen");
    is_offline = parser.get<bool>("offline");
    is_encoding = parser.get<bool>("encoding");

    String source = parser.get<String>("@source");

    if (is_offline)
    {
        string name = source + ".csv";
        offline_output_fp = fopen(name.c_str(), "w");
        if (!offline_output_fp)
        {
            printf("Failed to open %s, exiting...\n", name.c_str());
            return -1;
        }
        fprintf(offline_output_fp, "name,correct_order,correct_prob,1st_name,1st_prob,2nd_name,2nd_prob,3rd_name,3rd_prob,\n");

        if (is_encoding)
        {
            string name = source + ".enc";
            offline_encoding_fp = fopen(name.c_str(), "w");
            if (!offline_encoding_fp)
            {
                printf("Failed to open %s, exiting...\n", name.c_str());
                return -1;
            }
            fprintf(offline_encoding_fp, "name,enc0,enc1,enc2,enc3,\n");
        }
    }
    else
    {
        panel.setup();
    }

    bool source_is_camera = false;

    capture = safe_open_video(parser, source, &source_is_camera);

    bool is_running = true;
    bool is_weights_mode = false;

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

    while (is_running)
    {
        MTR_SCOPE_FUNC_I("frame", frame_count);

        {
            MTR_SCOPE(__FILE__, "pre process");
            if (!is_paused)
            {
                if (!safe_grab_video(capture, parser, frame, source, source_is_camera))
                {
                    break;
                }
#if USING_OCV_RESIZE_METHOD
                input_blob = dnn::blobFromImage(frame, 1.0f / 255, Size(net_inw, net_inh), Scalar(), false, true);
#endif
            }
        }

        float* netoutdata = NULL;
        {
            TickMeter tick;
            MTR_SCOPE(__FILE__, "run_net");
            tick.start();
#if USING_OCV_RESIZE_METHOD
            netoutdata = run_net(input_blob);
#else
            netoutdata = run_net_classifier(frame);
#endif
            tick.stop();
            //cout << "forward fee: " << tick.getTimeMilli() << "ms" << endl;
        }

        const auto& meta = layer_metas[current_layer_index];

        if (is_offline)
            offline();
        else
            viz();

        {
            MTR_SCOPE(__FILE__, "waitkey");
            int key = waitKey(1);
            if (key == 27) break;
            if (key == 'f') is_fullscreen = !is_fullscreen;
            if (key == 'p') is_paused = !is_paused;
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
                if (current_filter_index < 0) current_filter_index = meta.filter_dim[2] - 1;
            }
            else if (key == 's')
            {
                current_filter_index++;
                if (current_filter_index > int(meta.filter_dim[2] - 1)) current_filter_index = 0;
            }
        }
        frame_count++;
    }

    if (is_offline)
    {
        if (offline_output_fp)
            fclose(offline_output_fp);
        if (offline_encoding_fp)
            fclose(offline_encoding_fp);
    }

    return 0;
}
