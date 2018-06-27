// https://pjreddie.com/darknet/nightmare/

// 
// http://pjreddie.com/media/files/vgg-conv.weights
// darknet nightmare cfg/vgg-conv.cfg vgg-conv.weights data/dog.jpg 13
// darknet nightmare cfg/vgg-conv.cfg vgg-conv.weights data/dog.jpg 11 -rounds 4 -range 3

// http://pjreddie.com/media/files/jnet-conv.weights
// darknet nightmare cfg/jnet-conv.cfg jnet-conv.weights data/dog.jpg 13
// darknet nightmare cfg/jnet-conv.cfg jnet-conv.weights data/dog.jpg 11 -rounds 4 -range 3

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "run_darknet.h"
#include "MiniTraceHelper.h"
#include "VideoHelper.h"

#define CVUI_IMPLEMENTATION
#include "cvui/cvui.h"

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

Mat nightmare(Mat org, int max_layer, int range, int norm, int rounds, int iters, int octaves, float rate, float thresh)
{
    Mat im = org.clone();
    for (int e = 0; e < rounds; ++e) {
        fprintf(stderr, "Iteration: ");
        for (int n = 0; n < iters; ++n) {
            fprintf(stderr, "%d, ", n);
            int layer = max_layer + rand() % range - range / 2;
            int octave = rand() % octaves;
            im = optimize_mat(im, layer, 1 / pow(1.33333333, octave), rate, thresh, norm);
        }
    }
    return im;
}


struct ControlPanel
{
    ControlPanel()
    {
        cvui::init(WINDOW_NAME);
    }

    void update()
    {
        int x = 10;
        int y = 0;
        int dy_small = 16;
        int dy_large = 50;
        int width = 300;
        frame = cv::Scalar(49, 52, 49);

        //gui.add(isDreaming.set("dream", false));
        //gui.add(max_layer.set("max_layer", 13, 1, 13));
        //gui.add(iters.set("iterations", 1, 1, 10));
        //gui.add(octaves.set("octaves", 2, 1, 8));
        //gui.add(thresh.set("thresh", 0.85, 0.0, 1.0));
        //gui.add(range.set("range", 3, 1, 10));
        //gui.add(norm.set("norm", 1, 1, 10));
        //gui.add(rate.set("rate", 0.01, 0.0, 0.1));
        //gui.add(blendAmt.set("blendAmt", 0.5, 0.0, 1.0));

        cvui::text(frame, x, y += dy_large, "max_layer");
        cvui::trackbar(frame, x, y += dy_small, width, &max_layer, 1, 13);

        cvui::text(frame, x, y += dy_large, "iterations");
        cvui::trackbar(frame, x, y += dy_small, width, &iterations, 1, 10);

        cvui::text(frame, x, y += dy_large, "octaves");
        cvui::trackbar(frame, x, y += dy_small, width, &octaves, 1, 8);

        cvui::text(frame, x, y += dy_large, "thresh");
        cvui::trackbar(frame, x, y += dy_small, width, &thresh, 0.0f, 1.0f);

        cvui::text(frame, x, y += dy_large, "range");
        cvui::trackbar(frame, x, y += dy_small, width, &range, 1, 10);

        cvui::text(frame, x, y += dy_large, "norm");
        cvui::trackbar(frame, x, y += dy_small, width, &norm, 1, 10);
    
        cvui::text(frame, x, y += dy_large, "rate");
        cvui::trackbar(frame, x, y += dy_small, width, &rate, 0.0f, 0.1f);

        cvui::text(frame, x, y += dy_large, "blendAmt");
        cvui::trackbar(frame, x, y += dy_small, width, &blendAmt, 0.0f, 1.0f);

        y += dy_small;

        cvui::update();
        cv::imshow(WINDOW_NAME, frame);
    }

    const String WINDOW_NAME = "param";
    cv::Mat frame = cv::Mat(770, 350, CV_8UC3);

    // param
    bool dream = false;
    int max_layer = 13;
    int iterations = 1;
    int octaves = 2;
    int range = 3;
    float thresh = 0.85;
    int norm = 1;
    float rate = 0.01;
    float blendAmt = 5;
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

    ControlPanel control;

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
    {
        MTR_SCOPE(__FILE__, "init_net");
        init_net(cfg_path.c_str(), weight_path.c_str(), &net_inw, &net_inh, &net_outw, &net_outh);
    }

    float scale = 0.0f;
    
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
        }

        float *netoutdata = NULL;
        {
            MTR_SCOPE(__FILE__, "run_net");
            double time_begin = getTickCount();

            frame = nightmare(frame, control.max_layer, control.range, control.norm, 1, control.iterations, control.octaves, control.rate, control.thresh);

            double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
            cout << "forward fee: " << fee_time << "ms" << endl;
        }

        {
            MTR_SCOPE(__FILE__, "post process");
        }

        {
            MTR_SCOPE(__FILE__, "viz");
            {
                MTR_SCOPE(__FILE__, "imshow");
                imshow(TITLE, frame);

                MTR_SCOPE(__FILE__, "waitkey");

                control.update();

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
