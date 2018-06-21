#include <iostream>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#include "run_darknet.h"
#include "post_process.h"

#include "minitrace/minitrace.h"
#include "readerwriterqueue/readerwriterqueue.h"

#define CVUI_IMPLEMENTATION
#include "cvui/cvui.h"

using namespace moodycamel;

const char* params =
"{ help ?       | false             | print usage          }"
"{ proto        |openpose.cfg       | model configuration }"
"{ model        |openpose.weight    | model weights }"
"{ c camera     | 0                 | camera device number }"
"{ w width      | 0                 | width of video or camera device}"
"{ h height     | 0                 | height of video or camera device}"
"{ gui          | false             | show gui, you can also press SPACEBAR to toggle }"
"{ fps          | 0                 | fps of video or camera device }"
"{ image video  |                   | video or image for detection }"
"{ single_step  |                   | single step mode, press any key to move to next frame }"
"{ loop         | true              | whether to loop the video}"
"{ video_pos    | 0                 | current position of the video file in milliseconds. }"
"{ data_write_dir  |                | enables data write mode }"
"{ data_read_dir   |                | enables data read mode }"
;

bool is_gui_visible = false;


// Recommended macros:
//      MTR_SCOPE(__FILE__, "post processing");
//      MTR_SCOPE_FUNC();
//      MTR_META_THREAD_NAME("reader");
struct MiniTraceHelper
{
    void setup()
    {
        mtr_init("trace.json");
        mtr_register_sigint_handler();
        MTR_META_PROCESS_NAME("main process");
        MTR_META_THREAD_NAME("0) main thread");
    }

    ~MiniTraceHelper()
    {
        mtr_flush();
        mtr_shutdown();
    }
};

#define CONCURRENT_PKT_COUNT 1

struct NetOutpus
{
    vector<float> net_outputs;
    Mat frame;
    int idx;
};

ReaderWriterQueue<NetOutpus> q_output(CONCURRENT_PKT_COUNT);

struct ParamWindow
{
    void setup()
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

        cvui::text(frame, x, y += dy_large, "find_heatmap_peaks_thresh");
        cvui::trackbar(frame, x, y += dy_small, width, &find_heatmap_peaks_thresh, 0.0f, 1.0f);
        
        y += dy_small;

        cvui::text(frame, x, y += dy_large, "body_inter_min_above_th");
        cvui::trackbar(frame, x, y += dy_small, width, &body_inter_min_above_th,0, 20);

        cvui::text(frame, x, y += dy_large, "body_inter_th");
        cvui::trackbar(frame, x, y += dy_small, width, &body_inter_th, 0.0f, 1.0f);

        cvui::text(frame, x, y += dy_large, "body_min_subset_cnt");
        cvui::trackbar(frame, x, y += dy_small, width, &body_min_subset_cnt, 0, 20);

        cvui::text(frame, x, y += dy_large, "body_min_subset_score");
        cvui::trackbar(frame, x, y += dy_small, width, &body_min_subset_score, 0.0f, 1.0f);

        y += dy_small;

        cvui::text(frame, x, y += dy_large, "render_thresh");
        cvui::trackbar(frame, x, y += dy_small, width, &render_thresh, 0.0f, 1.0f);

        y += dy_small;

        cvui::update();
        cv::imshow(WINDOW_NAME, frame);
    }

    const String WINDOW_NAME = "param";
    cv::Mat frame = cv::Mat(770, 350, CV_8UC3);

    // params
    float find_heatmap_peaks_thresh = 0.05;

    int body_inter_min_above_th = 9;
    float body_inter_th = 0.05;
    int body_min_subset_cnt = 6;
    float body_min_subset_score = 0.4;

    float render_thresh = 0.05;
};

int main(int argc, char **argv)
{
    MiniTraceHelper mr_hepler;
    mr_hepler.setup();

    ParamWindow param_window;

    CommandLineParser parser(argc, argv, params);
    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    auto cfg_path = parser.get<string>("proto");
    auto weight_path = parser.get<string>("model");

    Mat frame;

    // 1. read args
    is_gui_visible = parser.get<bool>("gui");

    VideoCapture cap;
    String video_path = parser.get<String>("video");
    if (video_path.empty())
    {
        MTR_SCOPE(__FILE__, "cap.open(camera)");
        int camera = parser.get<int>("camera");
        cap.open(camera);
        if (!cap.isOpened())
        {
            cout << "Couldn't find camera: " << camera << endl;
            return -1;
        }
    }
    else
    {
        MTR_SCOPE(__FILE__, "cap.open(video_path)");
        if (frame.empty())
        {
            cap.open(video_path);
            if (!cap.isOpened())
            {
                cout << "Couldn't open image / video: " << video_path << endl;
                return -1;
            }
        }
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

    std::thread CUDA([&] {
        MTR_META_THREAD_NAME("2) CUDA");
        int frame_count = 0;
        while (is_running)
        {
            if (cap.isOpened())
            {
                MTR_SCOPE(__FILE__, "cap >> frame");
                cap >> frame; // get a new frame from camera/video or read image
                if (video_path.empty())
                {
                    MTR_SCOPE(__FILE__, "flip");
                    flip(frame, frame, 1);
                }
            }

            if (frame.empty())
            {
                if (parser.get<bool>("loop"))
                {
                    cap.open(video_path);
                    continue;
                }
                break;
            }

            MTR_SCOPE_FUNC_I("frame", frame_count);

            {
                MTR_SCOPE(__FILE__, "pre process");

                // 3. resize to net input size, put scaled image on the top left
                Mat netim = create_netsize_im(frame, net_inw, net_inh, &scale);

                // 4. normalized to float type
                netim.convertTo(netim, CV_32F, 1 / 256.f, -0.5);

                {
                    // 5. split channels
                    MTR_SCOPE(__FILE__, "split");
                    float *netin_data_ptr = net_inputs.data();
                    vector<Mat> input_channels;
                    for (int i = 0; i < 3; ++i)
                    {
                        Mat channel(net_inh, net_inw, CV_32FC1, netin_data_ptr);
                        input_channels.emplace_back(channel);
                        netin_data_ptr += (net_inw * net_inh);
                    }
                    split(netim, input_channels);
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

            NetOutpus pkt;
            pkt.net_outputs = { netoutdata, netoutdata + net_outh*net_outw*NET_OUT_CHANNELS };
            pkt.idx = frame_count;
            pkt.frame = frame;
            q_output.try_emplace(pkt);

            frame_count++;
        }

        is_running = false;
    });

    std::thread postCUDA([&]() {
        int frame_count = 0;
        MTR_META_THREAD_NAME("3) post CUDA");

        vector<float> heatmap_peaks(3 * (POSE_MAX_PEOPLE + 1) * (NET_OUT_CHANNELS - 1));
        vector<float> heatmap(net_inw * net_inh * NET_OUT_CHANNELS);

        param_window.setup();

        while (is_running)
        {
            NetOutpus pkt;
            if (!q_output.try_dequeue(pkt))
                continue;

            MTR_SCOPE_FUNC_I("frame", pkt.idx);
            frame_count++;

            float* netoutdata = pkt.net_outputs.data();

            vector<float> keypoints;
            vector<int> keyshape;
            {
                MTR_SCOPE(__FILE__, "post process");

                // 7. resize net output back to input size to get heatmap
                {
                    for (int i = 0; i < NET_OUT_CHANNELS; ++i)
                    {
                        MTR_SCOPE(__FILE__, "resize");
                        Mat netout(net_outh, net_outw, CV_32F, (netoutdata + net_outh*net_outw*i));
                        Mat nmsin(net_inh, net_inw, CV_32F, heatmap.data() + net_inh*net_inw*i);
                        resize(netout, nmsin, Size(net_inw, net_inh), 0, 0, CV_INTER_CUBIC);
                    }
                }

                // 8. get heatmap peaks
                find_heatmap_peaks(heatmap.data(), heatmap_peaks.data(), net_inw, net_inh, NET_OUT_CHANNELS, param_window.find_heatmap_peaks_thresh);

                // 9. link parts
                connect_bodyparts(keypoints, heatmap.data(), heatmap_peaks.data(), net_inw, net_inh, 
                    param_window.body_inter_min_above_th,
                    param_window.body_inter_th,
                    param_window.body_min_subset_cnt,
                    param_window.body_min_subset_score,
                    keyshape);
            }

            {
                MTR_SCOPE(__FILE__, "viz");
                // 10. draw result
                render_pose_keypoints(pkt.frame, keypoints, keyshape, param_window.render_thresh, scale);

                {
                    const int num_keypoints = keyshape[1];
                    for (int person = 0; person < keyshape[0]; ++person)
                    {
                        for (int part = 0; part < num_keypoints; ++part)
                        {
                            const int index = (person * num_keypoints + part) * keyshape[2];
                            if (keypoints[index + 2] > param_window.render_thresh)
                            {
                                Point center{keypoints[index] * scale, keypoints[index + 1] * scale};
                                if (center.y < 200)
                                {
                                }
                            }
                        }
                    }
                }

                // 11. show and save result
                {
                    MTR_SCOPE(__FILE__, "imshow");
                    cout << "people: " << shape[0] << endl;
                    imshow("jing-pose", pkt.frame);

                    if (is_gui_visible)
                    {
                        param_window.update();
                    }
                }

                {
                    MTR_SCOPE(__FILE__, "waitkey");
                    if (waitKey(1) == 27) break;
                }
            }
        }
        is_running = false;
    });

    CUDA.join();
    postCUDA.join();

    return 0;
}
