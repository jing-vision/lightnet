#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "run_darknet.h"
#include "face_io.h"
#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include <pthread.h>
#include <unistd.h>

#define WINDOW_NAME "demo"

cv::CascadeClassifier face_cascade;
dlib::shape_predictor predictor;
dlib::full_object_detection shape;
int netinw = 0;
int netinh = 0;
int netinarea = 0;
int netoutc = 0;
double epsilon = 0.0;
float *netsrc;
float *embeddings;
cv::Point2f frontface_pts[3];                                   // front facial feature position
cv::Point2f currentface_pts[3];                                 // current facial feature position
cv::Mat im;
cv::Mat gray;
cv::Mat warped_gray;                                            // aligned face image
cv::Scalar wg_mean;                                             // aligned face image mean
cv::Scalar wg_stddev;                                           // aligned face image standard deviation
cv::Mat warped_gray_float;                                      // normalized aligned face image
std::vector<cv::Rect> faces;
int frameidx;
cv::Mat cam_matrix;
cv::Mat dist_coeffs;
std::vector<cv::Point3d> object_pts;
std::vector<cv::Point2d> image_pts;
cv::Mat pose_mat;
cv::Mat euler_angle;
cv::Mat out_intrinsics;
cv::Mat out_rotation;
cv::Mat out_translation;
double last_euler_angle[3];
int savepic_idx;
float *to_be_saved_embeddings;

dlib::cv_image<unsigned char> dkgray;
int puttext_countdown;
char text_to_put[64];

static unsigned char is_recognizing;
static unsigned char is_putting_text;
static unsigned char is_run_convnet_thread;
cv::Scalar globalcolor;
volatile int face_num;
volatile unsigned char is_start_calc_embeddings;

void calc_embeddings
    (
    cv::Mat& aligned_im
    )
{
    aligned_im.convertTo(warped_gray_float, CV_32FC1); // uint8 -> float32
    cv::meanStdDev(aligned_im, wg_mean, wg_stddev);
    double std_adj = std::max(wg_stddev.val[0], epsilon);
    wg_stddev.val[0] = std_adj;
    cv::subtract(warped_gray_float, wg_mean, warped_gray_float);
    cv::divide(warped_gray_float, wg_stddev, warped_gray_float);
    std::memcpy(netsrc, warped_gray_float.data, netinarea * sizeof(float));
    std::memcpy(netsrc + netinarea, warped_gray_float.data, netinarea * sizeof(float));
    std::memcpy(netsrc + netinarea + netinarea, warped_gray_float.data, netinarea * sizeof(float));

    double time_begin = cv::getTickCount();
    float *netdst = run_net(netsrc);
    double fee_time = (cv::getTickCount() - time_begin) / cv::getTickFrequency() * 1000;
    std::cout << "forward fee: " << fee_time << "ms" << std::endl;

    double netdst_squaresum = 0.0f;
    for (int i = 0; i < netoutc; ++i)
        {
        netdst_squaresum += netdst[i] * netdst[i];
        }
    double sqsum_reciprocal = 1.0 / std::sqrt(std::max(netdst_squaresum, epsilon));
    for (int i = 0; i < netoutc; ++i)
        {
        embeddings[i] = netdst[i] * sqsum_reciprocal;
        }
}

void produce_features
    (
    void
    )
{
    const char *username = get_new_username();
    char picbuf[BUFLEN] = {'\0'};
    int i = 0;
    for (i = 0; i < NUM_EMB_EACH_USER; ++i)
        {
        std::sprintf(picbuf, "data/%s/%d.jpg", username, i);
        // TODO: redundant image i/o
        cv::Mat temp = cv::imread(picbuf, 0);
        calc_embeddings(temp);
        std::memcpy(to_be_saved_embeddings + i * netoutc, embeddings, netoutc * sizeof(float));
        }
    save_embeddings(username, to_be_saved_embeddings, netoutc);
}

unsigned char add_newpic
    (
    void
    )
{
    const int SKIP_FRAME_COLLECTION = 10;
    const double ANGLE_DIFF_TH = 5.0;
    char picbuf[BUFLEN] = {'\0'};

    cv::Scalar rendercolor = cv::Scalar(255, 100, 100);
    cv::rectangle(im, cv::Rect(im.cols/2 - 150, im.rows/2 - 150, 300, 300), rendercolor, 2);

    if (face_num == 1) // register one person each time
        {
        cv::putText(im, "Please roll you head slightly", cv::Point(30, 450), 0, 1.0, cv::Scalar(255, 100, 100), 2);
        ++frameidx;

        cv::Point2d left_brow_left_corner = cv::Point2d(shape.part(17).x(), shape.part(17).y());
        cv::Point2d left_brow_right_corner = cv::Point2d(shape.part(21).x(), shape.part(21).y());
        cv::Point2d right_brow_left_corne = cv::Point2d(shape.part(22).x(), shape.part(22).y());
        cv::Point2d right_brow_right_corner = cv::Point2d(shape.part(26).x(), shape.part(26).y());
        cv::Point2d left_eye_left_corner = cv::Point2d(shape.part(36).x(), shape.part(36).y());
        cv::Point2d left_eye_right_corner = cv::Point2d(shape.part(39).x(), shape.part(39).y());
        cv::Point2d right_eye_left_corner = cv::Point2d(shape.part(42).x(), shape.part(42).y());
        cv::Point2d right_eye_right_corner = cv::Point2d(shape.part(45).x(), shape.part(45).y());
        cv::Point2d nose_left_corner = cv::Point2d(shape.part(31).x(), shape.part(31).y());
        cv::Point2d nose_right_corner = cv::Point2d(shape.part(35).x(), shape.part(35).y());
        cv::Point2d mouth_left_corner = cv::Point2d(shape.part(48).x(), shape.part(48).y());
        cv::Point2d mouth_right_corner = cv::Point2d(shape.part(54).x(), shape.part(54).y());
        cv::Point2d mouth_bottom_corner = cv::Point2d(shape.part(57).x(), shape.part(57).y());
        cv::Point2d chin_corner = cv::Point2d(shape.part(8).x(), shape.part(8).y());
        cv::circle(im, left_brow_left_corner, 3, rendercolor, -1);
        cv::circle(im, left_brow_right_corner, 3, rendercolor, -1);
        cv::circle(im, right_brow_left_corne, 3, rendercolor, -1);
        cv::circle(im, right_brow_right_corner, 3, rendercolor, -1);
        cv::circle(im, left_eye_left_corner, 3, rendercolor, -1);
        cv::circle(im, left_eye_right_corner, 3, rendercolor, -1);
        cv::circle(im, right_eye_left_corner, 3, rendercolor, -1);
        cv::circle(im, right_eye_right_corner, 3, rendercolor, -1);
        cv::circle(im, nose_left_corner, 3, rendercolor, -1);
        cv::circle(im, nose_right_corner, 3, rendercolor, -1);
        cv::circle(im, mouth_left_corner, 3, rendercolor, -1);
        cv::circle(im, mouth_right_corner, 3, rendercolor, -1);
        cv::circle(im, mouth_bottom_corner, 3, rendercolor, -1);
        cv::circle(im, chin_corner, 3, rendercolor, -1);

        if (frameidx == SKIP_FRAME_COLLECTION)
            {
            image_pts.push_back(left_brow_left_corner);
            image_pts.push_back(left_brow_right_corner);
            image_pts.push_back(right_brow_left_corne);
            image_pts.push_back(right_brow_right_corner);
            image_pts.push_back(left_eye_left_corner);
            image_pts.push_back(left_eye_right_corner);
            image_pts.push_back(right_eye_left_corner);
            image_pts.push_back(right_eye_right_corner);
            image_pts.push_back(nose_left_corner);
            image_pts.push_back(nose_right_corner);
            image_pts.push_back(mouth_left_corner);
            image_pts.push_back(mouth_right_corner);
            image_pts.push_back(mouth_bottom_corner);
            image_pts.push_back(chin_corner);
            cv::Mat rotation_vec;
            cv::Mat translation_vec;
            // calc pose
            cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);
            // calc euler angle
            cv::Mat rotation_mat;
            cv::Rodrigues(rotation_vec, rotation_mat);
            cv::hconcat(rotation_mat, translation_vec, pose_mat);
            cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);
            image_pts.clear();
            double x_angle = euler_angle.at<double>(0) - last_euler_angle[0];
            double y_angle = euler_angle.at<double>(1) - last_euler_angle[1];
            double z_angle = euler_angle.at<double>(2) - last_euler_angle[2];
            // save user pic in different angle
            if (std::fabs(x_angle) > ANGLE_DIFF_TH || fabs(y_angle) > ANGLE_DIFF_TH || fabs(z_angle) > ANGLE_DIFF_TH)
                {
                const char *username = get_new_username();
                std::sprintf(picbuf, "data/%s/%d.jpg", username, savepic_idx);
                cv::imwrite(picbuf, warped_gray);
                last_euler_angle[0] = euler_angle.at<double>(0);
                last_euler_angle[1] = euler_angle.at<double>(1);
                last_euler_angle[2] = euler_angle.at<double>(2);
                ++savepic_idx;
                }
            frameidx = 0;
            }
        }


    if (savepic_idx == NUM_EMB_EACH_USER)
        {
        savepic_idx = 0;
        return 1;
        }
    else
        {
        return 0;
        }
}

void *convnet_thread_func
    (
    void *ptr
    )
{
    while (is_run_convnet_thread)
        {
        if (is_recognizing)
            {
            //size_t num_face = faces.size();
            //std::cout << std::endl << "[conv] num face:" << num_face << std::endl << std::endl;
            usleep(1);
            //if (face_num > 0)
            if (is_start_calc_embeddings)
                {
                calc_embeddings(warped_gray);
                float bio_confidence = 0.0f;
                int bio_idx = run_embeddings_knn(embeddings, netoutc, &bio_confidence);
                const char *personname = get_username_by_idx(bio_idx);
                is_putting_text = 1;
                std::strcpy(text_to_put, personname);
                if (bio_idx == MAX_NUM_USER)
                    {
                    globalcolor = cv::Scalar(50, 50, 255);
                    }
                else
                    {
                    globalcolor = cv::Scalar(50, 255, 50);
                    }
                }
            }
        else
            {
            usleep(1);
            }
        }
    return 0;
}

void *main_thread_func
    (
    void *ptr
    )
{
    // 1. loading files
    face_cascade.load("weights/haarcascade_frontalface_alt2.xml");
    dlib::deserialize("weights/shape_predictor_68_face_landmarks.dat") >> predictor;
    init_net("weights/facenet.cfg", "weights/facenet.weights", &netinw, &netinh, &netoutc);
    load_embeddings(netoutc);

    std::cout << "netinw: " << netinw << " netinh: " << netinh << " netoutc: " << netoutc << std::endl;

    // 2. init variables, alloc buffers
    netinarea = netinw * netinh;
    epsilon = 1.0 / std::sqrt(netinarea * 3);
    faces.resize(0);
    savepic_idx = 0;
    frameidx = 0;
    puttext_countdown = 0;
    netsrc = new float[netinarea * 3];
    embeddings = new float[netoutc];
    to_be_saved_embeddings = new float[NUM_EMB_EACH_USER * netoutc];
    globalcolor = cv::Scalar(255, 50, 50);

    // 3. init button flags
    unsigned char is_adding_name = 0;
    unsigned char is_registering = 0;

    // 4. frontal face 2D points location, for warpping
    float warpscale = netinw / 160.0;
    frontface_pts[0] = cv::Point2f(58.20558929f * warpscale, 28.47149849f * warpscale);     // left inner eye corner
    frontface_pts[1] = cv::Point2f(99.03411102f * warpscale, 27.64450073f * warpscale);     // right inner eye corner
    frontface_pts[2] = cv::Point2f(80.03263855f * warpscale, 120.09350586f * warpscale);    // bottom lip corner

    // 6. init video capture i/o (opencv)
    cv::VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
        {
        std::cout << "failed to access video0" << std::endl;
        return 0;
        }

    // 7. set intrinsics, corresponding 3d points for pose estimation
    cap >> im;
    double c_x = im.cols / 2.0;
    double c_y = im.rows / 2.0;
    double f_x = c_x * 1.732050808;
    double K[9] = { f_x, 0.0, c_x, 0.0, f_x, c_y, 0.0, 0.0, 1.0 };
    double D[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
    dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);
    object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
    object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
    object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
    object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
    object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
    object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
    object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
    object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
    object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
    object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
    object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
    object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
    object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
    object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner
    pose_mat = cv::Mat(3, 4, CV_64FC1);
    euler_angle = cv::Mat(3, 1, CV_64FC1);
    out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    out_rotation = cv::Mat(3, 3, CV_64FC1);
    out_translation = cv::Mat(3, 1, CV_64FC1);

    // 8. UI settings
    //cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    //cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN );
    cvui::init(WINDOW_NAME, 1);

    while (1)
        {
        cap >> im;
        cv::cvtColor(im, gray, CV_BGR2GRAY);
        dkgray = dlib::cv_image<unsigned char>(gray);
        face_cascade.detectMultiScale(gray, faces, 1.2, 3, CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(160, 160), cv::Size(400, 400));
        face_num = faces.size();
        if (face_num > 0)
            {
            dlib::rectangle dkrect(faces[0].x, faces[0].y, faces[0].x + faces[0].width, faces[0].y + faces[0].height);
            shape = predictor(dkgray, dkrect);
            currentface_pts[0] = cv::Point2f(shape.part(39).x(), shape.part(39).y());       // left inner eye corner
            currentface_pts[1] = cv::Point2f(shape.part(42).x(), shape.part(42).y());       // right inner eye corner
            currentface_pts[2] = cv::Point2f(shape.part(57).x(), shape.part(57).y());       // bottom mouth corner
            cv::Mat tofront_H = cv::getAffineTransform(currentface_pts, frontface_pts);
            cv::warpAffine(gray, warped_gray, tofront_H, cv::Size(netinw, netinw));
            is_start_calc_embeddings = 1;
            }
        else
            {
            is_start_calc_embeddings = 0;
            }

        // button functions
        if (is_recognizing)
            {
            if (face_num > 0)
                {
                cv::line(im, cv::Point(faces[0].x, faces[0].y + faces[0].height), faces[0].br(), globalcolor, 2);
                }
            }
        else if (is_registering)
            {
            unsigned char isadd = 0;
            if (is_adding_name)
                {
                isadd = add_newuser();
                if (isadd == 0)
                    {
                    is_registering = 0;
                    is_putting_text = 1;
                    std::strcpy(text_to_put, "User name already exist");
                    globalcolor = cv::Scalar(50, 50, 255);
                    continue;
                    }
                is_adding_name = 0;
                }
            if (add_newpic() == 1)
                {
                produce_features();
                is_putting_text = 1;
                std::strcpy(text_to_put, "Registration complete");
                globalcolor = cv::Scalar(50, 255, 50);
                is_registering = 0;
                }
            }

        // display text on screen
        if (is_putting_text)
            {
            if (puttext_countdown < 30)
                {
                if (face_num)
                    {
                    cv::putText(im, text_to_put, cv::Point(30, 450), 0, 1.0, globalcolor, 2);
                    }
                ++puttext_countdown;
                }
            else
                {
                puttext_countdown = 0;
                is_putting_text = 0;
                }
            }

        // display buttons
        if (cvui::button(im, 0, 0, 100, 30, "&Add user"))
            {
            if (get_num_user() < MAX_NUM_USER)
                {
                is_recognizing = 0;
                is_registering = 1;
                is_adding_name = 1;
                }
            else
                {
                is_putting_text = 1;
                std::strcpy(text_to_put, "Reach maximum user number");
                globalcolor = cv::Scalar(50, 50, 255);
                }
            }
        if (cvui::button(im, 100, 0, 200, 30, "&Start / stop recognition"))
            {
            if (is_recognizing)
                {
                is_recognizing = 0;
                }
            else
                {
                is_recognizing = 1;
                }
            }
        if (cvui::button(im, 300, 0, 75, 30, "&Quit"))
            {
            break;
            }

        cvui::update();
        cv::imshow("demo", im);
        }

    is_run_convnet_thread = 0;
    face_num = 0;
    free_net();
    free_embedddings();
    delete [] netsrc;
    delete [] embeddings;
    delete [] to_be_saved_embeddings;
    return 0;
}

int main
    (
    void
    )
{
    is_recognizing = 0;
    is_putting_text = 0;
    is_run_convnet_thread = 1;
    pthread_t main_thread;
    pthread_t convnet_thread;
    pthread_create(&main_thread, 0, main_thread_func, 0);
    pthread_create(&convnet_thread, 0, convnet_thread_func, 0);
    pthread_join(main_thread, 0);
    is_run_convnet_thread = 0;
    pthread_join(convnet_thread, 0);
    return 0;
}
