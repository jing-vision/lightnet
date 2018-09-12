#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::VideoCapture safe_open_video(const cv::CommandLineParser &parser, const cv::String &source, bool *source_is_camera = nullptr);

bool safe_grab_video(cv::VideoCapture& cap, const cv::CommandLineParser &parser, cv::Mat& frame, const cv::String& source, bool source_is_camera);
