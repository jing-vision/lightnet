#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

cv::String get_current_image_name(std::shared_ptr<cv::VideoCapture> cap);

std::shared_ptr<cv::VideoCapture> safe_open_video(const cv::CommandLineParser &parser, const cv::String &source, bool *source_is_camera = nullptr);

bool safe_grab_video(std::shared_ptr<cv::VideoCapture> cap, const cv::CommandLineParser &parser, cv::Mat& frame, const cv::String& source, bool source_is_camera);
