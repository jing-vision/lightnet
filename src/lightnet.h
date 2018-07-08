#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>

void init_net(const char *cfgfile,  const char *weightfile,
    int *inw, int *inh, 
    int *outw, int *outh,
    int* net_output_count = NULL);

float* run_net(float* indata);
float* run_net(cv::Mat frame);

cv::Mat float_to_mat(int w, int h, int c, float *data);

cv::Mat optimize_mat(cv::Mat orig, int max_layer, float scale, float rate, float thresh, int norm);

int get_layer_count();
std::vector<std::string> get_layer_names();
std::vector<cv::Mat> get_layer_output_tensor(int layer_idx);
