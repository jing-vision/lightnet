#ifndef RUN_DARKNET_H
#define RUN_DARKNET_H

#include <opencv2/core.hpp>

void init_net
(
    const char *cfgfile,
    const char *weightfile,
    int *inw,
    int *inh,
    int *outw,
    int *outh
);

float *run_net
(
    float *indata
);

cv::Mat optimize_mat(cv::Mat orig, int max_layer, float scale, float rate, float thresh, int norm);

#endif // RUN_DARKNET_H
