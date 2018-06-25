#ifndef RUN_DARKNET_H
#define RUN_DARKNET_H

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

#endif // RUN_DARKNET_H
