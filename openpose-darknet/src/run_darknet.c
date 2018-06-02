#include <network.h>

static network *net;

void init_net
    (
    char *cfgfile,
    char *weightfile,
    int *inw,
    int *inh,
    int *outw,
    int *outh
    )
{
    net = load_network_custom(cfgfile, weightfile, 0, 1);
    *inw = net->w;
    *inh = net->h;
    *outw = net->layers[net->n - 2].out_w;
    *outh = net->layers[net->n - 2].out_h;
}

float *run_net
    (
    float *indata
    )
{
    network_predict(*net, indata);
    return net->output;
}
