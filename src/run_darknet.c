#include <network.h>

static network *net;

void init_net
(
    const char *cfgfile,
    const char *weightfile,
    int *inw,
    int *inh,
    int *outw,
    int *outh
)
{
    net = load_network_custom(cfgfile, weightfile, 0, 1);
    *inw = net->w;
    *inh = net->h;

    layer* last_layer = get_network_layer(net, net->n - 2);
    *outw = last_layer->out_w;
    *outh = last_layer->out_h;
}

float *run_net
(
    float *indata
)
{
    network_predict(*net, indata);
    return net->output;
}
