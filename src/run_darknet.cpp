#include "run_darknet.h"
#include <network.h>

static network *net;

using namespace cv;

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
    net = load_network_custom((char*)cfgfile, (char*)weightfile, 0, 1);
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

image ipl_to_image(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    image out = make_image(w, h, c);
    int i, j, k, count = 0;;

    for (k = 0; k < c; ++k) {
        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                out.data[count++] = data[i*step + j*c + k] / 255.;
            }
        }
    }
    return out;
}

Mat optimize_mat(Mat orig, int max_layer, float scale, float rate, float thresh, int norm)
{
    IplImage ipl = orig;
    image im = ipl_to_image(&ipl);

    optimize_picture(net, im, max_layer, scale, rate, thresh, norm);

    Mat output(orig.rows, orig.cols, orig.type(), im.data, orig.step);
    return output;
}
