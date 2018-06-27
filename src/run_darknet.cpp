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

static image mat_to_image(Mat src)
{
    uint8_t *data = (uint8_t *)src.data;
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    int step = src.step;
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

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

static Mat image_to_mat(image im)
{
    Mat frame(im.h, im.w, CV_8UC3);
    int step = frame.step;
    uint8_t *data = (uint8_t*)(frame.data);

    for (int y = 0; y < im.h; ++y) {
        for (int x = 0; x < im.w; ++x) {
            for (int k = 0; k < im.c; ++k) {
                data[y*step + x*im.c + k] = (uint8_t)(get_pixel(im, x, y, k) * 255);
            }
        }
    }

    return frame;
}

Mat optimize_mat(Mat orig, int max_layer, float scale, float rate, float thresh, int norm)
{
    image im = mat_to_image(orig);

    optimize_picture(net, im, max_layer, scale, rate, thresh, norm);

    Mat output = image_to_mat(im);
    return output;
}
