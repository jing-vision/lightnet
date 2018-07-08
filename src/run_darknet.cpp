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

float* run_net(float* indata)
{
    network_predict(*net, indata);
    return net->output;
}

float* run_net(cv::Mat frame)
{
    return run_net(frame.ptr<float>());
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

Mat float_to_mat(int w, int h, int c, float *data)
{
    image im = { w, h, c, data };
    return image_to_mat(im);
}

Mat optimize_mat(Mat orig, int max_layer, float scale, float rate, float thresh, int norm)
{
    image im = mat_to_image(orig);

    optimize_picture(net, im, max_layer, scale, rate, thresh, norm);

    Mat output = image_to_mat(im);
    return output;
}

float* get_network_output_layer(int i)
{
    layer l = net->layers[i];
    if (l.type != REGION) cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

#if 0
image get_convolutional_weight(layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c;
    return float_to_image(w, h, c, l.weights + i*h*w*c);
}

image *get_weights(layer l)
{
    image *weights = (image*)malloc(l.n * sizeof(image));
    int i;
    for (i = 0; i < l.n; ++i) {
        weights[i] = copy_image(get_convolutional_weight(l, i));
        //normalize_image(weights[i]);
    }
    return weights;
}

// TODO: WIP
image *visualize_convolutional_layer(layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
#if 0
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
#endif
    return single_weights;
}
#endif
