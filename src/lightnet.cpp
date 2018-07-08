#include "lightnet.h"
#include <network.h>

using namespace std;
using namespace cv;

static network *net;

void init_net(const char *cfgfile, const char *weightfile,
    int *inw, int *inh,
    int *outw, int *outh,
    int* net_output_count)
{
    net = load_network_custom((char*)cfgfile, (char*)weightfile, 0, 1);
    *inw = net->w;
    *inh = net->h;

    if (net_output_count)
    {
        *net_output_count = net->outputs;
    }
    for (int i = net->n - 1; i > 0; i--)
    {
        layer* lay = get_network_layer(net, i);
        if (lay->type == CONVOLUTIONAL)
        {
            *outw = lay->out_w;
            *outh = lay->out_h;
            break;
        }
    }
}

vector<string> get_layer_names()
{
    vector<string> layerNames;
    for (int i = 0; i < net->n; i++)
    {
        layer* lay = get_network_layer(net, i);
    }

    for (int i = 0; i<net->n; i++)
    {
        LAYER_TYPE type = net->layers[i].type;
        string layerName;
        if (type == CONVOLUTIONAL) layerName = "Conv";
        else if (type == DECONVOLUTIONAL) layerName = "Deconv";
        else if (type == CONNECTED) layerName = "FC";
        else if (type == MAXPOOL) layerName = "MaxPool";
        else if (type == SOFTMAX) layerName = "Softmax";
        else if (type == DETECTION) layerName = "Detect";
        else if (type == DROPOUT) layerName = "Dropout";
        else if (type == CROP) layerName = "Crop";
        else if (type == ROUTE) layerName = "Route";
        else if (type == COST) layerName = "Cost";
        else if (type == NORMALIZATION) layerName = "Normalize";
        else if (type == AVGPOOL) layerName = "AvgPool";
        else if (type == LOCAL) layerName = "Local";
        else if (type == SHORTCUT) layerName = "Shortcut";
        else if (type == ACTIVE) layerName = "Active";
        else if (type == RNN) layerName = "RNN";
        else if (type == GRU) layerName = "GRU";
        else if (type == CRNN) layerName = "CRNN";
        else if (type == BATCHNORM) layerName = "Batchnorm";
        else if (type == NETWORK) layerName = "Network";
        else if (type == XNOR) layerName = "XNOR";
        else if (type == REGION) layerName = "Region";
        else if (type == YOLO) layerName = "Yolo";
        else if (type == REORG) layerName = "Reorg";
        else if (type == UPSAMPLE) layerName = "Upsample";
        else if (type == REORG_OLD) layerName = "Reorg_old";
        else if (type == BLANK) layerName = "Blank";
        else layerName = "Unknown";
        char info[100];
        sprintf(info, "%s %d", layerName.c_str(), i);
        layerNames.push_back(info);
    }

    return layerNames;
}

float* run_net(float* indata)
{
    float* output = network_predict(*net, indata);
    return output;
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
    Mat frame(im.h, im.w, CV_8UC(im.c));
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

int get_layer_count()
{
    return net->n;
}

vector<Mat> get_layer_output_tensor(int layer_idx)
{
    vector<Mat> tensor;

    layer l = net->layers[layer_idx];
    if (l.type != REGION)
    {
        CV_Assert(l.batch == 1 && "TODO: support non-one batch");
        cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
        tensor.resize(l.out_c);
        for (int i = 0; i < l.out_c; i++)
        {
            tensor[i] = float_to_mat(l.out_w, l.out_h, 1, l.output + l.out_w * l.out_h * i);
        }
    }

    return tensor;
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
