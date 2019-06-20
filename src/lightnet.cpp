#include "lightnet.h"
#include <darknet.h>

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
        layer* l = get_network_layer(net, i);
        if (l->type == CONVOLUTIONAL)
        {
            *outw = l->out_w;
            *outh = l->out_h;
            break;
        }
    }
}

vector<LayerMeta> get_layer_metas()
{
    vector<LayerMeta> layer_metas;
    for (int i = 0; i < net->n; i++)
    {
        layer* l = get_network_layer(net, i);
        LAYER_TYPE type = l->type;
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
        else if (type == SCALE_CHANNELS) layerName = "SCALE_CHANNELS";
        else if (type == ACTIVE) layerName = "Active";
        else if (type == RNN) layerName = "RNN";
        else if (type == GRU) layerName = "GRU";
        else if (type == LSTM) layerName = "LSTM";
        else if (type == CONV_LSTM) layerName = "CONV_LSTM";
        else if (type == CRNN) layerName = "CRNN";
        else if (type == BATCHNORM) layerName = "Batchnorm";
        else if (type == NETWORK) layerName = "Network";
        else if (type == XNOR) layerName = "XNOR";
        else if (type == REGION) layerName = "Region";
        else if (type == YOLO) layerName = "Yolo";
        else if (type == ISEG) layerName = "ISEG";
        else if (type == REORG) layerName = "Reorg";
        else if (type == REORG_OLD) layerName = "Reorg_Old";
        else if (type == UPSAMPLE) layerName = "Upsample";
        else if (type == LOGXENT) layerName = "LOGXENT";
        else if (type == L2NORM) layerName = "L2NORM";
        else if (type == EMPTY) layerName = "Empty";
        else if (type == BLANK) layerName = "Blank";
        else layerName = "Unknown";
        char info[100];
        sprintf(info, "%s %d", layerName.c_str(), i);
        LayerMeta meta = {};
        meta.name = info;
        meta.input_dim = { l->w, l->h, l->c };
        meta.output_dim = { l->out_w, l->out_h, l->out_c };
        if (l->type == SOFTMAX)
            meta.output_dim = { 1, 1, l->outputs };

        meta.filter_dim = { l->size, l->size, l->c };
        meta.filter_count = l->n;
        layer_metas.push_back(meta);
    }

    return layer_metas;
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
    Mat frame(im.h, im.w, CV_32FC(im.c));
    int step = frame.step;
    float *data = (float*)(frame.data);

    for (int y = 0; y < im.h; ++y) {
        for (int x = 0; x < im.w; ++x) {
            for (int k = 0; k < im.c; ++k) {
                data[y*step / sizeof(data[0]) + x*im.c + k] = get_pixel(im, x, y, k);
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

vector<Mat> get_layer_activations(int layer_idx)
{
    vector<Mat> activations;

    layer* l = get_network_layer(net, layer_idx);
    if (l->type == REGION) return{};
    if (l->type == ROUTE)
    {
        for (int i = 0; i < l->n; i++)
        {
            int index = l->input_layers[i];
            int input_size = l->input_sizes[i];
            auto acts = get_layer_activations(index);
            for (auto& act : acts)
            {
                activations.emplace_back(act);
            }
        }
    }

    CV_Assert(l->batch == 1 && "TODO: support non-one batch");
#define GPU 1
#if GPU // very hacky
    cuda_pull_array(l->output_gpu, l->output, l->outputs*l->batch);
#endif

    if (l->type == SOFTMAX)
    {
        activations.resize(l->outputs);
        //activations[i] = Mat(l, 1, CV_32F, l->output);
        for (int i = 0; i < l->outputs; i++)
        {
            activations[i] = float_to_mat(1, 1, 1, l->output + i);
        }
    }
    else
    {
        activations.resize(l->out_c);
        for (int i = 0; i < l->out_c; i++)
        {
            activations[i] = float_to_mat(l->out_w, l->out_h, 1, l->output + l->out_w * l->out_h * i);
        }
    }

    return activations;
}


vector<Mat> get_layer_weights(int layer_idx)
{
    vector<Mat> weights;

    layer* l = get_network_layer(net, layer_idx);
    if (l->type == REGION) return{};
    if (l->type == ROUTE) return{};
    if (l->type == YOLO) return{};

    if (l->type == CONNECTED)
    {
        weights.resize(l->out_c);
        for (int i = 0; i < l->out_c; i++)
        {
            weights[i] = float_to_mat(1, 1, 1, l->weights + i);
        }
        return weights;
    }

    weights.resize(l->n);
    int h = l->size;
    int w = l->size;
    int c = l->c;
    for (int i = 0; i < l->n; i++)
    {
        weights[i] = float_to_mat(w, h, c == 3 ? c : min(c, 100), l->weights + w * h * c * i);
    }

    return weights;
}

vector<int> top_k_indices(float *a, int n, int k)
{
    vector<int> indices(k);
    top_k(a, n, k, indices.data());

    return indices;
}
