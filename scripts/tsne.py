import pandas as pd
import numpy as np
import re
import csv
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

def crop_center(pil_img, crop_width, crop_height):
    # https://note.nkmk.me/en/python-pillow-square-circle-thumbnail/
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def getImage(path, new_size = 20):
    # return OffsetImage(plt.imread(path))
    img = Image.open(path)
    img = crop_max_square(img)
    img.thumbnail((new_size, new_size), Image.BILINEAR)  # resizes image in-place

    return OffsetImage(img)


pd.set_option('use_inf_as_na', True)

def read_encodings(filename):
    df = pd.read_csv(filename, skiprows=1, header=None)
    labels_df = df.iloc[:, 0]
    # labels_df = labels_df.str.replace(".*img", "", case = False) 

    tokens_df = df.iloc[:, 1:].astype(float)
    tokens_df = tokens_df.clip(-5, 5) # wtf
    tokens_df.fillna(0, inplace=True)
    # print(tokens_df.isnull())

    # find "string" columns
    # df.columns[df.dtypes=='object']
    # print(tokens_df.info())

    # tokens_df.replace([np.inf, -np.inf], np.nan)
    # tokens_df[tokens_df==np.inf]=np.nan
    # print(tokens_df.info())

    # print(tokens_df.info())
    return (labels_df, tokens_df)

'''
out_res: width/height of output square image
out_dim: number of small images in a row/column in output image
'''
def tsne_to_grid(X_2d):
    from lapjv import lapjv
    from scipy.spatial.distance import cdist

    out_dim = np.sqrt(len(X_2d))
    out_dim = int(out_dim)
    to_plot = np.square(out_dim)
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, out_dim), np.linspace(0, 1, out_dim))).reshape(-1, 2)
    cost_matrix = cdist(grid, X_2d[:to_plot], "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]

    return grid_jv, to_plot
    # out = np.ones((out_dim*out_res, out_dim*out_res, 3))

    # to_plot = np.square(out_dim)
    # for pos, img in zip(grid_jv, img_collection[0:to_plot]):
    #     h_range = int(np.floor(pos[0]* (out_dim - 1) * out_res))
    #     w_range = int(np.floor(pos[1]* (out_dim - 1) * out_res))
    #     out[h_range:h_range + out_res, w_range:w_range + out_res]  = image.img_to_array(img)

    # im = image.array_to_img(out)
    # im.save(out_dir + out_name, quality=100)

def tsne_plot(labels, tokens):
    "Creates and TSNE model and plots it"
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    X_2d = tsne_model.fit_transform(tokens)
    X_2d -= X_2d.min(axis=0)
    X_2d /= X_2d.max(axis=0)

    width = 1200
    grid, to_plot = tsne_to_grid(X_2d)
    out_dim = int(width / np.sqrt(to_plot))
   
    fig, ax = plt.subplots(figsize=(width/100, width/100))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    
    for pos, label in zip(grid, labels[0:to_plot]):
        ax.scatter(pos[0], pos[1])
        if False:
            ax.annotate(label,
                     xy=(pos[0], pos[1]),
                     xytext=(5, 2),
                     fontsize=9,
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        ab = AnnotationBbox(getImage(label, new_size = out_dim / 2), (pos[0], pos[1]), frameon=False)
        ax.add_artist(ab)

    plt.show()

if __name__ == '__main__':
    # enc_filename = 'e:/__svn_pool/lightnet/darknet19_448/tiny.txt.enc'
    enc_filename = 'e:/__svn_pool/lightnet/darknet19_448/train.txt.enc'
    labels, tokens = read_encodings(enc_filename)
    tsne_plot(labels, tokens)
