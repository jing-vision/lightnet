import pandas as pd
import numpy as np
import re
import csv
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

pd.set_option('use_inf_as_na', True)

def read_encodings(filename):
    df = pd.read_csv(filename, skiprows=1, header=None)
    labels_df = df.iloc[:, 0]
    labels_df = labels_df.str.replace(".*img", "", case = False) 

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

def tsne_plot(labels, tokens):
    "Creates and TSNE model and plots it"
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(12, 12)) 
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     fontsize=9,
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

if __name__ == '__main__':
    enc_filename = 'e:/__svn_pool/lightnet/darknet19_448/train.txt.enc'
    labels, tokens = read_encodings(enc_filename)
    tsne_plot(labels, tokens)
