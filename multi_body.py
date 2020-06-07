
import os
import sys
import data_loader as dl
import numpy as np
from models.fc2_20_2dense import FC2_20_2Dense

from urllib.request import urlopen
import tensorflow as tf
from keras import optimizers as opti
import keras.backend.tensorflow_backend as KTF

import matplotlib
import matplotlib.pyplot as plt

def get_sequence(proteinid):
##This function gets protein fasta sequence from uniprot
    link = 'https://www.uniprot.org/uniprot/'+proteinid+'.fasta'          
    myfile = urlopen(link).read().decode('utf-8')
    return ''.join(myfile.split('\n')[1:])

def factory_optimizer( optimizer_name, lr = 0.001 ):
    if optimizer_name == 'sgd':
        return opti.SGD( lr=lr ), 'sgd'
    elif optimizer_name == 'rmsprop':
        return opti.RMSprop( lr=lr ), 'rmsprop'
    elif optimizer_name == 'adagrad':
        return opti.Adagrad( lr=lr ), 'adagrad'
    elif optimizer_name == 'adadelta':
        return opti.Adadelta( lr=lr ), 'adadelta'
    elif optimizer_name == 'adamax':
        return opti.Adamax( lr=lr ), 'adamax'
    elif optimizer_name == 'nadam':
        return opti.Nadam( lr=lr ), 'nadam'
    else:
        return opti.Adam( lr=lr ), 'adam'
def factory_model( model_name ):
    model_name == 'fc2_20_2dense'
    return FC2_20_2Dense(), 'fc2_20_2dense'
    
def comput_prob(p1,p2,remove_polyq):

    seq_p1=get_sequence(p1)
    seq_p2=get_sequence(p2)
    if remove_polyq==1:
        seq_p1=seq_p1.replace('QQQQQQQQQQQQQQQQQQQQQQQ','')
        seq_p2=seq_p2.replace('QQQQQQQQQQQQQQQQQQQQQQQ','')

    text_file = open("temp_test.txt", "w")
    text_file.write('1166'+'\n'+p1+' '+p2+' '+seq_p1+' '+seq_p2+' '+str(1))
    text_file.close()
    model_name='fc2_20_2dense'
    if 'embed' in model_name:
        test_p1, test_p2, test_labels = dl.load_data_embed( "temp_test.txt" )

    else:
        test_p1, test_p2, test_labels = dl.load_data( "temp_test.txt" )
        test_data = [test_p1, test_p2]
    
    abstract_model, model_name = factory_model( model_name )
    model = abstract_model.get_model()
    optimizer, optimizer_name = factory_optimizer( 'adam', 0.001 )
    model.compile( optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'] )    
    model.load_weights('fc2_20_2dense_model.h5')
    
    score, acc = model.evaluate( test_data, test_labels )

    predict = model.predict( test_data, batch_size=64, verbose=1 )

    #print('The interaction probability between '+p1+' and '+p2+' is: ' +str(predict[0][0]) )

    return predict[0][0]

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



#receptor list
p1_list=['P40337','Q96SW2','P98170','Q00987','O43791']
#ligand list
p2_list=['P10275','P03968','P16234','Q9GZP0','P41181']

ps=np.zeros((len(p1_list),len(p2_list)))
for i in range(len(p1_list)):
    for j in range(len(p2_list)):
        ps[i][j]=comput_prob(p1_list[i],p2_list[j],1)

print(ps)

fig, ax = plt.subplots()

im, cbar = heatmap(ps, p1_list, p2_list, ax=ax,
                   cmap="YlGn", cbarlabel="Probability")
texts = annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()
plt.show()
