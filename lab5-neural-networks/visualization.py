import math

import numpy as np
import keras.models as k_models
import matplotlib.pyplot as plt


def plot_layer(name, activation, row_size=16, plot_width=16):
    
    # calculate size of plot
    size_x, size_y, channels_count = activation.shape
    col_size = math.ceil(channels_count / row_size)
    
    # create display grid
    display = np.zeros((size_y * col_size, size_x * row_size))
    
    # add channels on to display
    for c in range(channels_count):
        
        # calculate row and column
        row = c // row_size
        col = c % row_size
        
        # get channel
        channel = activation[:, :, c]
        
        # normalize to [0,1]
        channel -= np.nanmin(channel)
        channel /= np.nanmax(channel) or 1.0
        
        # add to display
        display[row*size_y:(row+1)*size_y, col*size_x:(col+1)*size_x] = channel
    
    # plot display
    plt.figure(figsize=(plot_width, plot_width/row_size*col_size))
    plt.axis('off')
    plt.imshow(display, cmap="viridis")
    plt.title(name)
    plt.show()
    

def visualize_activations(model, x, select_layers=False, **kwargs):
    
    # default select layers to all (except first one)
    if select_layers is False:
        select_layers = range(1, len(model.layers))
        
    # get layers
    outputs = [model.layers[i].output for i in select_layers]
    names = [model.layers[i].name for i in select_layers]
    
    # create model for calculating activations
    activation_model = k_models.Model(inputs=model.input, outputs=outputs)
    
    # calculate activations
    activations = activation_model.predict(x)
    
    # plot each layer
    for name, activation in zip(names, activations):
        plot_layer(name, activation[0], **kwargs)
