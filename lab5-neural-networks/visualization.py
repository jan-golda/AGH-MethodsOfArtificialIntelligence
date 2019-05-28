import math

import scipy
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


def generate_heatmap(model, x, shutter=(10, 10), shutter_value=[0,0,0]):
    
    # prepare output matrix
    heatmap = np.zeros(x.shape[1:3])
    
    # get main image prediction
    base_predictions = model.predict(x)[0]
    base_class = base_predictions.argmax()
    base_percentage = base_predictions[base_class]
    
    # slide shutter over image
    for row in range(math.ceil(x.shape[2] / shutter[1])):
        for col in range(math.ceil(x.shape[1] / shutter[0])):
            
            # calculate selector
            sel = (
                0,
                slice(row*shutter[1], min((row+1)*shutter[1], x.shape[2])),
                slice(col*shutter[0], min((col+1)*shutter[0], x.shape[1]))
            )
            
            # copy image and apply shutter
            img = x.copy()
            img[sel] = shutter_value
            
            # calculate prediction
            percentage = model.predict(img)[0][base_class]
            
            # add to heatmap
            heatmap[sel[1:]] = base_percentage - percentage
    
    return heatmap


def plot_heatmap(heatmap, blur=0.0, cmap='jet', **kwargs):
    
    # blur heatmap
    if blur:
        heatmap = scipy.ndimage.filters.gaussian_filter(heatmap, blur)
    
    # plot heatmap
    plt.matshow(heatmap, cmap=cmap, **kwargs)
    plt.colorbar()
    plt.show()

    
def plot_heatmap_cover(heatmap, x, blur=0.0, cutoff=0.0, **kwargs):
    
    # normalize heatmap
    heatmap = heatmap.copy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() or 1.0
    
    # cutof heatmap
    heatmap[heatmap < cutoff] = 0.0
    
    # blur heatmap
    if blur:
        heatmap = scipy.ndimage.filters.gaussian_filter(heatmap, blur)
    
    # cover image
    image = (x - 0.5) * heatmap[:,:,None] + 0.5
    
    plt.imshow(image, **kwargs)
    plt.show()
