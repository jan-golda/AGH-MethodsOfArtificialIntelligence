import numpy as np
import keras.backend as k
import keras.preprocessing.image as k_image

from keras.applications import vgg19
from scipy.optimize import fmin_l_bfgs_b
from PIL import Image


def gram_matrix(x):
    """ The gram matrix of an image tensor (feature-wise outer product) """
    assert k.ndim(x) == 3
    
    features = k.batch_flatten(k.permute_dimensions(x, (2, 0, 1)))   
    
    return k.dot(features, k.transpose(features))


def preprocess_image(img):
    """ Formats image into appropriate tensor """
    img = k_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x, img_size):
    """ Formats given tensor into image """
    x = x.reshape((*img_size, 3))
    
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    
    return Image.fromarray(x)


def content_loss(base, generated):
    """ An auxiliary loss function designed to maintain the "content" of the base image in the generated image """
    return k.sum(k.square(generated - base))


def style_loss(style, combination, img_size, channels = 3):
    """
    The "style loss" is designed to maintain the style of the reference image in the generated image.
    It is based on the gram matrices (which capture style) of feature maps from the style reference image and from the generated image.
    """
    assert k.ndim(style) == 3
    assert k.ndim(combination) == 3
    
    s = gram_matrix(style)
    c = gram_matrix(combination)
    
    return k.sum(k.square(s - c)) / (4.0 * (channels ** 2) * ((img_size[0] * img_size[1]) ** 2))


def total_style_loss(outputs, img_size):
    """ Returns combined style loss for given outputs"""
    return sum(style_loss(layer[1], layer[2], img_size) for layer in outputs) / len(outputs)


def total_variation_loss(generated, img_size):
    """ The 3rd loss function, total variation loss, designed to keep the generated image locally coherent """
    assert k.ndim(generated) == 4
    
    a = generated[:, :img_size[0] - 1, :img_size[1] - 1, :] - generated[:, 1:, :img_size[1] - 1, :]
    b = generated[:, :img_size[0] - 1, :img_size[1] - 1, :] - generated[:, :img_size[0] - 1, 1:, :]
    
    return k.sum(k.pow(k.square(a) + k.square(b), 1.25))


def eval_loss_and_grads(x, f_outputs, img_size):
    x = x.reshape((1, *img_size, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class StyleTransfer:
    
    def __init__(self, base_image_path, style_image_path, content_weight = 0.025, style_weight = 1.0, variation_weight = 1.0):
        """ Initializes style transfer by loading images and setting up calculation model """
        
        # set options
        self.base_image_path = base_image_path
        self.style_image_path = style_image_path
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.variation_weight = variation_weight
        
        # init style stansfer
        self._init_images()
        self._init_model()
        self._init_loss()
        self._init_evaluator()
    
    
    def _init_images(self):
        """ Loads images and converts them to tensors """
        
        # calculate size of images
        width, height = k_image.load_img(self.base_image_path).size
        self.img_size = (400, int(width * 400 / height))
        
        # load images
        self.base_image = k_image.load_img(self.base_image_path, target_size=self.img_size)
        self.style_image = k_image.load_img(self.style_image_path, target_size=self.img_size)
        
        # get tensor representations of our images
        self.base_tensor = k.variable(preprocess_image(self.base_image))
        self.style_tensor = k.variable(preprocess_image(self.style_image))
        
        # get placeholder for generated image
        self.generated_tensor = k.placeholder((1, *self.img_size, 3))
        
        # combine all three images into single tensor
        self.input_tensor = k.concatenate([self.base_tensor, self.style_tensor, self.generated_tensor], axis=0)
        
    
    def _init_model(self):
        """ Loads and sets up VGG19 model """
        
        # load VGG19 model with pre-trainde ImageNet weights
        self.model = vgg19.VGG19(input_tensor=self.input_tensor, weights='imagenet', include_top=False)
        
        # get outputs by layer name
        self.layers_outputs = {layer.name: layer.output for layer in self.model.layers}
        
    
    def _init_loss(self):
        """ Inits loss functions """
        
        # get content loss
        self.content_loss = content_loss(self.layers_outputs['block5_conv2'][0], self.layers_outputs['block5_conv2'][2])
        
        # get style loss
        style_outputs = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        style_outputs = [self.layers_outputs[name] for name in style_outputs]
        self.style_loss = total_style_loss(style_outputs, self.img_size)
        
        # get variation loss
        self.variation_loss = total_variation_loss(self.generated_tensor, self.img_size)
        
        # get final loss
        self.loss = (self.content_loss * self.content_weight) + (self.style_loss * self.style_weight) + (self.variation_loss * self.variation_weight)
    
        # create function for retreiving losses
        self.f_losses = k.function([self.generated_tensor], [self.loss, self.content_loss, self.style_loss, self.variation_loss])
    
    def _init_evaluator(self):
        """ Inits avaluating related stuff """
        
        # get the gradients of the generated image wrt the loss
        self.grads = k.gradients(self.loss, self.generated_tensor)
        
        # get outputs function
        self.f_outputs = k.function([self.generated_tensor], [self.loss, *self.grads])
        
        # set preprocessed base image as first result
        self.x = preprocess_image(self.base_image)
        
        # create fields for values
        self.loss_value = None
        self.grad_value = None
    
    
    def _evaluator_loss(self, x):
        assert self.loss_value is None
        
        self.loss_value, self.grad_value = eval_loss_and_grads(x, self.f_outputs, self.img_size)
        
        return self.loss_value

    
    def _evaluator_grads(self, x):
        assert self.loss_value is not None
        
        grad_value = np.copy(self.grad_value)
        self.loss_value = None
        self.grad_value = None
        
        return grad_value
        
    
    def optimize(self):
        """
        Runs scipy-based optimization (L-BFGS) over the pixels of the generated image so as to minimize the neural style loss
        """
        self.x, min_val, info = fmin_l_bfgs_b(self._evaluator_loss, self.x.flatten(), fprime=self._evaluator_grads, maxfun=20)
        print(min_val, info)
    
    
    def get_image(self):
        """ Returns generated image """
        return deprocess_image(self.x.copy(), self.img_size)
    
    
    def get_losses(self):
        """ Calculates current content loss """
        return tuple(self.f_losses([self.x.reshape((1, *self.img_size, 3))]))