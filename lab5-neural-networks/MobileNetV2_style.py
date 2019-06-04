import numpy as np
import keras.backend as k
import keras.preprocessing.image as k_image

from keras.applications import mobilenet_v2
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
    img = mobilenet_v2.preprocess_input(img)
    return img


def content_loss(base, generated):
    """ An auxiliary loss function designed to maintain the "content" of the base image in the generated image """
    return k.sum(k.square(generated - base))


def style_loss(style, combination, img_size, channels=3):
    """
    The "style loss" is designed to maintain the style of the reference image in the generated image.
    It is based on the gram matrices (which capture style) of feature maps
    from the style reference image and from the generated image.
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


class StyleTransfer:
    
    def __init__(self, base_image_path, style_image_path, content_weight=0.025, style_weight=1.0, variation_weight=1.0):
        """ Initializes style transfer by loading images and setting up calculation model """
        
        # set options
        self.base_image_path = base_image_path
        self.style_image_path = style_image_path
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.variation_weight = variation_weight
        
        # init style transfer
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
        self.model = mobilenet_v2.MobileNetV2(input_tensor=self.input_tensor, weights='imagenet', include_top=False)
        
        # get outputs by layer name
        self.layers_outputs = {layer.name: layer.output for layer in self.model.layers}

    def _init_loss(self):
        """ Inits loss functions """
        
        # get content loss
        self.content_loss = content_loss(
            self.layers_outputs['block_9_project'][0],
            self.layers_outputs['block_9_project'][2]
        )
        
        # get style loss
        style_outputs = ['block_1_expand', 'block_1_depthwise', 'block_2_expand', 'block_2_depthwise']
        style_outputs = [self.layers_outputs[name] for name in style_outputs]
        self.style_loss = total_style_loss(style_outputs, self.img_size)
        
        # get variation loss
        self.variation_loss = total_variation_loss(self.generated_tensor, self.img_size)
        
        # get final loss
        self.loss = self.content_loss * self.content_weight
        self.loss += self.style_loss * self.style_weight
        self.loss += self.variation_loss * self.variation_weight
    
        # create function for retrieving losses
        self.losses_function = k.function(
            [self.generated_tensor],
            [self.loss, self.content_loss, self.style_loss, self.variation_loss]
        )
    
    def _init_evaluator(self):
        """ Inits evaluating related stuff """
        
        # get the gradients of the generated image wrt the loss
        self.grads = k.gradients(self.loss, self.generated_tensor)
        
        # get outputs function
        self.loss_grads_function = k.function([self.generated_tensor], [self.loss, *self.grads])
        
        # set preprocessed base image as first result
        self.x = preprocess_image(self.base_image)

    def optimize(self):
        """
        Runs scipy-based optimization (L-BFGS) over the pixels of the generated image
        """
        x, _, _ = fmin_l_bfgs_b(self.evaluate_loss_and_grads, self.x.flatten(), maxfun=20)
        self.x = x.reshape((1, *self.img_size, 3))

    def evaluate_loss_and_grads(self, x):
        # calculate loss and gradients
        x = x.reshape((1, *self.img_size, 3))
        outputs = self.loss_grads_function([x])

        # get loss
        loss_value = outputs[0]

        # get gradients
        grad_values = np.array(outputs[1:]).flatten().astype('float64')

        return loss_value, grad_values

    def get_image(self):
        """ Returns generated image """
        x = self.x[0].copy().reshape((*self.img_size, 3))

        x = (x + 1) / 2 * 255
        x = np.clip(x, 0, 255).astype('uint8')
        return Image.fromarray(x)

    def get_losses(self):
        """ Calculates current content loss """
        return tuple(self.losses_function([self.x.reshape((1, *self.img_size, 3))]))
