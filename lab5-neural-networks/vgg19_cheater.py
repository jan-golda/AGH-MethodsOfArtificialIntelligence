import numpy as np

import keras.backend as k
import keras.preprocessing.image as k_image

from keras.applications import vgg19
from scipy.optimize import fmin_l_bfgs_b
from PIL import Image


class ClassCheater:

    def __init__(self, image_path, target_class):
        self.image_path = image_path
        self.target_class = target_class

        # load model
        self.model = vgg19.VGG19(weights='imagenet', include_top=True)

        # get loss
        self.loss = -k.log(self.model.output[0, target_class])

        # get gradients
        self.gradients = k.gradients(self.loss, self.model.input)

        # get loss and grads function
        self.loss_grads_function = k.function([self.model.input], [self.loss, *self.gradients])

        # load image
        self.image = k_image.load_img(self.image_path, target_size=(224, 224))

        # change image into tensor and preprocess it
        self.x = k_image.img_to_array(self.image)
        self.x = np.expand_dims(self.x, axis=0)
        self.x = vgg19.preprocess_input(self.x)

    def optimize(self):
        x, min_val, info = fmin_l_bfgs_b(self.evaluate_loss_and_grads, self.x.flatten())
        self.x = x.reshape((1, 224, 224, 3))

    def evaluate_loss_and_grads(self, x):
        # get loss and gradients
        x = x.reshape((1, 224, 224, 3))
        outputs = self.loss_grads_function([x])

        # get loss
        loss = outputs[0]

        # get gradients
        grads = np.array(outputs[1:]).flatten().astype('float64')

        return loss, grads

    def get_predictions(self, top=5):
        return vgg19.decode_predictions(self.model.predict(self.x), top=top)

    def get_image(self):
        x = self.x[0].copy()

        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68

        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')

        return Image.fromarray(x)
