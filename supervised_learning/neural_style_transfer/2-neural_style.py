#!/usr/bin/env python3
"""Neural Style Transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1',
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor

        style_image - the image used as a style reference, stored as
            a numpy.ndarray
        content_image - the image used as a content reference, stored
            as a numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
                len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or \
                len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(alpha, (int, float))) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (not isinstance(beta, (int, float))) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        if hasattr(tf, 'enable_eager_execution'):
            tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0
        and 1 and its largest side is 512 pixels

        image - a numpy.ndarray of shape (h, w, 3) containing the
            image to be scaled

        Returns: the scaled image
        """
        if not isinstance(image, np.ndarray) or \
                len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if w > h:
            w_new = 512
            h_new = int(h * (512 / w))
        else:
            h_new = 512
            w_new = int(w * (512 / h))

        image = np.expand_dims(image, axis=0)

        if hasattr(tf.image, 'resize_bicubic'):
            image = tf.image.resize_bicubic(image, (h_new, w_new))
        else:
            image = tf.image.resize(
                image, (h_new, w_new), method='bicubic')

        image = tf.clip_by_value(image / 255, 0, 1)

        return image

    def load_model(self):
        """
        Creates the model used to calculate cost

        the model's input should be the same as the VGG19 input
        the model's output should be a list containing the outputs
            of the VGG19 layers listed in style_layers followed by
            content_layer

        Saves the model in the instance attribute model
        """
        VGG19_model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        VGG19_model.save("VGG19_base_model")
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model(
            "VGG19_base_model", custom_objects=custom_objects)

        style_outputs = []
        content_output = None

        for layer in vgg.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name == self.content_layer:
                content_output = layer.output

            layer.trainable = False

        outputs = style_outputs + [content_output]

        model = tf.keras.models.Model(vgg.input, outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer

        input_layer - an instance of tf.Tensor or tf.Variable of
            shape (1, h, w, c) containing the layer output whose
            gram matrix should be calculated

        Returns: a tf.Tensor of shape (1, c, c) containing the gram
            matrix of input_layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape

        F = tf.reshape(input_layer, (h * w, c))
        n = tf.shape(F)[0]

        gram = tf.matmul(F, F, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)

        gram /= tf.cast(n, tf.float32)

        return gram
