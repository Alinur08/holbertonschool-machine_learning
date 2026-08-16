#!/usr/bin/env python3
"""Neural Style Transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer"""

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor

        style_image - the image used as a style reference,
            stored as a numpy.ndarray
        content_image - the image used as a content reference,
            stored as a numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
                style_image.shape[-1] != 3 or style_image.ndim != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or \
                content_image.shape[-1] != 3 or content_image.ndim != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(alpha, (int, float))) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (not isinstance(beta, (int, float))) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.model = self.load_model()
        self.generate_features()

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
                image.shape[-1] != 3 or image.ndim != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        image = image[tf.newaxis, :]
        image = tf.image.resize(
            image, size=(h_new, w_new), method='bicubic'
        )
        image = tf.clip_by_value(image / 255, 0, 1)

        return image

    def load_model(self):
        """
        Creates the model used to calculate cost from the VGG19
        Keras base model

        Returns: the model
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        vgg.save("vgg_base.h5")

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model(
            "vgg_base.h5", custom_objects=custom_objects)

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
        return model

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
        if not (isinstance(input_layer, (tf.Tensor, tf.Variable)) and
                len(input_layer.shape) == 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)

        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        Sets the public instance attributes:
            gram_style_features - a list of gram matrices calculated
                from the style layer outputs of the style image
            content_feature - the content layer output of the
                content image
        """
        vgg19 = tf.keras.applications.vgg19
        preprocess_style = vgg19.preprocess_input(self.style_image * 255)
        preprocess_content = vgg19.preprocess_input(self.content_image * 255)

        style_features = self.model(preprocess_style)[:-1]
        content_feature = self.model(preprocess_content)[-1]

        gram_style_features = [
            self.gram_matrix(feature) for feature in style_features]

        self.gram_style_features = gram_style_features
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer

        style_output - a tf.Tensor of shape (1, h, w, c) containing
            the layer style output of the generated image
        gram_target - a tf.Tensor of shape (1, c, c) containing the
            gram matrix of the target style output for that layer

        Returns: the layer's style cost
        """
        if not (isinstance(style_output, (tf.Tensor, tf.Variable)) and
                len(style_output.shape) == 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = int(style_output.shape[-1])
        if not (isinstance(gram_target, (tf.Tensor, tf.Variable)) and
                gram_target.shape == (1, c, c)):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for generated image

        style_outputs - a list of tf.Tensor style outputs for the
            generated image

        Returns: the style cost
        """
        length = len(self.style_layers)
        if not (isinstance(style_outputs, list) and
                len(style_outputs) == length):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length))

        weight = 1.0 / length
        J_style = 0
        for i in range(length):
            J_style += weight * self.layer_style_cost(
                style_outputs[i], self.gram_style_features[i])

        return J_style

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image

        content_output - a tf.Tensor containing the content output
            for the generated image

        Returns: the content cost
        """
        s = self.content_feature.shape
        if not (isinstance(content_output, (tf.Tensor, tf.Variable)) and
                content_output.shape == s):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s))

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image

        generated_image - a tf.Tensor of shape (1, nh, nw, 3)
            containing the generated image

        Returns: (J, J_content, J_style)
            J is the total cost
            J_content is the content cost
            J_style is the style cost
        """
        s = self.content_image.shape
        if not (isinstance(generated_image, (tf.Tensor, tf.Variable)) and
                generated_image.shape == s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        vgg19 = tf.keras.applications.vgg19
        preprocess = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(preprocess)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the generated image

        generated_image - a tf.Tensor of shape (1, nh, nw, 3)
            containing the generated image

        Returns: gradients, J_total, J_content, J_style
            gradients is a tf.Tensor containing the gradients for
                the generated image
            J_total is the total cost for the generated image
            J_content is the content cost for the generated image
            J_style is the style cost for the generated image
        """
        s = self.content_image.shape
        if not (isinstance(generated_image, (tf.Tensor, tf.Variable)) and
                generated_image.shape == s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        with tf.GradientTape() as tape:
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style
