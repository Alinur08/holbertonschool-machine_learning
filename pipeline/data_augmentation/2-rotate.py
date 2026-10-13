#!/usr/bin/env python3
""" Rotate an image """
import tensorflow as tf


def rotate_image(image):
    """ Rotate the image 90 degress counter cw """
    return tf.image.rot90(image)
