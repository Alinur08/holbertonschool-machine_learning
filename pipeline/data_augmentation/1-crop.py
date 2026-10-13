#!/usr/bin/env python3
""" Crop an image """
import tensorflow as tf


def crop_image(image, size):
    """ Crop of an image randomly """
    return tf.image.random_crop(image, size=size)
