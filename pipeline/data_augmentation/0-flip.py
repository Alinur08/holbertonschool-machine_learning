#!/usr/bin/env python3
""" Flip the image"""
import tensorflow as tf


def flip_image(image):
    """ Flip image in tf """
    return tf.image.flip_left_right(image)
