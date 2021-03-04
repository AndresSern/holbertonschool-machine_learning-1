#!/usr/bin/env python3
"""neural style transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """neural style transfer"""
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        ARGS:
            style_image - the image style reference,
            content_image - the image used as a content reference
            beta - the weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or len(
                style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(content_image, np.ndarray) or len(
                content_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if beta < 0:
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or len(
                image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')
        h, w, _ = image.shape
        old = w * h
        if h > w:
            h_new = 512
            w_new = w * h_new / h
        else:
            w_new = 512
            h_new = h * w_new / w

        """Returns a tensor with a length 1 axis inserted at index axis"""
        """image = tf.expand_dims(image, 0)"""
        image = tf.reshape(image, [1, h, w, 3])
        image = tf.image.resize_bicubic(image, (h_new, w_new))
        """rescaled from the range [0, 255] to [0, 1]."""
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        return image

    def load_model(self):
        """ load model"""
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        for layer in vgg.layers:
            layer.trainable = False
        """Load with the custom_object argument.
        custom_objects mapping  : change from maxpool to avgpool
        """
        vgg.save("my_model")
        mapping = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model("my_model", custom_objects=mapping
                                         )
        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = [vgg.get_layer(self.content_layer).output]
        outputs = outputs + content_output

        model = tf.keras.Model([vgg.input], outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """ calculate gram matrices"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(
                input_layer.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    def generate_features(self):
        """
        *extracts the features used to calculate neural style cost

        *gram_style_features - a list of gram matrices calculated from
            the style layer outputs of the style image
        *content_feature - the content layer output of the content image
        """
        content_input = self.content_image * 255
        style_input = self.style_image * 255
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
                content_input)
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
                style_input)
        outputs_content = self.model(preprocessed_content)
        outputs_style = self.model(preprocessed_style)

        num_style_layers = tf.size(self.style_layers)
        style_outputs, content_outputs = (
                outputs_style[:num_style_layers],
                outputs_content[num_style_layers:])

        style_outputs = [self.gram_matrix(
                style_output)for style_output in style_outputs]
        self.gram_style_features = style_outputs
        self.content_feature = content_outputs
