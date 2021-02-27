#!/usr/bin/env python3
""" v3 algorithm to perform object detection"""
import tensorflow.keras as K


class Yolo:
    """
    *model_path the path to where a Darknet model is stored
    *classes_path is the path to where the list of class names
    *class_t is a float representing the box score threshold for
    the initial filtering step
    *nms_t is a float representing the IOU threshold for non-max suppression
    *anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2):
        outputs is the number of outputs (predictions) made by the Darknet
        anchor_boxes is the number of anchor boxes used for each prediction
        2 => [anchor_box_width, anchor_box_height]
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as fl:
            self.class_names = [line[0:-1] for line in fl.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
