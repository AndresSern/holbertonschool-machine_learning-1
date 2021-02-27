#!/usr/bin/env python3
""" v3 algorithm to perform object detection"""
import tensorflow.keras as K
import numpy as np


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
        with open(classes_path, 'r') as f:
            self.class_names = [line[0:-1] for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, z):
        """ sigmoid function"""
        return (1 / (1 + np.exp(-z)))

    def process_outputs(self, outputs, image_size):
        """
        ARGS:
        outputs is a list of numpy.ndarrays containing :
        (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
        *grid_height & grid_width => the height and width of the grid
        *anchor_boxes => the number of anchor boxes used
        *4 => (t_x, t_y, t_w, t_h)
        *1 => box_confidence
        *classes => class probabilities for all classes
        """
        i = 0
        boxes = []
        image_height, image_width = image_size
        box_confidence = []
        box_class_probs = []
        for out in outputs:
            boxes.append(out[:, :, :, 0:4])
            box_confidence.append(self.sigmoid(out[:, :, :, 4:5]))
            box_class_probs.append(self.sigmoid(out[:, :, :, 5:]))

            t_x = boxes[i][:, :, :, 0]
            t_y = boxes[i][:, :, :, 1]
            t_w = boxes[i][:, :, :, 2]
            t_h = boxes[i][:, :, :, 3]

            grid_height, grid_width, anchor_boxes, _ = out.shape
            """ grid idices"""
            cx = np.indices((grid_height, grid_width, anchor_boxes))[1]
            cy = np.indices((grid_height, grid_width, anchor_boxes))[0]
            """ localisation in grid """
            bx = (self.sigmoid(t_x) + cx)
            by = (self.sigmoid(t_y) + cy)
            """ localisation in images of shape [13x13,26x26,525,52]"""
            bx = bx / grid_width
            by = by / grid_height

            """ from list of anchors
            anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
            """
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            """ localisation in image of the model input"""
            input_w = self.model.input.shape[1].value
            input_h = self.model.input.shape[2].value

            bw = pw * np.exp(t_w) / input_w
            bh = ph * np.exp(t_h) / input_h
            """ rescale coordinates to original dimensions"""
            x1 = (bx - bw / 2) * image_width
            x2 = (bx - bw / 2 + bw) * image_width
            y1 = (by - bh / 2) * image_height
            y2 = (by - bh / 2 + bh) * image_height

            boxes[i][:, :, :, 0] = x1
            boxes[i][:, :, :, 1] = y1
            boxes[i][:, :, :, 2] = x2
            boxes[i][:, :, :, 3] = y2
            i = i + 1
        return boxes, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        __boxes: a list of numpy.ndarrays of shape:
            (grid_height, grid_width, anchor_boxes, 4)
        __*box_confidences: a list of numpy.ndarrays of shape:
            (grid_height, grid_width, anchor_boxes, 1)
        __*box_class_probs: a list of numpy.ndarrays of shape:
            (grid_height, grid_width, anchor_boxes, classes)

        Returns a tuple of (filtered_boxes, box_classes, box_scores):
        """
        box_scores = []
        box_classes = []
        box_class_scores = []
        filtering_mask = []
        scores = []
        boxess = []
        classes = []

        for i in range(len(boxes)):
            """ Compute box scores"""
            box_scores.append(box_confidences[i] * box_class_probs[i])
            """ Find the box_classes thanks to the max box_scores,
            keep track of the corresponding score
            """
            box_classes.append(np.argmax(box_scores[i], axis=3))
            box_class_scores.append(np.max(box_scores[i], axis=3))
            """filtering mask based on "box_class_scores" by using threshold"""
            filtering_mask.append(box_class_scores[i] >= self.class_t)

        """ filter list by mask """
        scores += (d[s] for d, s in zip(box_class_scores, filtering_mask))
        boxess += (d[s] for d, s in zip(boxes, filtering_mask))

        classes += (d[s].flatten() for d, s in zip(box_classes,
                                                   filtering_mask))

        """ Flattening a list of NumPy arrays"""
        classes = np.concatenate(classes).ravel()
        boxess = np.concatenate(boxess)
        scores = np.concatenate(scores).ravel()
        return (boxess,  classes, scores)
