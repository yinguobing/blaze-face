"""Provide training/testing data."""

import cv2
import numpy as np
import tensorflow as tf

from anchors import Anchors, Boxes


class DetectionSample(object):

    def __init__(self, image_file, boxes):
        """Construct an object detection sample.

        Args:
            image: image file path.
            boxes: numpy array of bounding boxes [[x, y, w, h], ...]

        """
        self.image_file = image
        self.boxes = boxes

    def read_image(self, format="BGR"):
        """Read in image as numpy array in format of BGR by defult, else RGB.

        Args:
            format: the channel order, default BGR.

        Returns:
            a numpy array.
        """
        img = cv2.imread(self.image_file)
        if format != "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
