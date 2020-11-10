import cv2
import numpy as np
import tensorflow as tf

from anchors import Anchors


def decode(prediction, threshold):
    # Seprate classifications and boxes transformations.
    regression = prediction[:, :4]
    scores = tf.sigmoid(prediction[:, 5])

    # Decode the detection result.
    anchors = Anchors((0.15, 0.25), [1], (16, 16), (128, 128))
    a_8 = Anchors((0.4, 0.5, 0.6, 0.7, 0.8, 0.9), [1], (8, 8), (128, 128))
    anchors.stack(a_8)
    boxes = anchors.decode(regression).array

    # Select the best match with NMS.
    y1, y2, x1, x2 = tf.split(boxes, 4, axis=1)
    boxes = tf.concat([y1, x1, y2, x2], axis=1)
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, 10, threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

    return selected_boxes


def draw_face_boxes(image, boxes):
    height, width, _ = image.shape

    # Draw the boxes.
    for box in boxes:
        y_min, y_max, x_min, x_max = box / 128
        y_min, y_max = y_min * height, y_max * height
        x_min, x_max = x_min * width,  x_max * width
        cv2.rectangle(image, (int(x_min), int(y_min)),
                      (int(x_max), int(y_max)), (0, 255, 0), 2)
