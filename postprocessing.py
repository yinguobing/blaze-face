import cv2
import numpy as np
import tensorflow as tf

from anchors import Anchors, build_anchors


def decode(prediction, threshold):
    # Seprate classifications and boxes transformations.
    regression = prediction[:, :4]
    scores = tf.sigmoid(prediction[:, 5])

    # Decode the detection result.
    anchors = build_anchors()
    boxes = anchors.decode(regression).array

    # Select the best match with NMS.
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, 100, score_threshold=threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

    return selected_boxes


def draw_face_boxes(image, boxes):
    height, width, _ = image.shape

    # Draw the boxes.
    for box in boxes:
        y_min, x_min, y_max, x_max = box / 128
        y_min, y_max = y_min * height, y_max * height
        x_min, x_max = x_min * width,  x_max * width
        cv2.rectangle(image, (int(x_min), int(y_min)),
                      (int(x_max), int(y_max)), (0, 255, 0), 2)
