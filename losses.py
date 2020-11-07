"""Provides custom loss for BlazeFace

This module is modified from the keras implementation of RetinaNet. Refer
to the official code for more details:
https://keras.io/examples/vision/retinanet/
"""

import tensorflow as tf
from tensorflow import keras


def boxes_loss(delta, y_true, y_pred):
    difference = y_true - y_pred
    absolute_difference = tf.abs(difference)
    squared_difference = difference ** 2
    loss = tf.where(tf.less(absolute_difference, delta),
                    0.5 * squared_difference,
                    absolute_difference - 0.5,)
    return tf.reduce_sum(loss, axis=-1)


def cls_loss(alpha, gamma, y_true, y_pred):
    """Implements Focal loss"""
    cross_entropy = keras.losses.BinaryCrossentropy()(y_true, y_pred)
    probs = tf.nn.sigmoid(y_pred)
    alpha = tf.where(tf.equal(y_true, 1.0),
                     alpha, (1.0 - alpha))
    pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
    loss = alpha * tf.pow(1.0 - pt, gamma) * cross_entropy
    return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(keras.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(
            reduction="auto", name="RetinaNetLoss")
        self._alpha = alpha
        self._gamma = gamma
        self._delta = delta

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0),
                                dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0),
                              dtype=tf.float32)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)

        # Loss for classifications.
        cls_labels = y_true[:, :, 4]
        cls_predictions = y_pred[:, :, 4]

        clf_loss = cls_loss(self._alpha, self._gamma,
                            cls_labels, cls_predictions)

        clf_loss = tf.where(tf.equal(ignore_mask, 1.0),
                            tf.zeros_like(ignore_mask), 
                            tf.expand_dims(clf_loss, axis=-1))

        clf_loss = tf.math.divide_no_nan(
            tf.reduce_sum(clf_loss, axis=-1), normalizer)

        # Loss for boxes.
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]

        regr_loss = boxes_loss(self._delta, box_labels, box_predictions)
        regr_loss = tf.where(tf.equal(positive_mask, 1.0), regr_loss, 0.0)
        regr_loss = tf.math.divide_no_nan(
            tf.reduce_sum(regr_loss, axis=-1), normalizer)

        # Total loss.
        loss = clf_loss + regr_loss

        return loss
