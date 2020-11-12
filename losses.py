"""Provides custom loss for BlazeFace

These functions are copied from TensorFlow official models.
https://github.com/tensorflow/models/blob/master/official/vision/detection/modeling/losses.py
"""

import tensorflow as tf
from tensorflow import keras


def focal_loss(logits, targets, alpha, gamma, normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size
            [batch, height_in, width_in, num_predictions].
        targets: A float32 tensor of size
            [batch, height_in, width_in, num_predictions].
        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.
        gamma: A float32 scalar modulating loss from hard and easy examples.
        normalizer: A float32 scalar normalizes the total loss from all examples.

    Returns:
        loss: A float32 Tensor of size [batch, height_in, width_in, num_predictions]
            representing normalized loss on the prediction map.
    """
    positive_label_mask = tf.math.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    # Below are comments/derivations for computing modulator.
    # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    # for positive samples and 1 - sigmoid(x) for negative examples.
    #
    # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    # computation. For r > 0, it puts more weights on hard examples, and less
    # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    # its back-propagation is not stable when r < 1. The implementation here
    # resolves the issue.
    #
    # For positive samples (labels being 1),
    #    (1 - p_t)^r
    #  = (1 - sigmoid(x))^r
    #  = (1 - (1 / (1 + exp(-x))))^r
    #  = (exp(-x) / (1 + exp(-x)))^r
    #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
    #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
    #  = exp(- r * x - r * log(1 + exp(-x)))
    #
    # For negative samples (labels being 0),
    #    (1 - p_t)^r
    #  = (sigmoid(x))^r
    #  = (1 / (1 + exp(-x)))^r
    #  = exp(log((1 / (1 + exp(-x)))^r))
    #  = exp(-r * log(1 + exp(-x)))
    #
    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    neg_logits = -1.0 * logits
    modulator = tf.math.exp(gamma * targets * neg_logits -
                            gamma * tf.math.log1p(tf.math.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)
    weighted_loss /= normalizer
    return weighted_loss


class RetinanetClassLoss(object):
    """RetinaNet class loss."""

    def __init__(self, focal_loss_alpha, focal_loss_gamma, num_classes):
        self._num_classes = num_classes
        self._focal_loss_alpha = focal_loss_alpha
        self._focal_loss_gamma = focal_loss_gamma

    def __call__(self, cls_outputs, labels, num_positives):
        """Computes total detection loss.

        Computes total detection loss including box and class loss from all levels.

        Args:
            cls_outputs: classification result.
            labels: the dictionary that returned from dataloader that includes
             _   class groundtruth targets.
            num_positives: number of positive examples in the minibatch.

        Returns:
            an integar tensor representing total class loss.
        """
        # Sums all positives in a batch for normalization and avoids zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = tf.reduce_sum(input_tensor=num_positives) + 1.0

        cls_losses = self.class_loss(cls_outputs, labels, num_positives_sum)

        return cls_losses

    def class_loss(self, cls_outputs, cls_targets, num_positives,
                   ignore_label=-2):
        """Computes RetinaNet classification loss."""
        # Onehot encoding for classification labels.
        cls_targets_one_hot = tf.one_hot(
            tf.cast(cls_targets, tf.int32), self._num_classes)
        loss = focal_loss(tf.cast(cls_outputs, dtype=tf.float32),
                          tf.cast(cls_targets_one_hot, dtype=tf.float32),
                          self._focal_loss_alpha,
                          self._focal_loss_gamma,
                          num_positives)

        ignore_loss = tf.where(tf.equal(cls_targets, ignore_label),
                               tf.zeros_like(cls_targets, dtype=tf.float32),
                               tf.ones_like(cls_targets, dtype=tf.float32))
        ignore_loss = tf.expand_dims(ignore_loss, -1)
        ignore_loss = tf.tile(ignore_loss, [1, 1, self._num_classes])
        ignore_loss = tf.reshape(ignore_loss, tf.shape(input=loss))
        return tf.reduce_sum(input_tensor=ignore_loss * loss)


class RetinanetBoxLoss(object):
    """RetinaNet box loss."""

    def __init__(self, huber_loss_delta):
        self._huber_loss = tf.keras.losses.Huber(
            delta=huber_loss_delta,
            reduction=keras.losses.Reduction.SUM)

    def __call__(self, box_outputs, labels, num_positives):
        """Computes box detection loss.
        Computes total detection loss including box and class loss from all levels.

        Args:
            box_outputs: an OrderDict with keys representing levels and values
                representing box regression targets in [batch_size, height, width,
                num_anchors * 4].
            labels: the dictionary that returned from dataloader that includes
                box groundtruth targets.
            num_positives: number of positive examples in the minibatch.

        Returns:
            an integer tensor representing total box regression loss.
        """
        # Sums all positives in a batch for normalization and avoids zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = tf.reduce_sum(input_tensor=num_positives) + 1.0

        box_losses = self.box_loss(box_outputs, labels, num_positives_sum)

        # Sums all losses to total loss.
        return box_losses

    def box_loss(self, box_outputs, box_targets, num_positives):
        """Computes RetinaNet box regression loss."""
        # The delta is typically around the mean value of regression target.
        # for instances, the regression targets of 512x512 input with 6 anchors on
        # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
        normalizer = num_positives * 4.0
        mask = tf.cast(tf.not_equal(box_targets, 0.0), dtype=tf.float32)
        box_targets = tf.expand_dims(box_targets, axis=-1)
        box_outputs = tf.expand_dims(box_outputs, axis=-1)
        box_loss = self._huber_loss(box_targets, box_outputs,
                                    sample_weight=mask)
        box_loss /= normalizer
        return box_loss


class BlazeLoss(keras.losses.Loss):

    def __init__(self):
        super().__init__()
        self._cls_loss_fn = RetinanetClassLoss(0.25, 1.5, 2)
        self._box_loss_fn = RetinanetBoxLoss(0.1)
        self._box_loss_weight = 50

    def call(self, labels, outputs):

        cls_outputs = outputs[:, :, 4:]
        cls_labels = labels[:, :, 4]
        num_positives = tf.reduce_sum(
            input_tensor=tf.cast(tf.greater(cls_labels, -1), tf.float32))
        cls_loss = self._cls_loss_fn(cls_outputs, cls_labels, num_positives)

        box_outputs = outputs[:, :, :4]
        box_labels = labels[:, :, :4]
        reg_loss = self._box_loss_fn(box_outputs, box_labels, num_positives)

        total_loss = cls_loss + self._box_loss_weight * reg_loss

        return total_loss
