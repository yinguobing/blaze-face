"""Anchors used in object detection."""
import numpy as np


class Boxes(object):
    """A bunch of boxes."""

    def __init__(self, boxes=[]):
        """Create a bunch of boxes.

        Args:
            boxes: a list of boxes defined by [[y_min, y_max, x_min, x_max]. ...]
        """
        if boxes != []:
            self.boxes = np.array(boxes, dtype=np.float32)
            if self.boxes.ndim == 1:
                self.boxes = np.expand_dims(self.boxes, axis=0)
        else:
            self.boxes = None
