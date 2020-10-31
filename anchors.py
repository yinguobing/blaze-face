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

    def areas(self):
        """Return the areas of the boxes.

        Returns:
            areas of the boxes.
        """
        heights = self.boxes[:, 1] - self.boxes[:, 0]
        widths = self.boxes[:, 3] - self.boxes[:, 2]
        areas = np.multiply(heights, widths)

        return areas

    def intersection(self, boxes):
        """Return the intersection areas with other boxes.

        Args:
            boxes: the boxes to intersect with.

        Returns:
            intersection_areas.
        """
        max_y_min = np.maximum(self.boxes[:, 0], boxes.boxes[:, 0].transpose())
        min_y_max = np.minimum(self.boxes[:, 1], boxes.boxes[:, 1].transpose())
        heights = np.maximum(0, min_y_max - max_y_min)

        max_x_min = np.maximum(self.boxes[:, 2], boxes.boxes[:, 2].transpose())
        min_x_max = np.minimum(self.boxes[:, 3], boxes.boxes[:, 3].transpose())
        widths = np.maximum(0, min_x_max - max_x_min)

        areas = np.multiply(widths, heights)

        return areas
