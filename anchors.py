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
            self.array = np.array(boxes, dtype=np.float32)
            if self.array.ndim == 1:
                self.array = np.expand_dims(self.array, axis=0)
        else:
            self.array = None

    def areas(self):
        """Return the areas of the boxes.

        Returns:
            areas of the boxes.
        """
        heights = self.array[:, 1] - self.array[:, 0]
        widths = self.array[:, 3] - self.array[:, 2]
        areas = np.multiply(heights, widths)

        return areas

    def intersection(self, boxes):
        """Return the intersection areas with other boxes.

        Args:
            boxes: the boxes to intersect with.

        Returns:
            intersection_areas.
        """
        max_y_min = np.maximum(self.array[:, 0], boxes.array[:, 0].transpose())
        min_y_max = np.minimum(self.array[:, 1], boxes.array[:, 1].transpose())
        heights = np.maximum(0, min_y_max - max_y_min)

        max_x_min = np.maximum(self.array[:, 2], boxes.array[:, 2].transpose())
        min_x_max = np.minimum(self.array[:, 3], boxes.array[:, 3].transpose())
        widths = np.maximum(0, min_x_max - max_x_min)

        areas = np.multiply(widths, heights)

        return areas

    def iou(self, boxes):
        """Return the intersection over union with other boxes.

        Args:
            boxes: the input boxes.

        Returns:
            IoU values.
        """
        intersecion = self.intersection(boxes)
        areas_self = self.areas()
        areas_input = boxes.areas().transpose()
        union = np.add(areas_self, areas_input) - intersecion

        iou = np.divide(intersecion, union)

        return iou
