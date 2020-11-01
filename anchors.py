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


class Anchors(Boxes):
    """Anchor boxes used in object detection."""

    def __init__(self, sizes, ratios, feature_map_size, image_size):
        """ Construct anchor boxes for object detection.

        Args:
            sizes: a list of anchor boxes size.
            ratios: a list of anchor boxes ratio.
            feature_map_size: a tuple contains the size of the feature map.
            image_size: a tuple specified the size of the input image.
        """
        # Get all the anchors center points.
        nx, ny = feature_map_size
        half_width = image_size[0]/nx/2
        half_height = image_size[1]/ny/2
        x_g, y_g = np.meshgrid(np.linspace(0, image_size[0], nx, endpoint=False),
                               np.linspace(0, image_size[1], ny, endpoint=False))
        x_centers = x_g.flatten() + half_width
        y_centers = y_g.flatten() + half_height

        # Get the anchors' size and stack them all together.
        anchor_boxes = []
        for r in ratios:
            half_sizes_x = [s * image_size[0] * np.sqrt(r) / 2 for s in sizes]
            half_sizes_y = [s * image_size[1] * np.sqrt(r) / 2 for s in sizes]

            for hx, hy in zip(half_sizes_x, half_sizes_y):
                anchor_boxes.append(np.stack([y_centers - hy, y_centers + hy,
                                              x_centers - hx, x_centers + hx], axis=1))
        self.array = np.array(
            anchor_boxes, dtype=np.float32).reshape((-1, 4))
