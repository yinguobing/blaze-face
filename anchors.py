"""Anchors used in object detection."""
import numpy as np


class Boxes(object):
    """A bunch of boxes."""

    def __init__(self, boxes=[]):
        """Create a bunch of boxes.

        Args:
            boxes: a list of boxes defined by [[y_min, y_max, x_min, x_max]. ...],
                or a numpy array.
        """
        if isinstance(boxes, np.ndarray):
            self.array = boxes
        elif boxes != []:
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
        [y_min1, y_max1, x_min1, x_max1] = np.split(self.array, 4, axis=1)
        [y_min2, y_max2, x_min2, x_max2] = np.split(boxes.array, 4, axis=1)

        max_ymin = np.maximum(y_min1, y_min2.transpose())
        min_ymax = np.minimum(y_max1, y_max2.transpose())
        heights = np.maximum(0, min_ymax - max_ymin)

        max_xmin = np.maximum(x_min1, x_min2.transpose())
        min_xmax = np.minimum(x_max1, x_max2.transpose())
        widths = np.maximum(0, min_xmax - max_xmin)

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
        areas_input = boxes.areas()
        union = np.expand_dims(areas_self, axis=1) + \
            np.expand_dims(areas_input, axis=0) - intersecion

        iou = intersecion / union

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
            half_sizes_y = [s * image_size[1] / np.sqrt(r) / 2 for s in sizes]

            for hx, hy in zip(half_sizes_x, half_sizes_y):
                anchor_boxes.append(np.stack([y_centers - hy, y_centers + hy,
                                              x_centers - hx, x_centers + hx],
                                             axis=1))
        self.array = np.array(anchor_boxes, dtype=np.float32).reshape((-1, 4))

    def match(self, boxes, matched_threshold):
        """Match the anchors with ground truth boxes and return the indices.

        Args:
            boxes: ground truth boxes.
            matched_threshold: value larger than threshold will be considered positive.

        Returns:
            matched anchor boxes' indices.
        """
        # Get the IoUs to match from.
        ious = self.iou(boxes)

        # TODO: If we are lucky, there will always be enough anchor boxes for
        # ground truth boxes. What happens if there are not? *

        # First find all the matched boxes.
        indices_max = np.argmax(ious, axis=0)

        # TODO: What if more than one ground truth boxes are assigned to the
        # same anchor?

        # Then filter out those whose IoU is less than the threshold.
        ious_max = np.amax(ious, axis=0)
        matched_indices = np.where(
            ious_max > matched_threshold, indices_max, -1)
        matched_indices = matched_indices[matched_indices != -1]

        return matched_indices

    def encode(self, boxes, matched_indices):
        """Encode the matched boxes to training target."""

        # Set up the training target.
        training_target = np.zeros_like(self.array)

        # If no boxes are matched.
        if matched_indices.size == 0:
            return training_target

        # Then compute the offset of the anchor boxes.
        matched_anchors = self.array[matched_indices]
        matched_boxes = boxes.array[matched_indices]

        xa, ya, wa, ha = np.split(
            self._get_center_width_height(matched_anchors), 4, axis=1)
        x, y, w, h = np.split(
            self._get_center_width_height(matched_boxes), 4, axis=1)

        EPSILON = 1e-8
        ha += EPSILON
        wa += EPSILON
        h += EPSILON
        w += EPSILON

        ty = (y - ya) / ha * 10
        tx = (x - xa) / wa * 10
        th = np.log(h / ha) * 5
        tw = np.log(w / wa) * 5

        training_target[matched_indices] = np.hstack([tx, ty, tw, th])

        return training_target

    def decode(self, prediction):
        """Convert the prediction back to boxes.

        Args:
            prediction: network output of the regression result.

        Returns:
            decoded boxes.
        """
        xa, ya, wa, ha = np.split(
            self._get_center_width_height(self.array), 4, axis=1)
        tx, ty, tw, th = np.split(prediction, 4, axis=1)

        ty /= 10
        tx /= 10
        th /= 5
        tw /= 5

        h = np.exp(th) * ha
        w = np.exp(tw) * wa
        y = ty * ha + ya
        x = tx * wa + xa

        y_min = y - h / 2
        y_max = y + h / 2
        x_min = x - w / 2
        x_max = x + w / 2

        return Boxes(np.hstack([y_min, y_max, x_min, x_max]))

    @classmethod
    def _get_center_width_height(self, boxes):
        """Return the center points' x, y, boxes height and width."""
        y_min, y_max, x_min, x_max = np.split(boxes, 4, axis=1)
        y = (y_min + y_max) / 2
        x = (x_min + x_max) / 2
        h = y_max - y_min
        w = x_max - x_min

        return np.hstack([x, y, w, h])
