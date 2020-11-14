"""Anchors used in object detection."""
import numpy as np


class Boxes(object):
    """A bunch of boxes."""

    def __init__(self, boxes=[]):
        """Create a bunch of boxes.

        Args:
            boxes: a list of boxes defined by [[y_min, x_min, y_max, x_max]. ...],
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

    def __len__(self):
        return self.array.shape[0]

    def areas(self):
        """Return the areas of the boxes.

        Returns:
            areas of the boxes.
        """
        heights = self.array[:, 2] - self.array[:, 0]
        widths = self.array[:, 3] - self.array[:, 1]
        areas = np.multiply(heights, widths)

        return areas

    def intersection(self, boxes):
        """Return the intersection areas with other boxes.

        Args:
            boxes: the boxes to intersect with.

        Returns:
            intersection_areas.
        """
        [y_min1, x_min1, y_max1, x_max1] = np.split(self.array, 4, axis=1)
        [y_min2, x_min2, y_max2, x_max2] = np.split(boxes.array, 4, axis=1)

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

    def stack(self, boxes):
        """Stack two set of boxes together.

        Args:
            boxes: boxes to be stacked.
        """
        self.array = np.vstack([self.array, boxes.array])


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
        img_width, img_height = image_size

        shift_x = img_width / nx / 2
        shift_y = img_height / ny / 2
        centers_x, centers_y = np.meshgrid(
            np.linspace(0, img_width, nx, endpoint=False),
            np.linspace(0, img_height, ny, endpoint=False))

        centers_x = centers_x.flatten() + shift_x
        centers_y = centers_y.flatten() + shift_y

        # Get the anchors' size and stack them all together.
        anchor_boxes = []
        for r in ratios:
            anchor_sizes_x = [img_width * np.sqrt(r) * s / 2 for s in sizes]
            anchor_sizes_y = [img_height / np.sqrt(r) * s / 2 for s in sizes]

            for ax, ay in zip(anchor_sizes_x, anchor_sizes_y):
                anchor_boxes.append(np.stack([centers_y - ay, centers_x - ax,
                                              centers_y + ay, centers_x + ax],
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
        # ground truth boxes. What happens if there are not?

        # First find all the anchor boxes with max IoU.
        max_anchor_indices = np.argmax(ious, axis=0)

        # TODO: What if more than one ground truth boxes are assigned to the
        # same anchor?

        # Then filter out those whose IoU is less than the threshold.
        ious_max = np.amax(ious, axis=0)
        matched_anchor_indices = np.where(
            ious_max > matched_threshold, max_anchor_indices, -1)

        # Combine the anchor indices and box indices.
        matched_indices = np.vstack(
            [matched_anchor_indices, np.arange(len(boxes))])
        matched_indices = matched_indices[
            :, matched_anchor_indices != -1].transpose()

        return matched_indices

    def encode(self, boxes, matched_indices):
        """Encode the matched boxes to training target."""

        # Set up the training target.
        training_target = np.zeros_like(self.array)

        # If no boxes are matched.
        if matched_indices.size == 0:
            return training_target

        # Then compute the offset of the anchor boxes.
        matched_anchors = self.array[matched_indices[:, 0]]
        matched_boxes = boxes.array[matched_indices[:, 1]]

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

        training_target[matched_indices[:, 0]] = np.hstack([tx, ty, tw, th])

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

        return Boxes(np.hstack([y_min, x_min, y_max, x_max]))

    @classmethod
    def _get_center_width_height(self, boxes):
        """Return the center points' x, y, boxes height and width."""
        y_min, x_min, y_max, x_max = np.split(boxes, 4, axis=1)
        y = (y_min + y_max) / 2
        x = (x_min + x_max) / 2
        h = y_max - y_min
        w = x_max - x_min

        return np.hstack([x, y, w, h])


def build_anchors():
    """Build the anchor boxes for BlazeFace."""
    anchor_arrays = AnchorBox().get_anchors(128, 128)
    anchors = Anchors([1], [1], (2, 2), (128, 128))
    anchors.array = anchor_arrays
    return anchors


class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = np.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = np.reshape(
                    np.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(np.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = np.arange(feature_width, dtype=np.float32) + 0.5
        ry = np.arange(feature_height, dtype=np.float32) + 0.5
        centers = np.stack(np.meshgrid(rx, ry), axis=-1) * \
            self._strides[level - 3]
        centers = np.expand_dims(centers, axis=-2)
        centers = np.tile(centers, [1, 1, self._num_anchors, 1])
        dims = np.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = np.concatenate([centers, dims], axis=-1)
        return np.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                np.ceil(image_height / 2 ** i).astype(np.int32),
                np.ceil(image_width / 2 ** i).astype(np.int32),
                i,
            )
            for i in range(3, 8)
        ]
        return np.concatenate(anchors, axis=0)
