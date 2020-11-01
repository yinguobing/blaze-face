import unittest

import numpy as np

import anchors


class TestBoxesFunctions(unittest.TestCase):

    def setUp(self):
        self.boxes_0 = [1, 2, 1, 2]
        self.boxes_1 = [[1, 3, 1, 3], [6, 7, 6, 7]]
        self.boxes_2 = [[2, 5, 3, 4]]

    def test_init_with_one_box(self):
        b = anchors.Boxes(self.boxes_0).array
        self.assertTrue(
            np.allclose(b, np.array(self.boxes_0, dtype=np.float32)))

    def test_init_with_multiple_box(self):
        b = anchors.Boxes(self.boxes_1).array
        self.assertTrue(
            np.allclose(b, np.array(self.boxes_1, dtype=np.float32)))

    def test_boxes_areas(self):
        areas = anchors.Boxes(self.boxes_1).areas()
        self.assertTrue(
            np.allclose(areas, np.array([4, 1], dtype=np.float32)))

    def test_intersection(self):
        a = anchors.Boxes(self.boxes_0)
        b = anchors.Boxes(self.boxes_1)
        areas = a.intersection(b)
        self.assertTrue(
            np.allclose(areas, np.array([[1, 0]], dtype=np.float32)))

    def test_multiple_intersection(self):
        a = anchors.Boxes([[2, 4, 2, 4], [5, 7, 5, 7], [8, 9, 8, 9]])
        b = anchors.Boxes(self.boxes_1)
        areas = a.intersection(b)
        self.assertTrue(
            np.allclose(areas, np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)))

    def test_iou_no_intersection(self):
        a = anchors.Boxes(self.boxes_1)
        b = anchors.Boxes(self.boxes_2)
        areas = a.iou(b)
        self.assertTrue(
            np.allclose(areas, np.array([[0, 0]], dtype=np.float32)))

    def test_iou_intersection(self):
        a = anchors.Boxes(self.boxes_1)
        b = anchors.Boxes([2, 8, 2, 8])
        areas = a.iou(b)
        self.assertTrue(
            np.allclose(areas, np.array([[1/(4+36-1), 1/36]], dtype=np.float32)))

    def test_anchors_init(self):
        a = anchors.Anchors([0.5, 0.1], [1], (16, 16), (128, 128))
        self.assertTupleEqual(a.array.shape, (512, 4))


if __name__ == '__main__':
    unittest.main()
