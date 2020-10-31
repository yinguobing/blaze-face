import unittest

import numpy as np

import anchors


class TestBoxesFunctions(unittest.TestCase):

    def setUp(self):
        self.boxes_0 = [1, 2, 1, 2]
        self.boxes_1 = [[1, 3, 1, 3], [6, 7, 6, 7]]
        self.boxes_2 = [[2, 5, 3, 4]]

    def test_init_with_one_box(self):
        b = anchors.Boxes(self.boxes_0).boxes
        self.assertTrue(
            np.allclose(b, np.array(self.boxes_0, dtype=np.float32)))

    def test_init_with_multiple_box(self):
        b = anchors.Boxes(self.boxes_1).boxes
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


if __name__ == '__main__':
    unittest.main()
