import unittest

import numpy as np

import anchors
from visualization import Visualizer


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
            np.allclose(areas, np.array([[1/(4+36-1)], [1/36]], dtype=np.float32)))

    def test_anchors_init(self):
        s = [0.3, 0.2]
        r = [1, 2]
        m, n = featmap_size = (2, 2)
        a = anchors.Anchors(s, r, featmap_size, (128, 128))
        self.assertTupleEqual(a.array.shape, (m*n*len(s)*len(r), 4))

    def test_matching(self):
        s = [0.5, 0.1]
        r = [1]
        m, n = featmap_size = (16, 16)
        a = anchors.Anchors(s, r, featmap_size, (128, 128))
        gt = anchors.Boxes([[32, 64, 32, 64], [54, 66, 50, 60]])
        t = a.match(gt, matched_threshold=0)
        self.assertTrue(np.allclose(t, [68, 374]))

    def test_matching_failed(self):
        s = [0.1]
        r = [1]
        m, n = featmap_size = (2, 2)
        a = anchors.Anchors(s, r, featmap_size, (128, 128))
        gt = anchors.Boxes([0, 5, 0, 5])
        t = a.match(gt, matched_threshold=0)
        self.assertTrue(np.allclose(t, []))

    def test_get_center_width_height(self):
        b = anchors.Boxes([[32, 60, 32, 60],
                           [26, 38, 90, 102],
                           [80, 88, 55, 68],
                           [70, 122, 70, 122]])
        r = anchors.Anchors._get_center_width_height(b.array)
        self.assertTrue(
            np.allclose(r, np.array([[46, 46, 28, 28],
                                     [96, 32, 12, 12],
                                     [61.5, 84, 13, 8],
                                     [96, 96, 52, 52]], dtype=np.float32)))

    def test_anchor_transformation(self):
        a = anchors.Anchors([0.3], [1], (2, 2), (128, 128))
        gt = anchors.Boxes([[32, 60, 32, 60],
                            [26, 38, 90, 102],
                            [80, 88, 55, 68],
                            [70, 122, 70, 122]])
        i = a.match(gt, 0)
        t = a.encode(gt, i)
        self.assertTrue(
            np.allclose(t, np.array([[3.645833, 3.645833, -1.579265, -1.579265],
                                     [0.,  0., -5.8157535, -5.815754],
                                     [0.,  0., 0., 0.],
                                     [0.,  0.,  1.5159321, 1.5159321]])))


if __name__ == '__main__':
    unittest.main()
