import unittest

import numpy as np

import anchors


class TestBoxesFunctions(unittest.TestCase):

    def setUp(self):
        self.boxes_0 = [2, 1, 2, 1]
        self.boxes_1 = [[1, 3, 1, 3], [7, 6, 7, 6]]
        self.boxes_2 = [[5, 2, 4, 2]]

    def test_init_with_one_box(self):
        b = anchors.Boxes(self.boxes_0).boxes
        self.assertTrue(
            np.allclose(b, np.array(self.boxes_0, dtype=np.float32)))

    def test_init_with_multiple_box(self):
        b = anchors.Boxes(self.boxes_1).boxes
        self.assertTrue(
            np.allclose(b, np.array(self.boxes_1, dtype=np.float32)))


if __name__ == '__main__':
    unittest.main()
