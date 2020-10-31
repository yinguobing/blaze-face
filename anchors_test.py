import unittest
import anchors
import numpy as np


class TestBoxesFunctions(unittest.TestCase):

    def setUp(self):
        self.boxes_0 = [2, 1, 2, 1]
        self.boxes_1 = [[1, 3, 1, 3], [7, 6, 7, 6]]
        self.boxes_2 = [[5, 2, 4, 2]]

    def InitWithOneBox(self):
        b = anchors.Boxes(self.boxes_0)
        self.assertEqual(b, np.array(self.boxes_0, dtype=np.float32))

    def InitWithMultipleBox(self):
        b = anchors.Boxes(self.boxes_1)
        self.assertEqual(b, np.array(self.boxes_1, dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
