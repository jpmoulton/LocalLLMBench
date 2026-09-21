import unittest

from flatten_once import depth, flatten_once


class FlattenOnceTest(unittest.TestCase):
    def test_flattens(self):
        self.assertEqual(flatten_once([[1, 2], [3]]), [1, 2, 3])

    def test_depth(self):
        self.assertEqual(depth([[1]]), 2)
