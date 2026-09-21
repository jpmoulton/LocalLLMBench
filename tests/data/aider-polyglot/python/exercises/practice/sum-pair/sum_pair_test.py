import unittest

from sum_pair import sum_pair


class SumPairTest(unittest.TestCase):
    def test_adds(self):
        self.assertEqual(sum_pair(2, 3), 5)

    def test_negative(self):
        self.assertEqual(sum_pair(-2, 3), 1)
