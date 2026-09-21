import unittest

from digit_sum import digit_sum


class DigitSumTest(unittest.TestCase):
    def test_single(self):
        self.assertEqual(digit_sum(7), 7)

    def test_many(self):
        self.assertEqual(digit_sum(123), 6)
