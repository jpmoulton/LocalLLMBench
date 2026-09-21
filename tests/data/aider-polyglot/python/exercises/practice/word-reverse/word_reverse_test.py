import unittest

from word_reverse import reverse


class WordReverseTest(unittest.TestCase):
    def test_reverses(self):
        self.assertEqual(reverse("abc"), "cba")
