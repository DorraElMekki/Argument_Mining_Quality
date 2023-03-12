import unittest

from ..main import train_test_bert_with_categorical_features


class MainTest(unittest.TestCase):
    def test_e2e_main(self):
        train_test_bert_with_categorical_features()
