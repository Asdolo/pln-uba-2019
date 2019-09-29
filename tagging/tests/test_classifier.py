# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from tagging.classifier import feature_dict


class TestFeatureDict(TestCase):

    def test_feature_dict(self):
        sent = 'El gato come pescado .'.split()

        fdict = {
            'word_lowercase': 'el',    # lower
            'is_uppercase': False,  # isupper
            'is_capitalized': True,   # istitle
            'is_digit': False,  # isdigit
            'previous_word_lowercase': '<s>',
            'previous_word_is_uppercase': False,
            'previous_word_is_capitalized': False,
            'previous_word_is_digit': False,
            'next_word_lowercase': 'gato',
            'next_word_is_uppercase': False,
            'next_word_is_capitalized': False,
            'next_word_is_digit': False,
        }

        self.assertEqual(feature_dict(sent, 0), fdict)
