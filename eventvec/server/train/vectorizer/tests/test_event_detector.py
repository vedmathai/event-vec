import unittest

from eventvec.server.train.vectorizer.event_detector import EventDetector
from eventvec.utils.spacy_utils.utils import get_spacy_doc
from eventvec.server.train.vectorizer.dep_parser_model import parse_sentence


class TestEventDetector(unittest.TestCase):

    def test_event_detector(self):
        extractor = EventDetector()
        spacy_doc = get_spacy_doc('the boy had been drawing a bird after learning how to')
        for sentence in spacy_doc.sents:
            root, psentence = parse_sentence(sentence)
        event_nodes = extractor.detect(psentence)
        self.assertEqual(
            [i.orth() for i in event_nodes],
            ['drawing', 'learning']
        )


if __name__ == '__main__':
    unittest.main()