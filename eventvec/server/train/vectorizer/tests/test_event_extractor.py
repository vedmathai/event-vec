import unittest

import spacy

from eventvec.server.train.vectorizer.event_extractor import EventExtractor
from eventvec.server.train.vectorizer.dep_parser_model import parse_sentence
from eventvec.utils.spacy_utils.utils import get_spacy_doc


class TestEventExtractor(unittest.TestCase):

    def test_event_extraction(self):
        extractor = EventExtractor()
        spacy_doc = get_spacy_doc('the boy had been drawing a bird.')
        for sentence in spacy_doc.sents:
            root, psentence = parse_sentence(sentence)
        verb_node = psentence[4]
        event = extractor.extract(verb_node, psentence)
        self.assertEqual(
            event.to_dict(),
            {
                "subject_nodes": [
                    "boy"
                ],
                "object_nodes": [
                    "a",
                    "bird"
                ],
                "verb_nodes": [
                    "had",
                    "been",
                    "drawing"
                ],
                "root_node": "drawing"
            }
        )

if __name__ == '__main__':
    unittest.main()