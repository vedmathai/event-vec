import unittest

from eventvec.server.entry_points.vectorizer.event_extractor import EventExtractor
from eventvec.server.entry_points.vectorizer.event_relationship_extractor import EventRelationshipExtractor
from eventvec.server.entry_points.vectorizer.dep_parser_model import parse_sentence
from eventvec.utils.spacy_utils.utils import SpacyUtils


class TestEventRelationshipExtractor(unittest.TestCase):

    def test_event_relationship_extraction(self):
        event_extractor = EventExtractor()
        relationship_extractor = EventRelationshipExtractor()
        spacy_utils = SpacyUtils()
        spacy_doc = spacy_utils.get_spacy_doc('the boy drew a bird after learning how to.')
        for sentence in spacy_doc.sents:
            root, psentence = parse_sentence(sentence)
        verb_node_1 = psentence[2]
        verb_node_2 = psentence[6]
        event_1 = event_extractor.extract(verb_node_1, psentence)
        event_2 = event_extractor.extract(verb_node_2, psentence)
        relationships = relationship_extractor.extract(psentence, event_1, event_2)
        self.assertEqual(
            [i.to_dict() for i in relationships],
            [
                {
                    "event_1": {
                        "subject_nodes": [
                            "boy"
                        ],
                        "object_nodes": [
                            "a",
                            "bird"
                        ],
                        "verb_nodes": [
                            "drew"
                        ],
                        "root_node": "drew"
                    },
                    "event_2": {
                        "subject_nodes": [],
                        "object_nodes": [],
                        "verb_nodes": [
                            "learning"
                        ],
                        "root_node": "learning"
                    },
                    "relationship_distribution": {
                        "BEGINS": 0.05263157894736842,
                        "ENDS": 0.10526315789473684,
                        "AFTER": 0.7368421052631579,
                        "IS_INCLUDED": 0.05263157894736842,
                        "BEFORE": 0.05263157894736842
                    },
                    "relationships": {
                        "BEGINS": 2,
                        "ENDS": 4,
                        "AFTER": 28,
                        "IS_INCLUDED": 2,
                        "BEFORE": 2
                    }
                }
            ]
        )


if __name__ == '__main__':
    unittest.main()