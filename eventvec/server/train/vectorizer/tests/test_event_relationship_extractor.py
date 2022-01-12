import unittest

from eventvec.server.train.vectorizer.event_extractor import EventExtractor
from eventvec.server.train.vectorizer.event_relationship_extractor import EventRelationshipExtractor
from eventvec.server.train.vectorizer.dep_parser_model import parse_sentence
from eventvec.utils.spacy_utils.utils import get_spacy_doc


class TestEventRelationshipExtractor(unittest.TestCase):

    def test_event_relationship_extraction(self):
        event_extractor = EventExtractor()
        relationship_extractor = EventRelationshipExtractor()
        spacy_doc = get_spacy_doc('the boy drew a bird after learning how to.')
        for sentence in spacy_doc.sents:
            root, psentence = parse_sentence(sentence)
        verb_node_1 = psentence[2]
        verb_node_2 = psentence[6]
        event_1 = event_extractor.extract(verb_node_1, psentence)
        event_2 = event_extractor.extract(verb_node_2, psentence)
        relationships = relationship_extractor.extract(psentence, event_1, event_2)
        self.assertEqual(
            sorted([i.to_dict() for i in relationships], key=lambda x: x['relationship']),
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
                    "relationship": "AFTER",
                    "relationship_score": 28
                },
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
                    "relationship": "BEFORE",
                    "relationship_score": 2
                },
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
                    "relationship": "BEGINS",
                    "relationship_score": 2
                },
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
                    "relationship": "ENDS",
                    "relationship_score": 4
                },
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
                    "relationship": "IS_INCLUDED",
                    "relationship_score": 2
                }
            ]

        )


if __name__ == '__main__':
    unittest.main()