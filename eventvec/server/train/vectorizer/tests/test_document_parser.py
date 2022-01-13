import unittest

from eventvec.server.train.vectorizer.document_parser import DocumentParser


class TestDocumentParser(unittest.TestCase):

    def test_document_parser(self):
        parser = DocumentParser()
        document_text = 'The boy was drew a bird after going to school. His teacher taught him before passing him.'
        document = parser.parse(document_text)
        sorted_relationships = sorted([i.to_dict() for i in document.relationships()], key=lambda x: x['relationship'])
        self.assertEqual(
            sorted_relationships,
            [
                {
                    "event_1": {
                        "subject_nodes": [],
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
                            "going"
                        ],
                        "root_node": "going"
                    },
                    "relationship": "AFTER",
                    "relationship_score": 28
                },
                {
                    "event_1": {
                        "subject_nodes": [],
                        "object_nodes": [],
                        "verb_nodes": [
                            "going"
                        ],
                        "root_node": "going"
                    },
                    "event_2": {
                        "subject_nodes": [
                            "teacher"
                        ],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "taught"
                        ],
                        "root_node": "taught"
                    },
                    "relationship": "AFTER",
                    "relationship_score": 28
                },
                {
                    "event_1": {
                        "subject_nodes": [],
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
                            "going"
                        ],
                        "root_node": "going"
                    },
                    "relationship": "BEFORE",
                    "relationship_score": 2
                },
                {
                    "event_1": {
                        "subject_nodes": [],
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
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "passing"
                        ],
                        "root_node": "passing"
                    },
                    "relationship": "BEFORE",
                    "relationship_score": 9
                },
                {
                    "event_1": {
                        "subject_nodes": [],
                        "object_nodes": [],
                        "verb_nodes": [
                            "going"
                        ],
                        "root_node": "going"
                    },
                    "event_2": {
                        "subject_nodes": [
                            "teacher"
                        ],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "taught"
                        ],
                        "root_node": "taught"
                    },
                    "relationship": "BEFORE",
                    "relationship_score": 2
                },
                {
                    "event_1": {
                        "subject_nodes": [
                            "teacher"
                        ],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "taught"
                        ],
                        "root_node": "taught"
                    },
                    "event_2": {
                        "subject_nodes": [],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "passing"
                        ],
                        "root_node": "passing"
                    },
                    "relationship": "BEFORE",
                    "relationship_score": 9
                },
                {
                    "event_1": {
                        "subject_nodes": [],
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
                            "going"
                        ],
                        "root_node": "going"
                    },
                    "relationship": "BEGINS",
                    "relationship_score": 2
                },
                {
                    "event_1": {
                        "subject_nodes": [],
                        "object_nodes": [],
                        "verb_nodes": [
                            "going"
                        ],
                        "root_node": "going"
                    },
                    "event_2": {
                        "subject_nodes": [
                            "teacher"
                        ],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "taught"
                        ],
                        "root_node": "taught"
                    },
                    "relationship": "BEGINS",
                    "relationship_score": 2
                },
                {
                    "event_1": {
                        "subject_nodes": [],
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
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "passing"
                        ],
                        "root_node": "passing"
                    },
                    "relationship": "DURING",
                    "relationship_score": 4
                },
                {
                    "event_1": {
                        "subject_nodes": [
                            "teacher"
                        ],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "taught"
                        ],
                        "root_node": "taught"
                    },
                    "event_2": {
                        "subject_nodes": [],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "passing"
                        ],
                        "root_node": "passing"
                    },
                    "relationship": "DURING",
                    "relationship_score": 4
                },
                {
                    "event_1": {
                        "subject_nodes": [],
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
                            "going"
                        ],
                        "root_node": "going"
                    },
                    "relationship": "ENDS",
                    "relationship_score": 4
                },
                {
                    "event_1": {
                        "subject_nodes": [],
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
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "passing"
                        ],
                        "root_node": "passing"
                    },
                    "relationship": "ENDS",
                    "relationship_score": 2
                },
                {
                    "event_1": {
                        "subject_nodes": [],
                        "object_nodes": [],
                        "verb_nodes": [
                            "going"
                        ],
                        "root_node": "going"
                    },
                    "event_2": {
                        "subject_nodes": [
                            "teacher"
                        ],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "taught"
                        ],
                        "root_node": "taught"
                    },
                    "relationship": "ENDS",
                    "relationship_score": 4
                },
                {
                    "event_1": {
                        "subject_nodes": [
                            "teacher"
                        ],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "taught"
                        ],
                        "root_node": "taught"
                    },
                    "event_2": {
                        "subject_nodes": [],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "passing"
                        ],
                        "root_node": "passing"
                    },
                    "relationship": "ENDS",
                    "relationship_score": 2
                },
                {
                    "event_1": {
                        "subject_nodes": [],
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
                            "going"
                        ],
                        "root_node": "going"
                    },
                    "relationship": "IS_INCLUDED",
                    "relationship_score": 2
                },
                {
                    "event_1": {
                        "subject_nodes": [],
                        "object_nodes": [],
                        "verb_nodes": [
                            "going"
                        ],
                        "root_node": "going"
                    },
                    "event_2": {
                        "subject_nodes": [
                            "teacher"
                        ],
                        "object_nodes": [
                            "boy"
                        ],
                        "verb_nodes": [
                            "taught"
                        ],
                        "root_node": "taught"
                    },
                    "relationship": "IS_INCLUDED",
                    "relationship_score": 2
                }
            ]
        )


if __name__ == '__main__':
    unittest.main()