import unittest

from eventvec.server.entry_points.vectorizer.document_parser import DocumentParser


class TestDocumentParser(unittest.TestCase):

    def test_document_parser(self):
        parser = DocumentParser()
        document_text = 'The boy was drew a bird after going to school. His teacher taught him before passing him.'
        document = parser.parse(document_text)
        relationships = document.relationships()
        relationships = [i.to_dict() for i in  relationships]
        self.assertEqual(
            relationships,
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
                    "relationship_distribution": {
                        "DURING": 0.26666666666666666,
                        "ENDS": 0.13333333333333333,
                        "BEFORE": 0.6
                    },
                    "relationships": {
                        "DURING": 4,
                        "ENDS": 2,
                        "BEFORE": 9
                    }
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
                    "relationship_distribution": {
                        "DURING": 0.26666666666666666,
                        "ENDS": 0.13333333333333333,
                        "BEFORE": 0.6
                    },
                    "relationships": {
                        "DURING": 4,
                        "ENDS": 2,
                        "BEFORE": 9
                    }
                }
            ]

        )


if __name__ == '__main__':
    unittest.main()