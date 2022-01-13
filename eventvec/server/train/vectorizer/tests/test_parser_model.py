import unittest

import spacy

from eventvec.server.train.vectorizer.dep_parser_model import (
    parse_sentence, get_path, follow_down
)
from eventvec.utils.spacy_utils.utils import SpacyUtils

class TestDependencyParserModel(unittest.TestCase):
    def setUp(self) -> None:
        self.spacy_utils = SpacyUtils()

    def test_parsing_tree(self):
        spacy_doc = self.spacy_utils.get_spacy_doc('this is a test.')
        for sentence in spacy_doc.sents:
            root, psentence = parse_sentence(sentence)
        self.assertEqual(root.lemma(), 'be')
        tokens = [i.lemma() for i in psentence]
        self.assertEqual(tokens, ['this', 'be', 'a', 'test', '.'])

    def test_path_between_two_sentences(self):
        spacy_doc = self.spacy_utils.get_spacy_doc('this is a test.')
        for sentence in spacy_doc.sents:
            root, psentence = parse_sentence(sentence)
        path = get_path(psentence, 0, 2)
        tokens = [i.lemma() for i in path]
        self.assertEqual(tokens, ['this', 'be', 'test', 'a'])

    def test_follow_down_orth(self):
        spacy_doc = self.spacy_utils.get_spacy_doc('The boy wrote an exam.')
        for sentence in spacy_doc.sents:
            root, psentence = parse_sentence(sentence)
            node = psentence[2]
            path = follow_down(node, ['nsubj', 'nsubj>det'])
            orth_path = [i.orth() for i in sorted(path, key=lambda x: x.i())]
            self.assertEqual(orth_path, ['The', 'boy'])
            path = follow_down(node, ['dobj', 'dobj>det'])
            orth_path = [i.orth() for i in sorted(path, key=lambda x: x.i())]
            self.assertEqual(orth_path, ['an', 'exam'])

if __name__ == '__main__':
    unittest.main()