from eventvec.server.datamodels.featurized_document_datamodel.featurized_token import FeaturizedToken  # noqa
from eventvec.server.datamodels.featurized_document_datamodel.utils import find_common_parent, traverse_up

class FeaturizedSentence:
    def __init__(self):
        self._tokens = []
        self._root = None

    def add_token(self, token):
        self._tokens.append(token)

    def tokens(self):
        return self._tokens

    def root(self):
        return self._root

    def text(self):
        return ' '.join(i.text() for i in self._tokens)

    def set_root(self, root):
        self._root = root

    @staticmethod
    def from_spacy(sentence, document):
        fsent = FeaturizedSentence()
        i2token = {}
        for token in sentence:
            ft = FeaturizedToken.from_spacy(token, sentence, document)
            i2token[token.i] = ft
            fsent.add_token(ft)
            if token.dep_ == 'ROOT':
                fsent.set_root(ft)
        for parent in sentence:
            for child in parent.children:
                child_token = i2token[child.i]
                parent_token = i2token[parent.i]
                child_token.set_parent(parent_token)
                parent_token.add_child(child_token)
        return fsent

    @staticmethod
    def dependency_path_between_tokens(token_1, token_2):
        if token_1.i_in_sentence() == token_2.i_in_sentence():
            return []
        path_1 = traverse_up(token_1)
        path_2 = traverse_up(token_2)
        common = find_common_parent(path_1, path_2)
        left = path_1[common:][::-1]
        right = path_2[common+1:]
        return left + right