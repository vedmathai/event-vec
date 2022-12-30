from eventvec.server.model.featurized_document_model.featurized_token import FeaturizedToken  # noqa


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
    def from_spacy(sentence):
        fsent = FeaturizedSentence()
        i2token = {}
        for token in sentence:
            ft = FeaturizedToken.from_spacy(token, sentence)
            i2token[token.i] = ft
            fsent.add_token(ft)
            if token.dep_ == 'ROOT':
                fsent.set_root(token)
        for parent in sentence:
            for child in parent.children:
                child_token = i2token[child.i]
                parent_token = i2token[parent.i]
                child_token.set_parent(parent_token)
                parent_token.add_child(child_token)
        return fsent
