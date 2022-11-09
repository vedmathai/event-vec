from eventvec.server.model.featurized_document_model.featurized_token import FeaturizedToken  # noqa


class FeaturizedSentence:
    def __init__(self):
        self._tokens = []

    def add_token(self, token):
        self._tokens.append(token)

    def tokens(self):
        return self._tokens

    @staticmethod
    def from_spacy(sentence):
        fsent = FeaturizedSentence()
        for token in sentence:
            fsent.add_token(FeaturizedToken.from_spacy(token))
        return fsent
