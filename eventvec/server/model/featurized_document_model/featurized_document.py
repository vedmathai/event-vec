from eventvec.server.model.featurized_document_model.featurized_sentence import FeaturizedSentence  # noqa


class FeaturizedDocument:
    def __init__(self):
        self._sentences = []

    def add_sentence(self, sentence):
        self._sentences.append(sentence)

    def sentences(self):
        return self._sentences

    @staticmethod
    def from_spacy(document):
        fdoc = FeaturizedDocument()
        for sentence in document.sents:
            fdoc.add_sentence(FeaturizedSentence.from_spacy(sentence))
        return fdoc
