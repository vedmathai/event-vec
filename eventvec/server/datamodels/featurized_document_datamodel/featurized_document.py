from eventvec.server.datamodels.featurized_document_datamodel.featurized_sentence import FeaturizedSentence  # noqa
from eventvec.server.datamodels.featurized_document_datamodel.utils import resolve_coreference_pointers


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
            fdoc.add_sentence(FeaturizedSentence.from_spacy(sentence, document))
        fdoc = resolve_coreference_pointers(fdoc)
        return fdoc
