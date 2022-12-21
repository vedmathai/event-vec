from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer


class BookCorpusFeaturizer():
    def __init__(self):
        self._linguistic_featurizer = LinguisticFeaturizer()

    def featurize(self, file_contents, data):
        featurized = self._linguistic_featurizer.featurize_document(file_contents)  # noqa
