import spacy, coreferee



from eventvec.server.datamodels.featurized_document_datamodel.featurized_document import FeaturizedDocument  # noqa
from eventvec.server.datamodels.featurized_document_datamodel.featurized_sentence import FeaturizedSentence  # noqa


nlp = spacy.load('en_core_web_lg')
#nlp.add_pipe('coreferee')


class LinguisticFeaturizer():
    def featurize_document(self, document):
        spacy_doc = nlp(document)
        featurized_document = FeaturizedDocument.from_spacy(spacy_doc)
        return featurized_document

    def featurize_sentence(self, sentence):
        spacy_doc = nlp(sentence)
        featurized_sentence = FeaturizedSentence.from_spacy(
            list(spacy_doc.sents)[0], None
        )
        return featurized_sentence
