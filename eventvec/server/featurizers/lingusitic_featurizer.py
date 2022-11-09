import spacy


from eventvec.server.model.featurized_document_model.featurized_document import FeaturizedDocument  # noqa
from eventvec.server.model.featurized_document_model.featurized_sentence import FeaturizedSentence  # noqa


nlp = spacy.load('en_core_web_sm')


class LinguisticFeaturizer():
    def featurize_document(self, document):
        spacy_doc = nlp(document)
        featurized_document = FeaturizedDocument.from_spacy(spacy_doc)
        return featurized_document

    def featurize_sentence(self, sentence):
        spacy_doc = nlp(sentence)
        featurized_sentence = FeaturizedSentence.from_spacy(spacy_doc.sents[0])
        return featurized_sentence


if __name__ == '__main__':
    text = ("When Sebastian Thrun started working on self-driving cars at "
            "Google in 2007, few people outside of the company took him "
            "seriously.")
    lf = LinguisticFeaturizer()
    fd = lf.featurize_document(text)
    sentences = fd.sentences()
    for sent in sentences:
        for token in sent.tokens():
            print(token.text())
