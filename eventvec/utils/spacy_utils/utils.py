import spacy
nlp = spacy.load("en_core_web_trf")


def get_spacy_doc(doc):
    return nlp(doc)