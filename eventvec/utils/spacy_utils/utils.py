import spacy
import coreferee

nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')


def get_spacy_doc(doc):
    return nlp(doc)