import spacy
import coreferee

class SpacyUtils:
    _nlp = None
    _sentence_counter = 0
    def __init__(self):
        if SpacyUtils._nlp is None:
            self.reload_spacy()

    def reload_spacy(self):
        SpacyUtils._nlp = spacy.load('en_core_web_trf')
        SpacyUtils._nlp.add_pipe('coreferee')
        SpacyUtils._sentence_counter = 0

    def get_spacy_doc(self, doc):
        if SpacyUtils._sentence_counter > 10000: # There is some memory leak inside of spacy so reiniting it after some fixed count
            self.reload_spacy()
        SpacyUtils._sentence_counter += 1
        return SpacyUtils._nlp(doc)