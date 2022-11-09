class FeaturizedToken:
    def __init__(self):
        self._text = None
        self._lemma = None
        self._tense = None
        self._aspect = None

    def text(self):
        return self._text

    def lemma(self):
        return self._lemma

    def aspect(self):
        return self._aspect

    def tense(self):
        return self._tense

    def set_text(self, text):
        self._text = text

    def set_lemma(self, lemma):
        self._lemma = lemma

    def set_aspect(self, aspect):
        self._aspect = aspect

    def set_tense(self, tense):
        self._tense = tense

    @staticmethod
    def from_spacy(token):
        ftoken = FeaturizedToken()
        ftoken._text = token.text
        ftoken._lemma = token.lemma_
        morph_dict = token.morph.to_dict()
        ftoken._tense = morph_dict.get('Tense')
        ftoken._aspect = morph_dict.get('Aspect')
        return ftoken
