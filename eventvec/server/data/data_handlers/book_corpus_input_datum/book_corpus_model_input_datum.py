class BookCorpusModelInputDatum:
    """
        A catchall container for all the information about the datum.
        It can include raw information from the dataset, processed information,
        and even post processing information. Having all this information in
        one place makes it easy to use whichever information is important
        and even print it.
    """
    def __init__(self):
        self._sentence_encoded = None
        self._entity_token_i = None
        self._original_sentence = None
        self._sentence = None
        self._decoded_sentence = None
        self._target = None
        self._tense = None
        self._aspect = None
        self._pos = None
        self._tag = None
        self._target = None
        self._is_trainable = False

    def sentence_encoded(self):
        return self._sentence_encoded

    def entity_token_i(self):
        return self._entity_token_i

    def sentence(self):
        return self._sentence

    def original_sentence(self):
        return self._original_sentence

    def decoded_sentence(self):
        return self._decoded_sentence

    def target(self):
        return self._target

    def tense(self):
        return self._tense

    def aspect(self):
        return self._aspect

    def pos(self):
        return self._pos

    def tag(self):
        return self._tag

    def is_trainable(self):
        return self._is_trainable

    def set_sentence_encoded(self, sentence_encoded):
        self._sentence_encoded = sentence_encoded

    def set_entity_token_i(self, entity_token_i):
        self._entity_token_i = entity_token_i

    def set_sentence(self, sentence):
        self._sentence = sentence

    def set_original_sentence(self, original_sentence):
        self._original_sentence = original_sentence

    def set_decoded_sentence(self, decoded_sentence):
        self._decoded_sentence = decoded_sentence

    def set_target(self, target):
        self._target = target

    def set_tense(self, tense):
        self._tense = tense

    def set_aspect(self, aspect):
        self._aspect = aspect

    def set_pos(self, pos):
        self._pos = pos

    def set_tag(self, tag):
        self._tag = tag

    def set_is_trainable(self, is_trainable):
        self._is_trainable = is_trainable
