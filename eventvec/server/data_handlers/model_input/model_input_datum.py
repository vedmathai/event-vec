class ModelInputDatum:
    """
        A catchall container for all the information about the datum.
        It can include raw information from the dataset, processed information,
        and even post processing information. Having all this information in
        one place makes it easy to use whichever information is important
        and even print it.
    """
    def __init__(self):
        self._from_sentence_encoded = None
        self._to_sentence_encoded = None
        self._from_entity_token_i = None
        self._to_entity_token_i = None
        self._from_original_sentence = None
        self._to_original_sentence = None
        self._from_sentence = None
        self._to_sentence = None
        self._from_decoded_sentence = None
        self._to_decoded_sentence = None
        self._feature_encoding = None
        self._sentence_pair_encoded = None
        self._target = None

        self._from_entity_start_token_i = None
        self._from_entity_end_token_i = None
        self._to_entity_start_token_i = None
        self._to_entity_end_token_i = None
        self._relationship = None
        self._token_order = None
        self._from_tense = None
        self._to_tense = None
        self._from_aspect = None
        self._to_aspect = None
        self._from_pos = None
        self._to_pos = None
        self._is_trainable = False

    def from_sentence_encoded(self):
        return self._from_sentence_encoded

    def to_sentence_encoded(self):
        return self._to_sentence_encoded

    def from_entity_token_i(self):
        return self._from_entity_token_i

    def to_entity_token_i(self):
        return self._to_entity_token_i

    def from_sentence(self):
        return self._from_sentence

    def from_original_sentence(self):
        return self._from_original_sentence

    def from_sentence_string(self):
        return ' '.join(self._from_sentence)

    def to_sentence_string(self):
        return ' '.join(self._to_sentence)

    def from_decoded_sentence(self):
        return self._from_decoded_sentence

    def to_decoded_sentence(self):
        return self._to_decoded_sentence

    def to_sentence(self):
        return self._to_sentence

    def to_original_sentence(self):
        return self._to_original_sentence

    def feature_encoding(self):
        return self._feature_encoding

    def sentence_pair_encoded(self):
        return self._sentence_pair_encoded

    def target(self):
        return self._target

    def from_entity_start_token_i(self):
        return self._from_entity_start_token_i

    def from_entity_end_token_i(self):
        return self._from_entity_end_token_i

    def to_entity_start_token_i(self):
        return self._to_entity_start_token_i

    def to_entity_end_token_i(self):
        return self._to_entity_end_token_i

    def relationship(self):
        return self._relationship

    def token_order(self):
        return self._token_order

    def from_tense(self):
        return self._from_tense

    def to_tense(self):
        return self._to_tense

    def from_aspect(self):
        return self._from_aspect

    def to_aspect(self):
        return self._to_aspect

    def from_pos(self):
        return self._from_pos

    def to_pos(self):
        return self._to_pos

    def is_trainable(self):
        return self._is_trainable

    def set_from_sentence_encoded(self, from_sentence_encoded):
        self._from_sentence_encoded = from_sentence_encoded

    def set_to_sentence_encoded(self, to_sentence_encoded):
        self._to_sentence_encoded = to_sentence_encoded

    def set_to_entity_token_i(self, to_entity_token_i):
        self._to_entity_token_i = to_entity_token_i

    def set_from_entity_token_i(self, from_entity_token_i):
        self._from_entity_token_i = from_entity_token_i

    def set_from_sentence(self, from_sentence):
        self._from_sentence = from_sentence

    def set_to_sentence(self, to_sentence):
        self._to_sentence = to_sentence

    def set_from_original_sentence(self, from_original_sentence):
        self._from_original_sentence = from_original_sentence

    def set_to_original_sentence(self, to_original_sentence):
        self._to_original_sentence = to_original_sentence

    def set_from_decoded_sentence(self, from_decoded_sentence):
        self._from_decoded_sentence = from_decoded_sentence

    def set_to_decoded_sentence(self, to_decoded_sentence):
        self._to_decoded_sentence = to_decoded_sentence

    def set_feature_encoding(self, feature_encoding):
        self._feature_encoding = feature_encoding

    def set_sentence_pair_encoded(self, sentence_pair_encoded):
        self._sentence_pair_encoded = sentence_pair_encoded

    def set_target(self, target):
        self._target = target

    def set_from_entity_start_token_i(self, from_entity_start_token_i):
        self._from_entity_start_token_i = from_entity_start_token_i

    def set_from_entity_end_token_i(self, from_entity_end_token_i):
        self._from_entity_end_token_i = from_entity_end_token_i

    def set_to_entity_start_token_i(self, to_entity_start_token_i):
        self._to_entity_start_token_i = to_entity_start_token_i

    def set_to_entity_end_token_i(self, to_entity_end_token_i):
        self._to_entity_end_token_i = to_entity_end_token_i

    def set_relationship(self, relationship):
        self._relationship = relationship

    def set_token_order(self, token_order):
        self._token_order = token_order

    def set_from_tense(self, from_tense):
        self._from_tense = from_tense

    def set_to_tense(self, to_tense):
        self._to_tense = to_tense

    def set_from_aspect(self, from_aspect):
        self._from_aspect = from_aspect

    def set_to_aspect(self, to_aspect):
        self._to_aspect = to_aspect

    def set_from_pos(self, from_pos):
        self._from_pos = from_pos

    def set_to_pos(self, to_pos):
        self._to_pos = to_pos

    def set_is_trainable(self):
        self._is_trainable = True
