import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BertInput:
    def __init__(self):
        self._from_input_ids = None
        self._from_attention_mask = None
        self._from_token_type_ids = None
        self._to_input_ids = None
        self._to_attention_mask = None
        self._to_token_type_ids = None
        self._whole_sentence_input_ids = None
        self._whole_sentence_attention_masks = None
        self._whole_sentence_token_type_ids = None
        self._whole_token_i = None
        self._feature_encoding = None
        self._from_token_i = None
        self._to_token_i = None

    def from_input_ids(self):
        return self._from_input_ids

    def from_attention_mask(self):
        return self._from_attention_mask

    def from_token_type_ids(self):
        return self._from_token_type_ids

    def to_input_ids(self):
        return self._to_input_ids

    def to_attention_mask(self):
        return self._to_attention_mask

    def to_token_type_ids(self):
        return self._to_token_type_ids

    def whole_sentence_input_ids(self):
        return self._whole_sentence_input_ids

    def whole_sentence_attention_masks(self):
        return self._whole_sentence_attention_masks

    def whole_sentence_token_type_ids(self):
        return self._whole_sentence_token_type_ids

    def whole_token_i(self):
        return self._whole_token_i

    def feature_encoding(self):
        return self._feature_encoding

    def from_token_i(self):
        return self._from_token_i

    def to_token_i(self):
        return self._to_token_i

    @staticmethod
    def from_input_datum(datum):
        bi = BertInput()
        from_sentence_inputs = datum.from_sentence_encoded()
        from_input_ids = from_sentence_inputs['input_ids']
        from_attention_mask = from_sentence_inputs['attention_mask']
        from_token_type_ids = from_sentence_inputs['token_type_ids']
        to_sentence_inputs = datum.to_sentence_encoded()
        to_input_ids = to_sentence_inputs['input_ids']
        to_attention_mask = to_sentence_inputs['attention_mask']
        to_token_type_ids = to_sentence_inputs['token_type_ids']
        whole_sentence = datum.sentence_pair_encoded()
        whole_sentence_input_ids = whole_sentence['input_ids']
        whole_sentence_attention_masks = whole_sentence['attention_mask']
        whole_sentence_token_type_ids = whole_sentence['token_type_ids']

        from_attention_mask = from_attention_mask.to(device)
        from_token_type_ids = from_token_type_ids.to(device)
        from_input_ids = from_input_ids.squeeze(1).to(device)

        to_attention_mask = to_attention_mask.to(device)
        to_token_type_ids = to_token_type_ids.to(device)
        to_input_ids = to_input_ids.squeeze(1).to(device)

        whole_sentence_attention_masks = whole_sentence_attention_masks.to(device)  # noqa
        whole_sentence_token_type_ids = whole_sentence_token_type_ids.to(device)  # noqa
        whole_sentence_input_ids = whole_sentence_input_ids.squeeze(1).to(device)  # noqa

        feature_encoding = torch.from_numpy(np.array(datum.feature_encoding()))

        bi._from_input_ids = from_input_ids
        bi._from_attention_mask = from_attention_mask
        bi._from_token_type_ids = from_token_type_ids
        bi._to_input_ids = to_input_ids
        bi._to_attention_mask = to_attention_mask
        bi._to_token_type_ids = to_token_type_ids
        bi._whole_sentence_input_ids = whole_sentence_input_ids
        bi._whole_sentence_attention_masks = whole_sentence_attention_masks
        bi._whole_sentence_token_type_ids = whole_sentence_token_type_ids
        bi._from_token_i = datum.from_entity_token_i()
        bi._to_token_i = datum.to_entity_token_i()
        bi._feature_encoding = feature_encoding

        return bi
