import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LLMInput:
    def __init__(self):
        self._input_ids = None
        self._attention_mask = None
        self._token_type_ids = None
        self._token_i = None

    def input_ids(self):
        return self._input_ids

    def attention_mask(self):
        return self._attention_mask

    def token_type_ids(self):
        return self._token_type_ids

    def token_i(self):
        return self._token_i

    @staticmethod
    def from_input_datum(datum):
        li = LLMInput()
        sentence_inputs = datum.sentence_encoded()
        input_ids = sentence_inputs['input_ids']
        attention_mask = sentence_inputs['attention_mask']
        token_type_ids = sentence_inputs['token_type_ids']

        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        input_ids = input_ids.squeeze(1).to(device)

        li._input_ids = input_ids
        li._attention_mask = attention_mask
        li._token_type_ids = token_type_ids
        li._token_i = datum.entity_token_i()

        return li
