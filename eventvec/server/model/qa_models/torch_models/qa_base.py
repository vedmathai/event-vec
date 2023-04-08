import os
import torch.nn as nn
from transformers import BigBirdTokenizer, BigBirdModel
import torch

from eventvec.server.config import Config


class QuestionAnsweringBase(nn.Module):

    def __init__(self):
        super().__init__()
        config = Config.instance()
        self._tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base', pad_token="[PAD]")
        self._model = BigBirdModel.from_pretrained('google/bigbird-roberta-base', block_size=16, num_random_blocks=2)
        modules = [self._model.embeddings, *self._model.encoder.layer[:-3]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        self._start_classifier_weight = torch.nn.Parameter(torch.randn(768), requires_grad=True)
        self._end_classifier_weight = torch.nn.Parameter(torch.randn(768), requires_grad=True)


    def forward(self, question, paragraph):
        inputs = self._tokenizer([question], [paragraph], return_tensors="pt", padding='longest', max_length=1000)
        outputs = self._model(**inputs)
        hidden_states = outputs.last_hidden_state
        start_logits = torch.matmul(hidden_states, self._start_classifier_weight)
        end_logits = torch.matmul(hidden_states, self._end_classifier_weight)
        return start_logits, end_logits

    def wordid2tokenid(self, question, paragraph):
        wordid2tokenid = {}
        inputs = self._tokenizer([question], [paragraph], return_tensors="pt", padding='longest', max_length=1000)
        tokens = self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]) #input tokens
        stoken_i = 0
        for ti, t in enumerate(tokens):
            if t[0] == '▁':
                wordid2tokenid[stoken_i] = ti
                stoken_i += 1
        return wordid2tokenid
