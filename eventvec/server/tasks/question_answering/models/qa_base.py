import os
import torch.nn as nn
from transformers import BigBirdTokenizer, BigBirdModel, RobertaTokenizer, RobertaModel
import torch

from eventvec.server.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuestionAnsweringBase(nn.Module):

    def __init__(self, run_config):
        super().__init__()
        config = Config.instance()
        if run_config.llm() == 'bigbird':
            self._tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base', pad_token="[PAD]")
            self._model = BigBirdModel.from_pretrained('google/bigbird-roberta-base', attention_type="original_full").to(device)
        if run_config.llm() == 'roberta':
            self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self._model = RobertaModel.from_pretrained('roberta-base')
        modules = [self._model.embeddings, *self._model.encoder.layer[:-4]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
        self._dropout = nn.Dropout(0.5).to(device)
        self._token_classifier_1 = torch.nn.Linear(768, 16).to(device)
        self._tense_classifier_1 = torch.nn.Linear(768, 16).to(device)
        self._token_classifier_activation = nn.Tanh().to(device)
        self._tense_classifier_activation = nn.Tanh().to(device)
        self._token_classifier_2 = torch.nn.Linear(16, 2).to(device)
        self._tense_classifier_2 = torch.nn.Linear(16, 4).to(device)

    def forward(self, question, paragraph):
        inputs = self._tokenizer([question], [paragraph], return_tensors="pt", padding='longest', max_length=1000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        hidden_states = outputs.last_hidden_state
        dropout_output = self._dropout(hidden_states[0])
        token_classification_1_output = self._token_classifier_1(dropout_output)
        tense_classification_1_output = self._tense_classifier_1(dropout_output)
        activation_output = self._token_classifier_activation(token_classification_1_output)
        tense_activation_output = self._tense_classifier_activation(tense_classification_1_output)
        token_classification_2_output = self._token_classifier_2(activation_output)
        tense_classification_2_output = self._tense_classifier_2(tense_activation_output)
        return token_classification_2_output, tense_classification_2_output

    def wordid2tokenid(self, question, paragraph):
        wordid2tokenid = {}
        inputs = self._tokenizer([question], [paragraph], return_tensors="pt", padding='longest', max_length=1000)
        tokens = self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]) #input tokens
        stoken_i = 0
        for ti, t in enumerate(tokens):
            if t[0] == '▁':
                wordid2tokenid[stoken_i] = ti
                stoken_i += 1
        return wordid2tokenid, tokens
