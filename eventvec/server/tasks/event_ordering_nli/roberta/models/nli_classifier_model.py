import os
import torch.nn as nn
from torch import cat
import torch
from transformers import BertModel, RobertaModel, RobertaTokenizer, DistilBertTokenizer, DistilBertModel, RobertaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, MPNetModel
from numpy import arange
import numpy as np
import os

dollar_data = '/data/engs-exaggeration/lady6977'
from eventvec.server.config import Config

LLM_INPUT = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')

llm = 'roberta'
class NLITemporalClassifierModel(nn.Module):

    def __init__(self, run_config, dropout=0.5):
        super(NLITemporalClassifierModel, self).__init__()
        config = Config.instance()
        self._forward_type = run_config.forward_type()
        self._llm = run_config.llm()
        self._run_config = run_config
        self._experiment_type = config.experiment_type()
        self._save_location = config.model_save_location()
        if llm == 'roberta':
            self._tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large")
            self.llm = RobertaModel.from_pretrained("FacebookAI/roberta-large").to(device) # noqa
        if llm == 'mpnet':
            self._tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
            self.llm = MPNetModel.from_pretrained('microsoft/mpnet-base').to(device)
        modules = [self.llm.embeddings, *self.llm.encoder.layer[:]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.nli_linear1 = nn.Linear(LLM_INPUT, 352).to(device)
        self.relu = nn.ReLU()
        self.nli_relationship_classifier = nn.Linear(352, 3).to(device)

        self.connector_linear1 = nn.Linear(LLM_INPUT, 352).to(device)
        self.connector_relationship_classifier = nn.Linear(352, 5).to(device)

    def forward(self, datum, forward_type, train_test):
        return self.nli_forward(datum, train_test)

    def nli_forward(self, datum, train_test):
        if llm in ['roberta', 'mpnet']:
            encoded_sentence = self._tokenizer(
                [datum.premise()], [datum.hypothesis()],
                padding='max_length',
                max_length=500,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=True
            )
            encoded_sentence = {k: v.to(device) for k, v in encoded_sentence.items()}
            output = self.llm(**encoded_sentence)
            input = output.pooler_output
        linear_output1 = self.nli_linear1(input)

        relu_output = self.relu(linear_output1)
        if train_test == 'test':
            self.dropout.eval()
        if train_test == 'train':
            self.dropout.train()
        dropout_output2 = self.dropout(relu_output)
        output = self.nli_relationship_classifier(dropout_output2)
        return output

    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self._save_location)

    def load(self):
        if os.path.exists(self._save_location):
            state_dict = torch.load(self._save_location, map_location=torch.device(device))
            self.load_state_dict(state_dict, strict=False)
        else:
            print('Warning: Model doesn\'t exist. Going with default '
                  'initialized')