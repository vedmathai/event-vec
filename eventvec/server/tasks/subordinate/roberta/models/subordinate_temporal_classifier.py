import os
import torch.nn as nn
import torch
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoTokenizer, MPNetModel
import numpy as np
import os

dollar_data = '/data/engs-exaggeration/lady6977'
from eventvec.server.config import Config

LLM_INPUT = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')

tense2index = {
    'past': 0,
    'present': 1,
    'future': 2,
}

aspect2index = {
    'simple': 0,
    'continuous': 1,
    'perfect': 2,
    'perfect-continuous': 3,
}

is_quote2index = {
    'yes': 1,
    'no': 0,
}

temporal_marker2index = {
    'no_marker': 0,
    'yesterday': 1,
    'tomorrow': 2,
    'now': 3,
    'today': 4,
    'everyday': 5,
}


llm = 'roberta'
class SubordinateTemporalClassifierModel(nn.Module):

    def __init__(self, run_config, dropout=0.5):
        super(SubordinateTemporalClassifierModel, self).__init__()
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
        if self._forward_type == 'llm_only':
            linear_input_size = LLM_INPUT
        if self._forward_type == 'llm+features':
            linear_input_size = LLM_INPUT + 3 + 4 + 3 + 4 + 2 + 6
        if self._forward_type == 'features':
            linear_input_size = 3 + 4 + 3 + 4 + 2 + 6
        self.nli_linear1 = nn.Linear(linear_input_size, 352).to(device)
        self.relu = nn.ReLU()
        self.nli_relationship_classifier = nn.Linear(352, 3).to(device)


    def forward(self, datum, forward_type, train_test):
        return self.nli_forward(datum, train_test)

    def nli_forward(self, datum, train_test):
        if self._forward_type in ['llm_only', 'llm+features']:
            if llm in ['roberta', 'mpnet']:
                encoded_sentence = self._tokenizer(
                    [datum.example()],
                    padding='max_length',
                    max_length=500,
                    truncation=True,
                    return_tensors='pt',
                    return_token_type_ids=True
                )
                encoded_sentence = {k: v.to(device) for k, v in encoded_sentence.items()}
                output = self.llm(**encoded_sentence)
                input = output.pooler_output
        if self._forward_type in ['llm+features']:
            matrix_tense = [0, 0, 0]
            matrix_aspect = [0, 0, 0, 0]
            sub_tense = [0, 0, 0]
            sub_aspect = [0, 0, 0, 0]
            is_queue = [0, 0]
            temporal_marker = [0, 0, 0, 0, 0, 0]
            matrix_tense[tense2index[datum.matrix_tense().lower()]] = 1
            matrix_aspect[aspect2index[datum.matrix_aspect().lower()]] = 1
            sub_tense[tense2index[datum.sub_tense().lower()]] = 1
            sub_aspect[aspect2index[datum.sub_aspect().lower()]] = 1
            is_queue[is_quote2index[datum.is_quote().lower()]] = 1
            temporal_marker[temporal_marker2index[datum.temporal_marker().lower()]] = 1
            feature_encoding = torch.tensor([matrix_tense + matrix_aspect + sub_tense + sub_aspect + is_queue + temporal_marker]).to(device)
            input = torch.cat([input, feature_encoding], dim=1)
        if self._forward_type in ['features']:
            matrix_tense = [0, 0, 0]
            matrix_aspect = [0, 0, 0, 0]
            sub_tense = [0, 0, 0]
            sub_aspect = [0, 0, 0, 0]
            is_queue = [0, 0]
            temporal_marker = [0, 0, 0, 0, 0, 0]
            matrix_tense[tense2index[datum.matrix_tense().lower()]] = 1
            matrix_aspect[aspect2index[datum.matrix_aspect().lower()]] = 1
            sub_tense[tense2index[datum.sub_tense().lower()]] = 1
            sub_aspect[aspect2index[datum.sub_aspect().lower()]] = 1
            is_queue[is_quote2index[datum.is_quote().lower()]] = 1
            temporal_marker[temporal_marker2index[datum.temporal_marker().lower()]] = 1
            feature_encoding = torch.tensor([matrix_tense + matrix_aspect + sub_tense + sub_aspect + is_queue + temporal_marker]).to(torch.float).to(device)
            input = torch.cat([feature_encoding], dim=1)
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