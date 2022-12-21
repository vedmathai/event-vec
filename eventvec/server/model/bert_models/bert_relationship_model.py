import os
import torch.nn as nn
from torch import cat
import torch
from transformers import BertModel, RobertaModel

from eventvec.server.config import Config
from eventvec.server.model.bert_models.bert_input import BertInput

LLM_INPUT = 768 + 768
FEATURE_INPUT = 1 * 0 + (7 * 1 + 4 * 1 + 17 * 1 + 16 * 1) * 2


class BertRelationshipClassifier(nn.Module):

    def __init__(self, run_config, dropout=0.5):
        super(BertRelationshipClassifier, self).__init__()
        config = Config.instance()
        self._model_type = run_config.model_type()
        self._llm = run_config.llm()
        self._experiment_type = config.experiment_type()
        self._save_location = config.model_save_location()
        if run_config.llm() == 'bert':
            self.llm = BertModel.from_pretrained('bert-base-uncased') # noqa
            for param in self.llm.parameters():
                param.requires_grad = False
        if run_config.llm() == 'roberta':
            self.llm = RobertaModel.from_pretrained('roberta-base') # noqa
            modules = [self.llm.embeddings, *self.llm.encoder.layer[:-1]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        if run_config.model_type() == 'llm_only':
            self.linear1 = nn.Linear(LLM_INPUT, 352)
        if run_config.model_type() == 'llm+features':
            self.linear1 = nn.Linear(LLM_INPUT + FEATURE_INPUT, 352)
        if run_config.model_type() == 'features_only':
            self.linear1 = nn.Linear(FEATURE_INPUT, 352)
        self.relu = nn.ReLU()
        self.relationship_classifier = nn.Linear(352, 5)

    def forward(self, datum):
        bi = BertInput.from_input_datum(datum)

        if self._model_type in ['features_only', 'llm+features']:
            feature_encoding = bi.feature_encoding()

        if self._model_type in ['llm+features', 'llm_only']:
            hidden_output, pooled_output = self.llm(
                input_ids=bi.whole_sentence_input_ids(),
                attention_mask=bi.whole_sentence_attention_masks(),
                token_type_ids=bi.whole_sentence_token_type_ids(),
                return_dict=False
            )
            token_1 = hidden_output[0][bi.from_token_i()].unsqueeze(0)
            token_2 = hidden_output[0][bi.to_token_i()].unsqueeze(0)

        if self._model_type in ['llm+features']:
            catted_features = cat([token_1, token_2, feature_encoding], dim=1)
            linear_output1 = self.linear1(catted_features)

        if self._model_type in ['llm_only']:
            catted_features = cat([token_1, token_2], dim=1)
            linear_output1 = self.linear1(catted_features)

        if self._model_type in ['features_only']:
            linear_output1 = self.linear1(feature_encoding.float())

        relu_output = self.relu(linear_output1)
        dropout_output2 = self.dropout(relu_output)
        output = self.relationship_classifier(dropout_output2)
        return output

    def save(self):
        save = False
        state_dict = self.state_dict()
        if save is True:
            torch.save(state_dict, self._save_location)

    def load(self):
        if os.path.exists(self._save_location):
            state_dict = torch.load(self._save_location)
            self.load_state_dict(state_dict, strict=False)
        else:
            print('Warning: Model doesn\'t exist. Going with default '
                  'initialized')
