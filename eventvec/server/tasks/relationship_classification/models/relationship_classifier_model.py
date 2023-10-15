import os
import torch.nn as nn
from torch import cat
import torch
from transformers import BertModel, RobertaModel

from eventvec.server.config import Config
from eventvec.server.tasks.relationship_classification.models.bert_input import BertInput

LLM_INPUT = 768 + 768
num_tenses = 4
num_aspects = 3
FEATURE_INPUT = (num_tenses + num_aspects) * 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RelationshipClassifierModel(nn.Module):

    def __init__(self, run_config, dropout=0.5):
        super(RelationshipClassifierModel, self).__init__()
        config = Config.instance()
        self._forward_type = run_config.forward_type()
        self._llm = run_config.llm()
        self._experiment_type = config.experiment_type()
        self._save_location = config.model_save_location()
        if run_config.llm() == 'bert':
            self.llm = BertModel.from_pretrained('bert-base-uncased').to(device) # noqa
            for param in self.llm.parameters():
                param.requires_grad = False
        if run_config.llm() == 'roberta':
            self.llm = RobertaModel.from_pretrained('roberta-base').to(device) # noqa
            modules = [self.llm.embeddings, *self.llm.encoder.layer[:-1]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        if run_config.forward_type() == 'llm_only':
            self.linear1 = nn.Linear(LLM_INPUT, 352).to(device)
        if run_config.forward_type() == 'llm+features':
            self.linear1 = nn.Linear(LLM_INPUT + FEATURE_INPUT, 352).to(device)
        if run_config.forward_type() == 'features_only':
            self.linear1 = nn.Linear(FEATURE_INPUT, 352).to(device)
        self.relu = nn.ReLU()
        self.relationship_classifier = nn.Linear(352, 6).to(device)

    def forward(self, datum):
        bi = BertInput.from_input_datum(datum)

        if self._forward_type in ['features_only', 'llm+features']:
            feature_encoding = bi.feature_encoding().to(device)

        if self._forward_type in ['llm+features', 'llm_only']:
            hidden_output, pooled_output = self.llm(
                input_ids=bi.whole_sentence_input_ids(),
                attention_mask=bi.whole_sentence_attention_masks(),
                token_type_ids=bi.whole_sentence_token_type_ids(),
                return_dict=False
            )
            token_1 = hidden_output[0][bi.from_token_i()].unsqueeze(0).to(device)
            token_2 = hidden_output[0][bi.to_token_i()].unsqueeze(0).to(device)

        if self._forward_type in ['llm+features']:
            catted_features = cat([token_1, token_2, feature_encoding], dim=1).to(device)
            linear_output1 = self.linear1(catted_features)

        if self._forward_type in ['llm_only']:
            catted_features = cat([token_1, token_2], dim=1)
            linear_output1 = self.linear1(catted_features)

        if self._forward_type in ['features_only']:
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
