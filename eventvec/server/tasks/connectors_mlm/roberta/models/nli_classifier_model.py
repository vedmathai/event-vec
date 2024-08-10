import os
import torch.nn as nn
from torch import cat
import torch
from transformers import BertModel, RobertaModel, RobertaTokenizer, DistilBertTokenizer, DistilBertModel, RobertaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from numpy import arange
import numpy as np

from eventvec.server.config import Config
from eventvec.server.tasks.relationship_classification.models.bert_input import BertInput
from eventvec.server.tasks.entailment_classification.featurizers.clause_matcher import ClauseMatcher
from eventvec.server.tasks.factuality_estimator.inferers.infer import FactualityClassificationInfer  # noqa
from eventvec.server.tasks.factuality_estimator.inferers.infer_ml import FactualityRegressionInferML  # noqa

ranges = [(i, i + 0.3) for i in arange(-4, 4, 0.3)]

FEATURE_INPUT = len(ranges)

LLM_INPUT = 1024 #768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


llm = 'roberta'
class NLIConnectorClassifierModel(nn.Module):

    def __init__(self, run_config, dropout=0.5):
        super(NLIConnectorClassifierModel, self).__init__()
        config = Config.instance()
        self._forward_type = run_config.forward_type()
        self._llm = run_config.llm()
        self._run_config = run_config
        self._experiment_type = config.experiment_type()
        model_key = run_config.factuality_inference_model()
        self._save_location = config.model_save_location()
        if llm == 'roberta':
            self._tokenizer = RobertaTokenizer.from_pretrained("transformers_cache/roberta-large")
            self.llm = RobertaModel.from_pretrained('transformers_cache/roberta-large', output_attentions=True).to(device) # noqa
        modules = [self.llm.embeddings, *self.llm.encoder.layer[:]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.nli_linear1 = nn.Linear(LLM_INPUT, 352).to(device)
        self.relu = nn.ReLU()
        self.nli_relationship_classifier = nn.Linear(352, 3).to(device)

        self.connector_linear1 = nn.Linear(LLM_INPUT, 352).to(device)
        self.connector_relationship_classifier = nn.Linear(352, 5).to(device)

    def forward(self, datum, forward_type, train_test):
        if forward_type == 'nli':
            return self.nli_forward(datum, train_test)
        if forward_type == 'connector':
            return self.connector_forward(datum, train_test)

    def connector_forward(self, datum, train_test):
        if llm == 'roberta':
            encoded_sentence = self._tokenizer(
                [datum.para()],
                padding='max_length',
                max_length=500,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=True
            )
            encoded_sentence = {k: v.to(device) for k, v in encoded_sentence.items()}
            output = self.llm(**encoded_sentence)
            pooler_output = output.pooler_output
            input = pooler_output
        linear_output1 = self.connector_linear1(input)

        relu_output = self.relu(linear_output1)
        if train_test == 'test':
            self.dropout.eval()
        if train_test == 'train':
            self.dropout.train()
        dropout_output2 = self.dropout(relu_output)
        output = self.connector_relationship_classifier(dropout_output2)
        return output

    def nli_forward(self, datum, train_test):
        if llm == 'roberta':
            encoded_sentence = self._tokenizer(
                [datum.sentence_1()], [datum.sentence_2()],
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