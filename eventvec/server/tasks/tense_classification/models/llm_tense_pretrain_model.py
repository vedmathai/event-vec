import os
import torch.nn as nn
import torch
from transformers import BertModel, RobertaModel


from eventvec.server.config import Config
from eventvec.server.model.llm_tense_pretraining_model.llm_input_model import LLMInput  # noqa

LLM_INPUT = 768


class LLMTensePretrainer(nn.Module):

    def __init__(self, run_config, dropout=0.4):
        super(LLMTensePretrainer, self).__init__()
        self._model_type = run_config.model_type()
        self._llm = run_config.llm()
        if run_config.llm() == 'bert':
            self.llm = BertModel.from_pretrained('bert-base-uncased') # noqa
            for param in self.llm.parameters():
                param.requires_grad = False
        if run_config.llm() == 'roberta':
            self.llm = RobertaModel.from_pretrained('roberta-base') # noqa
            modules = [self.llm.embeddings, *self.llm.encoder.layer[:-2]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.tense_linear1 = nn.Linear(LLM_INPUT, 352)
        self.relu = nn.ReLU()
        self.tense_classifier = nn.Linear(352, 7)

    def forward(self, datum):
        li = LLMInput.from_input_datum(datum)

        hidden_output, pooled_output = self.llm(
            input_ids=li.input_ids(),
            attention_mask=li.attention_mask(),
            token_type_ids=li.token_type_ids(),
            return_dict=False
        )
        token = hidden_output[0][li.token_i() + 1].unsqueeze(0)
        linear_output1 = self.tense_linear1(token)
        relu_output = self.relu(linear_output1)
        dropout_output1 = self.dropout(relu_output)
        output = self.tense_classifier(dropout_output1)
        return output

    def save(self):
        config = Config.instance()
        save_location = config.model_save_location()
        state_dict = self.state_dict()
        torch.save(state_dict, save_location)

    def load(self):
        config = Config.instance()
        save_location = config.model_save_location()
        if os.path.exists(save_location):
            state_dict = torch.load(save_location)
            self.load_state_dict(state_dict, strict=False)
        else:
            print('Warning: Model doesn\'t exist. Going with default'
                  'initialized')
