import os
import torch.nn as nn
from torch import cat
import torch
from transformers import BertModel, RobertaModel, DistilBertModel, DistilBertTokenizer, XLMRobertaModel, XLMRobertaTokenizer
from transformers import BertTokenizer, RobertaTokenizer
# DeBERTa
from transformers import DebertaTokenizer, DebertaModel
from transformers import T5Tokenizer, T5Model
from transformers import BartTokenizer, BartModel
from transformers import AlbertTokenizer, AlbertModel
from transformers import ElectraTokenizer, ElectraModel

# !! Look at the cuations below


from eventvec.server.config import Config
from eventvec.server.tasks.relationship_classification.models.bert_input import BertInput

LLM_INPUT = 768
LLM_INPUT_DICT = {
    'roberta': 768,
    'electra': 1024,
    'distilbert': 768,
    'deberta': 1024,
    'xlm_roberta': 1024,
    'bart': 1024,
    'albert': 4096,
}
DELIMITER = 'Ġ'
XLM_ROBERTA_DELIMITER = '▁'
DEBERTA_DELIMITER = '#'
SENTENCE_BREAK = '</s>',

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer_dict = {
    #'roberta': RobertaTokenizer.from_pretrained("transformers_cache/roberta-large"),
    #'distilbert': DistilBertTokenizer.from_pretrained("transformers_cache/distilbert-base-uncased"),
    'xlm_roberta': XLMRobertaTokenizer.from_pretrained("transformers_cache/xlm-roberta-large"),
    #'deberta': DebertaTokenizer.from_pretrained("transformers_cache/deberta-large"),
    #'t5': T5Tokenizer.from_pretrained("transformers_cache/t5-large"),
    #'bart': BartTokenizer.from_pretrained("transformers_cache/bart-large"),
    #'albert': AlbertTokenizer.from_pretrained("transformers_cache/albert-xxlarge-v2"),
    #'electra': ElectraTokenizer.from_pretrained("transformers_cache/electra-large-discriminator"),
}

model_dict = {
    #'roberta': RobertaModel.from_pretrained("transformers_cache/roberta-large"),
    #'distilbert': DistilBertModel.from_pretrained("transformers_cache/distilbert-base-uncased"),
    'xlm_roberta': XLMRobertaModel.from_pretrained("transformers_cache/xlm-roberta-large"),
    #'deberta': DebertaModel.from_pretrained("transformers_cache/deberta-large"),
    #'t5': T5Model.from_pretrained("transformers_cache/t5-large"),
    #'bart': BartModel.from_pretrained("transformers_cache/bart-large"),
    #'albert': AlbertModel.from_pretrained("transformers_cache/albert-xxlarge-v2"),
    #'electra': ElectraModel.from_pretrained("transformers_cache/electra-large-discriminator"),
}
class FactualityEstimatorModel(nn.Module):

    def __init__(self, run_config, dropout=0.5):
        super(FactualityEstimatorModel, self).__init__()
        config = Config.instance()
        self._tokenizer = tokenizer_dict[run_config.llm()]
        self._experiment_type = config.experiment_type()
        self._llm_name = run_config.llm()
        self._save_location = '{}_{}.pkl'.format(config.model_save_location(), self._llm_name)
        self.llm = model_dict[run_config.llm()].to(device)
        modules = []
        if self._llm_name in ['roberta', 'xlm_roberta', 'deberta', 'electra']:
            modules = [self.llm.embeddings, *self.llm.encoder.layer[:-1]]
        if self._llm_name in ['bart']:
            modules = [*self.llm.encoder.layers[:], *self.llm.decoder.layers[:]]
        if self._llm_name == 'distilbert':
            modules = [self.llm.embeddings, *self.llm.transformer.layer[:-1]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
        self.dropout = nn.Dropout(dropout).to(device)
        self._pooler = nn.MaxPool1d(1, stride=1).to(device)
        self.linear1 = nn.Linear(LLM_INPUT_DICT[self._llm_name], 352).to(device)

        self.relu = nn.ReLU().to(device)
        self.estimator = nn.Linear(352, 1).to(device)
        self.double()

    def forward(self, datum):
        sentence = datum.text()
        token = datum.event_string()

        if self._llm_name in ['roberta', 'deberta']:
            encoded_sentence = self._tokenizer(
                [sentence],
                padding='max_length',
                max_length=500,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=True
            )
            answer_indices = self._align_answer_roberta(encoded_sentence, token)
        if self._llm_name in ['xlm_roberta', 't5', 'albert']:
            encoded_sentence = self._tokenizer(
                [sentence],
                padding='max_length',
                max_length=500,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=True
            )
            answer_indices = self._align_answer_xlm_roberta(encoded_sentence, token)
        if self._llm_name in ['distilbert', 'electra']:
            encoded_sentence = self._tokenizer(
                [sentence],
                padding='max_length',
                max_length=500,
                truncation=True,
                return_tensors='pt',
                #return_token_type_ids=True
            )
            answer_indices = self._align_answer_distilbert(encoded_sentence, token)
        if self._llm_name in ['bart']:
            encoded_sentence = self._tokenizer(
                [sentence],
                padding='max_length',
                max_length=500,
                truncation=True,
                return_tensors='pt',
                #return_token_type_ids=True
            )
            answer_indices = self._align_answer_roberta(encoded_sentence, token)
        encoded_sentence = {k: v.to(device) for k, v in encoded_sentence.items()}
        output = self.llm(**encoded_sentence)
        hidden_output = output.last_hidden_state
        answer_hidden = []

        for index in answer_indices[:1]: # !! Look into this max pooling later
            answer_hidden.append(hidden_output[0][index])
        if len(answer_hidden) == 0:
            answer_hidden.append(hidden_output[0][0])
        answer_hidden = torch.stack(answer_hidden)
        answer_hidden = answer_hidden.to(device)
        answer_hidden = self._pooler(answer_hidden)

        linear_output1 = self.linear1(answer_hidden)

        relu_output = self.relu(linear_output1)
        dropout_output2 = self.dropout(relu_output)
        output = self.estimator(dropout_output2)
        return output

    def _align_answer_roberta(self, encoded_sentence, event_string):
        token_delimiter = DELIMITER
        sentence_break = SENTENCE_BREAK
        summed_token_indices = []
        all_answer_indices = []
        summed_token = ""
        token_i = 0
        tokens = self._tokenizer.convert_ids_to_tokens(encoded_sentence['input_ids'][0]) #input tokens
        while token_i < len(tokens):
            token = tokens[token_i]
            if token[0] == token_delimiter :
                summed_token = token[1:]
                summed_token_indices = [token_i]
            if token[0] != token_delimiter and token not in [',', '.', '?', ':', ';', '"', "'", sentence_break]:
                summed_token += token
                summed_token_indices.append(token_i)
            if summed_token.lower() == event_string.lower():
                all_answer_indices.extend(summed_token_indices)
            token_i += 1
        return all_answer_indices
    
    def _align_answer_xlm_roberta(self, encoded_sentence, event_string):
        token_delimiter = XLM_ROBERTA_DELIMITER
        sentence_break = SENTENCE_BREAK
        summed_token_indices = []
        all_answer_indices = []
        summed_token = ""
        token_i = 0
        tokens = self._tokenizer.convert_ids_to_tokens(encoded_sentence['input_ids'][0]) #input tokens
        while token_i < len(tokens):
            token = tokens[token_i]
            if token[0] == token_delimiter :
                summed_token = token[1:]
                summed_token_indices = [token_i]
            if token[0] != token_delimiter and token not in [',', '.', '?', ':', ';', '"', "'", sentence_break]:
                summed_token += token
                summed_token_indices.append(token_i)
            if summed_token.lower() == event_string.lower():
                all_answer_indices.extend(summed_token_indices)
            token_i += 1
        return all_answer_indices
    
    def _align_answer_distilbert(self, encoded_sentence, event_string):
        token_delimiter = '#'
        sentence_break = SENTENCE_BREAK
        summed_token_indices = []
        all_answer_indices = []
        summed_token = ""
        token_i = 0
        tokens = self._tokenizer.convert_ids_to_tokens(encoded_sentence['input_ids'][0]) #input tokens
        while token_i < len(tokens):
            token = tokens[token_i]
            if token[0] != token_delimiter:
                summed_token = token
                summed_token_indices = [token_i]
            if token[0] == token_delimiter and token not in [',', '.', '?', ':', ';', '"', "'", sentence_break]:
                summed_token += token[2:]
                summed_token_indices.append(token_i)
            if summed_token.lower() == event_string.lower():
                all_answer_indices.extend(summed_token_indices)

            token_i += 1
        return all_answer_indices
    
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
