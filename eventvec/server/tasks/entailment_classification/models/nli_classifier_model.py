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

facuality_infer_dict = {
    "factuality_infer": FactualityClassificationInfer,
    "factuality_infer_ml": FactualityRegressionInferML,
}

llm = 'llama2'
class NLIClassifierModel(nn.Module):

    def __init__(self, run_config, dropout=0.5):
        super(NLIClassifierModel, self).__init__()
        config = Config.instance()
        self._forward_type = run_config.forward_type()
        self._llm = run_config.llm()
        self._run_config = run_config
        self._experiment_type = config.experiment_type()
        self._clause_matcher = ClauseMatcher()
        model_key = run_config.factuality_inference_model()
        self._factuality_infer = facuality_infer_dict.get(model_key)()
        self._factuality_infer.load(run_config)
        self._save_location = config.model_save_location()
        if llm == 'roberta':
            self._tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
            self.llm = RobertaModel.from_pretrained('roberta-large', output_attentions=True).to(device) # noqa
        if llm == 'distilbert':
            self._tokenizer = DistilBertTokenizer.from_pretrained("transformers_cache/distilbert-base-uncased")
            self.llm = DistilBertModel.from_pretrained("transformers_cache/distilbert-base-uncased").to(device) # noqa
        if llm == 'llama2':
            self._tokenizer = AutoTokenizer.from_pretrained("transformers_cache/Llama-2-7b-hf")
            self._llm = AutoModelForCausalLM.from_pretrained("transformers_cache/Llama-2-7b-hf")
        #modules = [self.llm.embeddings, *self.llm.encoder.layer[:-1]]
        #for module in modules:
        #    for param in module.parameters():
        #        param.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        if run_config.forward_type() == 'llm_only':
            self.linear1 = nn.Linear(LLM_INPUT, 352).to(device)
        if run_config.forward_type() == 'llm+features':
            # self.linear1 = nn.Linear(LLM_INPUT + FEATURE_INPUT * 2 + 2, 352).to(device)
            self.linear1 = nn.Linear(LLM_INPUT + 2, 352).to(device)

        self.relu = nn.ReLU()
        self.relationship_classifier = nn.Linear(352, 3).to(device)

    def forward(self, datum, train_test):
        event_string, event_string_2 = self._clause_matcher.match(datum.sentence_1(), datum.sentence_2())
        average_entropy = 0
        calculate_average_entropy = True
        if self._run_config.forward_type() == 'llm+features':
            factuality_score = self._factuality_infer.infer(datum.sentence_1(), event_string)
            for ii, i in enumerate(ranges):
                if factuality_score >= i[0] and factuality_score < i[1]:
                    facuality_flag = ii
                    break
            
            factuality_array = [0] * len(ranges)
            factuality_array[facuality_flag] = 1
            factuality_array = torch.tensor(factuality_array).to(device)

            factuality_score_2 = self._factuality_infer.infer(datum.sentence_2(), event_string_2)
            for ii, i in enumerate(ranges):
                if factuality_score_2 >= i[0] and factuality_score_2 < i[1]:
                    facuality_flag = ii
                    break
            
            factuality_array_2 = [0] * len(ranges)
            factuality_array_2[facuality_flag] = 1
            factuality_array_2 = torch.tensor(factuality_array_2).to(device)
            factuality_direct = torch.tensor([factuality_score]).to(device).unsqueeze(0)
            factuality_direct_2 = torch.tensor([factuality_score_2]).to(device).unsqueeze(0)
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
            if calculate_average_entropy is True and train_test == 'test':
                average_entropy = []
                token = output.attentions[-1][0][-1][0]
                entropy = 0
                for i in token:
                    if i > 0:
                        k = i.detach().cpu().numpy()
                        entropy += -1 * k * np.log(k)
                #average_entropy = np.mean(np.array(average_entropy))
                average_entropy = entropy
            pooler_output = output.pooler_output
        if llm == 'llama2':
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
            pooler_output = output.pooler_output
        if llm == 'distilbert':
            encoded_sentence = self._tokenizer(
                [datum.sentence_1()], [datum.sentence_2()],
                padding='max_length',
                max_length=500,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=False
            )
            encoded_sentence = {k: v.to(device) for k, v in encoded_sentence.items()}
            output = self.llm(**encoded_sentence)
            pooler_output = output.last_hidden_state[0][0].unsqueeze(0)
        if self._forward_type == 'llm+features':
            #input = torch.cat((pooler_output, factuality_array.unsqueeze(0), factuality_array_2.unsqueeze(0), factuality_direct, factuality_direct_2), 1)
            input = torch.cat((pooler_output, factuality_direct, factuality_direct_2), 1)
        else:
            input = pooler_output
        linear_output1 = self.linear1(input)

        relu_output = self.relu(linear_output1)
        dropout_output2 = self.dropout(relu_output)
        output = self.relationship_classifier(dropout_output2)
        return output, event_string_2, average_entropy

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