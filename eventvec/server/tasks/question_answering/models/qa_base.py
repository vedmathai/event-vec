import os
import torch.nn as nn
from transformers import BigBirdTokenizer, BigBirdModel, RobertaTokenizer, RobertaModel
import torch

from eventvec.server.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_layer_size = 16

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
        self._forward_type = run_config.forward_type()
        self._run_config = run_config
        self._dropout = nn.Dropout(0.5).to(device)
        self._base_layer_classifier = torch.nn.Linear(768, hidden_layer_size).to(device)
        self._base_classifier_activation = nn.Tanh().to(device)
        classifier_size = hidden_layer_size
        if run_config.forward_type() == 'features' and run_config.use_tense() is True:
            classifier_size += 4
        if run_config.forward_type() == 'features' and run_config.use_aspect() is True:
            classifier_size += 3
        if run_config.forward_type() == 'features' and run_config.use_pos() is True:
            classifier_size += 5

        if run_config.forward_type() == 'attention' and run_config.use_tense() is True:
            classifier_size += 16
        if run_config.forward_type() == 'attention' and run_config.use_aspect() is True:
            classifier_size += 16
        if run_config.forward_type() == 'attention' and run_config.use_pos() is True:
            classifier_size += 16
        if run_config.forward_type() == 'attention_mul':
            classifier_size = hidden_layer_size + 16

        self._token_classifier = torch.nn.Linear(classifier_size, 2).to(device)
        #self._token_classifier = torch.nn.Linear(hidden_layer_size, 2).to(device)

        self._tense_attention = nn.MultiheadAttention(16, 16).to(device)
        self._aspect_attention = nn.MultiheadAttention(16, 16).to(device)
        self._pos_attention = nn.MultiheadAttention(16, 16).to(device)
        self._attention = nn.MultiheadAttention(16, 16).to(device)
        self._question_event_classifier = torch.nn.Linear(hidden_layer_size, 2).to(device)
        self._tense_classifier = torch.nn.Linear(hidden_layer_size, 4).to(device)
        self._tense_intermediate_classifier = torch.nn.Linear(768, hidden_layer_size).to(device)

        self._pos_classifier = torch.nn.Linear(hidden_layer_size, 5).to(device)
        self._pos_intermediate_classifier = torch.nn.Linear(768, hidden_layer_size).to(device)

        self._aspect_classifier = torch.nn.Linear(hidden_layer_size, 3).to(device)
        self._aspect_intermediate_classifier = torch.nn.Linear(768, hidden_layer_size).to(device)
        self._question_classifier = torch.nn.Linear(768, 12).to(device)

    def forward(self, question, paragraph):
        forwards_dict = {
            'plain': self.plain_forward,
            'features': self.features_forward,
            'attention': self.attention_forward,
            'attention_mul': self.attention_forward_mul,
        }
        return forwards_dict[self._forward_type](question, paragraph)

    def plain_forward(self, question, paragraph):
        inputs = self._tokenizer([question], [paragraph], return_tensors="pt", padding='longest', max_length=1000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        hidden_states = outputs.last_hidden_state
        dropout_output = self._dropout(hidden_states[0])
        base_classification_output = self._base_layer_classifier(dropout_output)
        activation_output = self._base_classifier_activation(base_classification_output)
        token_classification_output = self._token_classifier(activation_output)
        return (
            token_classification_output,
            0, 0, 0, 0, 0
        )
    
    def features_forward(self, question, paragraph):
        inputs = self._tokenizer([question], [paragraph], return_tensors="pt", padding='longest', max_length=1000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        hidden_states = outputs.last_hidden_state
        dropout_output = self._dropout(hidden_states[0])
        base_classification_output = self._base_layer_classifier(dropout_output)
        activation_output = self._base_classifier_activation(base_classification_output)
        question_event_classification_output = self._question_event_classifier(activation_output)
        tense_intermediate_output = self._tense_intermediate_classifier(dropout_output)
        tense_activation_output = self._base_classifier_activation(tense_intermediate_output)
        tense_classification_output = self._tense_classifier(tense_activation_output)
        
        aspect_intermediate_output = self._aspect_intermediate_classifier(dropout_output)
        aspect_activation_output = self._base_classifier_activation(aspect_intermediate_output)
        aspect_classification_output = self._aspect_classifier(aspect_activation_output)

        pos_intermediate_output = self._pos_intermediate_classifier(dropout_output)
        pos_activation_output = self._base_classifier_activation(pos_intermediate_output)
        pos_classification_output = self._pos_classifier(pos_activation_output)
        
        token_classifier_input = activation_output
        if self._run_config.use_tense() is True:
            token_classifier_input = torch.concat((
                token_classifier_input,
                tense_classification_output
            ), dim=1)
        if self._run_config.use_aspect() is True:
            token_classifier_input = torch.concat((
                token_classifier_input,
                aspect_classification_output
            ), dim=1)
        if self._run_config.use_pos() is True:
            token_classifier_input = torch.concat((
                token_classifier_input,
                pos_classification_output
            ), dim=1)

        token_classification_output = self._token_classifier(token_classifier_input)
        return (
            token_classification_output,
            question_event_classification_output,
            tense_classification_output,
            aspect_classification_output,
            pos_classification_output,
            0
        )
    
    def attention_forward(self, question, paragraph):
        inputs = self._tokenizer([question], [paragraph], return_tensors="pt", padding='longest', max_length=1000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        hidden_states = outputs.last_hidden_state
        dropout_output = self._dropout(hidden_states[0])
        base_classification_output = self._base_layer_classifier(dropout_output)
        activation_output = self._base_classifier_activation(base_classification_output)
        question_event_classification_output = self._question_event_classifier(activation_output)
        
        tense_intermediate_output = self._tense_intermediate_classifier(dropout_output)
        tense_activation_output = self._base_classifier_activation(tense_intermediate_output)
        tense_classification_output = self._tense_classifier(tense_activation_output)
        tense_attention_output, tense_attention_output_weights = self._tense_attention(tense_activation_output.unsqueeze(0), activation_output.unsqueeze(0), activation_output.unsqueeze(0))

        aspect_intermediate_output = self._aspect_intermediate_classifier(dropout_output)
        aspect_activation_output = self._base_classifier_activation(aspect_intermediate_output)
        aspect_classification_output = self._aspect_classifier(aspect_activation_output)
        aspect_attention_output, aspect_attention_output_weights = self._aspect_attention(aspect_activation_output.unsqueeze(0), activation_output.unsqueeze(0), activation_output.unsqueeze(0))
        
        pos_intermediate_output = self._pos_intermediate_classifier(dropout_output)
        pos_activation_output = self._base_classifier_activation(pos_intermediate_output)
        pos_classification_output = self._pos_classifier(pos_activation_output)
        pos_attention_output, pos_attention_output_weights = self._pos_attention(pos_activation_output.unsqueeze(0), activation_output.unsqueeze(0), activation_output.unsqueeze(0))
        
        token_classifier_input = activation_output
        
        if self._run_config.use_tense() is True:
            token_classifier_input = torch.concat((
                token_classifier_input,
                tense_attention_output[0],
            ), dim=1)
        if self._run_config.use_aspect() is True:
            token_classifier_input = torch.concat((
                token_classifier_input,
                aspect_attention_output[0],
            ), dim=1)
        if self._run_config.use_pos() is True:
            token_classifier_input = torch.concat((
                token_classifier_input,
                pos_attention_output[0],
            ), dim=1)
    
        token_classification_output = self._token_classifier(token_classifier_input)
        return (
            token_classification_output,
            question_event_classification_output,
            tense_classification_output,
            aspect_classification_output,
            pos_classification_output,
            0
        )
    
    def attention_forward_mul(self, question, paragraph):
        inputs = self._tokenizer([question], [paragraph], return_tensors="pt", padding='longest', max_length=1000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        hidden_states = outputs.last_hidden_state
        dropout_output = self._dropout(hidden_states[0])
        base_classification_output = self._base_layer_classifier(dropout_output)
        activation_output = self._base_classifier_activation(base_classification_output)
        question_event_classification_output = self._question_event_classifier(activation_output)
        
        tense_intermediate_output = self._tense_intermediate_classifier(dropout_output)
        tense_activation_output = self._base_classifier_activation(tense_intermediate_output)
        tense_classification_output = self._tense_classifier(tense_activation_output)
        tense_attention_output, tense_attention_output_weights = self._tense_attention(tense_activation_output.unsqueeze(0), activation_output.unsqueeze(0), activation_output.unsqueeze(0))

        aspect_intermediate_output = self._aspect_intermediate_classifier(dropout_output)
        aspect_activation_output = self._base_classifier_activation(aspect_intermediate_output)
        aspect_classification_output = self._aspect_classifier(aspect_activation_output)
        aspect_attention_output, aspect_attention_output_weights = self._aspect_attention(aspect_activation_output.unsqueeze(0), activation_output.unsqueeze(0), activation_output.unsqueeze(0))
        
        pos_intermediate_output = self._pos_intermediate_classifier(dropout_output)
        pos_activation_output = self._base_classifier_activation(pos_intermediate_output)
        pos_classification_output = self._pos_classifier(pos_activation_output)
        pos_attention_output, pos_attention_output_weights = self._pos_attention(pos_activation_output.unsqueeze(0), activation_output.unsqueeze(0), activation_output.unsqueeze(0))
        
        token_classifier_input = activation_output
        
        features = tense_attention_output[0] * aspect_attention_output[0]
        token_classifier_input = torch.concat((token_classifier_input, features), dim = 1)
        token_classification_output = self._token_classifier(token_classifier_input)
        return (
            token_classification_output,
            question_event_classification_output,
            tense_classification_output,
            aspect_classification_output,
            pos_classification_output,
            0
        )

    def forward_question_classification(self, question, paragraph):
        inputs = self._tokenizer([question], [paragraph], return_tensors="pt", padding='longest', max_length=1000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        question_inputs = self._tokenizer([question], return_tensors="pt", padding='longest', max_length=1000)
        question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
        outputs = self._model(**inputs)
        question_outputs = self._model(**inputs)
        hidden_states = outputs.last_hidden_state
        pooled_output = question_outputs.pooler_output
        pooled_dropout_output = self._dropout(pooled_output)
        question_classification_output = self._question_classifier(pooled_dropout_output)
        dropout_output = self._dropout(hidden_states[0])
        base_classification_output = self._base_layer_classifier(dropout_output)
        activation_output = self._base_classifier_activation(base_classification_output)
        question_event_classification_output = self._question_event_classifier(activation_output)
        tense_classification_output = self._tense_classifier(activation_output)
        aspect_classification_output = self._aspect_classifier(activation_output)
        pos_classification_output = self._pos_classifier(activation_output)
        question_classification = question_classification_output * torch.ones([len(activation_output), 12]).to(device)
        token_classifier_input = torch.concat((
            self._base_classifier_activation(question_classification),
            self._base_classifier_activation(question_event_classification_output),
            activation_output,
            self._base_classifier_activation(tense_classification_output),
            self._base_classifier_activation(aspect_classification_output),
            self._base_classifier_activation(pos_classification_output),
        ), dim=1)
        token_classification_output = self._token_classifier(token_classifier_input)
        return (
            token_classification_output,
            question_event_classification_output,
            tense_classification_output,
            aspect_classification_output,
            pos_classification_output,
            question_classification_output,
        )

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
