from multiprocessing import pool
import torch.nn as nn
from torch import cat
from transformers import BertModel


class BertRelationshipClassifier(nn.Module):

    def __init__(self, dropout=0.3):

        super(BertRelationshipClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased') # noqa
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768 + 768, 352)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(352, 3)
        self.softmax = nn.Softmax()

    def forward(self, from_input_ids, from_attention_mask,
                from_token_type_ids, to_input_ids, to_attention_mask,
                to_token_type_ids, whole_sentence_input_ids,
                whole_sentence_attention_masks, whole_sentence_type_ids,
                from_token_i, to_token_i, feature_encoding):
        hidden_output, pooled_output = self.bert(
            input_ids=whole_sentence_input_ids, attention_mask=whole_sentence_attention_masks,
            token_type_ids=whole_sentence_type_ids, return_dict=False
        )
        token_1 = hidden_output[0][from_token_i].unsqueeze(0)
        token_2 = hidden_output[0][to_token_i].unsqueeze(0)
        #dropout_output1 = self.dropout(pooled_output_1)
        feature_encoding = feature_encoding.unsqueeze(0)
        catted_features = cat([token_1, token_2], dim=1)
        linear_output1 = self.linear1(catted_features)
        relu_output = self.relu(linear_output1)
        #dropout_output2 = self.dropout(relu_output)
        linear_output2 = self.linear2(relu_output)
        return linear_output2
