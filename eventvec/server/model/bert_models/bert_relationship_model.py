from multiprocessing import pool
import torch.nn as nn
from transformers import BertModel


class BertRelationshipClassifier(nn.Module):

    def __init__(self, dropout=0.1):

        super(BertRelationshipClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased') # noqa
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 352)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(352, 3)
        self.softmax = nn.Softmax()

    def forward(self, input_id, mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask,
            token_type_ids=token_type_ids, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output1 = self.linear1(dropout_output)
        #relu_output = self.relu(linear_output1)
        linear_output2 = self.linear2(linear_output1)
        return linear_output2
