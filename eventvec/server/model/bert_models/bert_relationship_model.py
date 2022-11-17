import torch.nn as nn
from torch import cat
from transformers import BertModel


from eventvec.server.model.bert_models.bert_input import BertInput


class BertRelationshipClassifier(nn.Module):

    def __init__(self, dropout=0.4):

        super(BertRelationshipClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased') # noqa
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear((768 + 768) * 1+ (1 + 7 + 7 + 4 + 4 + 9 + 9)*1, 352)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(352, 3)
        self.softmax = nn.Softmax()

    def forward(self, datum):
        bi = BertInput.from_input_datum(datum)
        hidden_output, pooled_output = self.bert(
            input_ids=bi.whole_sentence_input_ids(),
            attention_mask=bi.whole_sentence_attention_masks(),
            token_type_ids=bi.whole_sentence_token_type_ids(),
            return_dict=False
        )
        token_1 = hidden_output[0][bi.from_token_i()].unsqueeze(0)
        token_2 = hidden_output[0][bi.to_token_i()].unsqueeze(0)
        # dropout_output1 = self.dropout(pooled_output_1)
        # token1_dropout = self.dropout(token_1)
        # token2_dropout = self.dropout(token_2)
        feature_encoding = bi.feature_encoding()
        catted_features = cat([token_1, token_2, feature_encoding], dim=1)
        #catted_features = cat([token_1, token_2], dim=1)
        #linear_output1 = self.linear1(feature_encoding.float())
        linear_output1 = self.linear1(catted_features)
        relu_output = self.relu(linear_output1)
        # dropout_output2 = self.dropout(relu_output)
        linear_output2 = self.linear2(relu_output)
        return linear_output2
