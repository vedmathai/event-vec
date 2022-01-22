import torch.nn as nn 
import torch


class EventModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(EventModel, self).__init__()
        self.i2o = nn.Linear(input_size * 4, output_size, device=device)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, verb_vector, subject_vector, object_vector, date_vector):
        input_combined = torch.cat((verb_vector, subject_vector, object_vector, date_vector), 1)
        output = self.i2o(input_combined)
        output = self.relu(output)
        output = self.dropout(output)
        return output
