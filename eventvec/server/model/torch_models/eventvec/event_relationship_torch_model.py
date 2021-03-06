import torch.nn as nn
import torch

class EventRelationshipModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(EventRelationshipModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()
        self.i2h = nn.Linear(input_size * 2, output_size, device=device)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_1, input_2):
        input_combined = torch.cat((input_1, input_2), 1)
        hidden = self.i2h(input_combined)
        #hidden = self.relu(hidden)
        output = hidden
        output = self.dropout(hidden)
        #output = self.softmax(hidden)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=self.device)
