import torch.nn as nn
import torch

class EventPartsRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(EventPartsRNN, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size, device=device)
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(hidden_size + hidden_size, hidden_size, device=device)
        self.relu = nn.ReLU()
        self.i2o = nn.Linear(hidden_size + hidden_size, output_size, device=device)
        self.o2o = nn.Linear(hidden_size + output_size, output_size, device=device)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input, hidden):
        input = self.embedding(input)
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        hidden = self.relu(hidden)
        output = self.i2o(input_combined)
        output = self.relu(output)
        #output_combined = torch.cat((hidden, output), 1)
        #output = self.o2o(output_combined)
        #output = self.relu(output)
        output = self.dropout(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=self.device)

    def initOutput(self):
        return torch.zeros(1, self.hidden_size, device=self.device)
