from torch import nn


class Variables():
    def __init__(self):
        self.learning_rate = 10e-5
        self.criterion = nn.NLLLoss
        self.device = 'cuda'
        self.n_iters = int(1e6)
        self.print_every = 5e3
        self.plot_every = 1e6
        self.vocabulary_size = 0

        self.data_folder = '/home/vedmathai/python/temporal/timebank_1_2/data/timeml'
