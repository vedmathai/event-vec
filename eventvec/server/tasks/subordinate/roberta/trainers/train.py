import numpy as np
import os
from torch import nn
import torch
from tqdm import tqdm
from torch.optim import Adam
from jadelogs import JadeLogger


from eventvec.server.tasks.subordinate.roberta.models.subordinate_temporal_classifier import SubordinateTemporalClassifierModel  # noqa
from eventvec.server.tasks.event_vectorization.datahandlers.data_handler_registry import DataHandlerRegistry


TRAIN_SAMPLE_SIZE = int(8000 / 5)
TEST_SAMPLE_SIZE = 2000
EPOCHS = 60
LEARNING_RATE = 1e-6  # 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 10000000

labels2idx = {
    'after': 0,
    'before': 1,
    'during': 2,
}

idx2label = {
    v: k for k, v in labels2idx.items()
}

class TemporalSubordinateClassificationTrain:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._data_handler_registry = DataHandlerRegistry()
        self._total_loss = 0
        self._all_losses = []
        self._iteration = 0
        self._last_iteration = 0
        self._loss = None

    def load(self, run_config):
        data_handler = 'subordinate_datahandler'
        self._data_handler = self._data_handler_registry.get_data_handler(data_handler)
        self._data_handler.load(run_config)
        self._target_type = run_config.test_dataset()
        self._model = SubordinateTemporalClassifierModel(run_config)
        self._model_optimizer = Adam(
            [
                {'params': self._model.nli_linear1.parameters(), 'lr':0.01},
                {'params': self._model.nli_relationship_classifier.parameters(), 'lr':0.01},
            ],
            lr=LEARNING_RATE,
        )
        self._criterion = nn.CrossEntropyLoss()
        self._softmax = nn.Softmax(dim=1)

    def zero_grad(self):
        self._model.zero_grad()

    def optimizer_step(self):
        self._model_optimizer.step()

    def train_nli_step(self, datum):
        event_predicted_vector = self.classify(datum, 'nli', 'train')
        relationship_target, target = self.temporal_target(datum)
        event_prediction_loss = self._criterion(
            event_predicted_vector, relationship_target
        )
        predicted = event_predicted_vector.argmax(dim=1).item()
        predicted_label = idx2label[predicted]
        if self._loss is None:
            self._loss = event_prediction_loss
        else:
            self._loss += event_prediction_loss
        return event_prediction_loss, predicted_label, target
    
    def temporal_target(self, datum):
        relationship_target = np.array([0 for i in range(3)]).astype(float)
        if self._target_type == 'dct_is_sub':
            target = datum.dct_is_sub()
        if self._target_type == 'dct_is_matrix':
            target = datum.dct_is_matrix()
        if self._target_type == 'matrix_is_sub':
            target = datum.matrix_is_sub()
        for i in labels2idx:
            if i in target:
                label_idx = labels2idx[i]
                relationship_target[label_idx] = 1
        relationship_target = torch.from_numpy(relationship_target).to(device)
        relationship_target = relationship_target.unsqueeze(0)
        return relationship_target, target

    def classify(self, datum, model_type, train_test):
        output = self._model(datum, model_type, train_test)
        return output

    def train_epoch(self):
        self.zero_grad()
        train_sample = self._data_handler.train_data()
        self._jade_logger.new_train_batch()
        for datum_i, datum in enumerate(tqdm(train_sample)):
            loss, predicted_nli_label, expected_label = self.train_nli_step(datum)
            self._all_losses += [loss.item()]
            self._iteration += 1

            if self._loss is not None and self._iteration % 10 == 0:
                self._loss.backward()
                self.optimizer_step()
                self.zero_grad()
                self._loss = None
            self._jade_logger.new_train_datapoint(expected_label, predicted_nli_label, loss.item(), {})

    def train(self, run_config):
        self._jade_logger.new_experiment()
        self._jade_logger.set_experiment_type('classification')
        self._jade_logger.set_total_epochs(run_config.epochs())
        for epoch in range(EPOCHS):
            self._jade_logger.new_epoch()
            self._epoch = epoch
            self.train_epoch()
            self.evaluate(run_config)

    def evaluate(self, run_config):
        self.evaluate_nli(run_config)

    def evaluate_nli(self, run_config):
        with torch.no_grad():
            test_sample = self._data_handler.test_data()
            self._jade_logger.new_evaluate_batch()
            for datumi, datum in enumerate(test_sample):
                predicted_label = []
                event_predicted_vector = self.classify(datum, 'nli', 'test')
                softmaxed = self._softmax(event_predicted_vector)

                relationship_target, target = self.temporal_target(datum)
                batch_loss = self._criterion(
                    event_predicted_vector,
                    relationship_target
                )
                loss = batch_loss.item()
                vector = softmaxed[0]
                for ii, i in enumerate(vector):
                    if i > 0.5:
                        predicted_label.append(idx2label[ii])

                self._jade_logger.new_evaluate_datapoint(
                    target,
                    predicted_label,
                    loss,
                    {
                        'matrix_tense': datum.matrix_tense(),
                        'matrix_aspect': datum.matrix_aspect(),
                        'sub_tense': datum.sub_tense(),
                        'sub_aspect': datum.sub_aspect(),
                        'is_quote': datum.is_quote(),
                        'temporal_marker': datum.temporal_marker(),
                    }
                )
