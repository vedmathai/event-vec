import numpy as np
import os
from torch import nn
import torch
from tqdm import tqdm
from torch.optim import Adam
from jadelogs import JadeLogger


from eventvec.server.tasks.relationship_classification.models.relationship_classifier_model import RelationshipClassifierModel  # noqa
from eventvec.server.tasks.tense_classification.models.llm_tense_pretrain_model import LLMTensePretrainer  # noqa
from eventvec.server.tasks.event_vectorization.datahandlers.data_handler_registry import DataHandlerRegistry



TRAIN_SAMPLE_SIZE = int(8000 / 5)
TEST_SAMPLE_SIZE = 2000
EPOCHS = 40
LEARNING_RATE = 1e-3  # 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 10000000

labels_simpler = {
    'before': 0,
    'includes': 1,
    'is_included': 2,
    'during': 3,
    'after': 4,
    'none': 5,
    'modal': 5,
    'counter_factive': 5,
    'vague': 5,
    'factive': 5,
    'evidential': 5,
    'conditional': 5,
    'neg_evidential': 5,
    'factual': 5,
}

labels_reverse_map = {v: k for k, v in labels_simpler.items()}

class RelationshipClassificationTrain:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._total_loss = 0
        self._all_losses = []
        self._data_handler_registry = DataHandlerRegistry()
        self._iteration = 0
        self._last_iteration = 0
        self._loss = None

    def load(self, run_config):
        data_handler = 'bert_data_handler'
        self._data_handler = self._data_handler_registry.get_data_handler(data_handler)
        self._data_handler.load()
        self._input_data = self._data_handler.model_input_data()
        self._model = RelationshipClassifierModel(run_config)
        self._model_optimizer = Adam(
            self._model.parameters(),
            lr=LEARNING_RATE,
        )
        self._criterion = nn.CrossEntropyLoss()

    def zero_grad(self):
        self._model.zero_grad()

    def optimizer_step(self):
        self._model_optimizer.step()

    def train_step(self, datum):
        event_predicted_vector = self.classify(datum)
        relationship_target = self.relationship_target(datum)
        event_prediction_loss = self._criterion(
            event_predicted_vector, relationship_target
        )
        predicted = event_predicted_vector.argmax(dim=1).item()
        predicted_label = labels_reverse_map[predicted]
        if self._loss is None:
            self._loss = event_prediction_loss
        else:
            self._loss += event_prediction_loss
        return event_prediction_loss, predicted_label

    def relationship_target(self, datum):
        relationship_target = np.array([0 for i in range(6)]).astype(float)
        target = labels_simpler[datum.relationship()]
        relationship_target[target] = 1
        relationship_target = torch.from_numpy(relationship_target).to(device)
        relationship_target = relationship_target.unsqueeze(0)
        return relationship_target

    def classify(self, datum):
        output = self._model(datum)
        return output

    def train_epoch(self):
        self.zero_grad()
        train_sample = self._input_data.sample_train_data(TRAIN_SAMPLE_SIZE)
        self._jade_logger.new_train_batch()
        for datum in tqdm(train_sample):
            loss, predicted_label = self.train_step(datum)
            self._all_losses += [loss.item()]
            self._iteration += 1
            if self._loss is not None and self._iteration % 10 == 0:
                self._loss.backward()
                self.optimizer_step()
                self.zero_grad()
                self._loss = None
            self._jade_logger.new_train_datapoint(datum.relationship(), predicted_label, loss.item(), {"use": datum.is_interested()})
            

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
        with torch.no_grad():
            test_sample = self._input_data.sample_test_data(TEST_SAMPLE_SIZE)
            self._jade_logger.new_evaluate_batch()
            for datumi, datum in enumerate(test_sample):
                event_predicted_vector = self.classify(datum)
                relationship_target = self.relationship_target(datum)
                batch_loss = self._criterion(
                    event_predicted_vector,
                    relationship_target
                )
                loss = batch_loss.item()
                predicted = event_predicted_vector.argmax(dim=1).item()
                predicted_label = labels_reverse_map[predicted]
                self._jade_logger.new_evaluate_datapoint(datum.relationship(), predicted_label, loss, {"use": datum.is_interested()})
