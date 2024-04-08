import numpy as np
import os
from torch import nn
import torch
from tqdm import tqdm
from torch.optim import Adam
from jadelogs import JadeLogger

from eventvec.server.tasks.factuality_estimator.models.factuality_estimator_model import FactualityEstimatorModel  # noqa
from eventvec.server.tasks.event_vectorization.datahandlers.data_handler_registry import DataHandlerRegistry
from eventvec.server.tasks.factuality_estimator.datahandlers.model_datahandler import FactualityRoBERTaDataHandler  # noqa
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer



TRAIN_SAMPLE_SIZE = int(8000 / 5)
TEST_SAMPLE_SIZE = 2000
EPOCHS = 40
LEARNING_RATE = 1e-5  # 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 10000000


class FactualityEstimationTrain:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._total_loss = 0
        self._all_losses = []
        self._data_handler_registry = DataHandlerRegistry()
        self._iteration = 0
        self._last_iteration = 0
        self._loss = None
        self._factuality_categorizer = FactualityCategorizer()

    def load(self, run_config):
        self._data_handler = FactualityRoBERTaDataHandler()
        self._data_handler.load()
        self._model = FactualityEstimatorModel(run_config)
        self._model_optimizer = Adam(
            self._model.parameters(),
            lr=LEARNING_RATE,
        )
        self._criterion = nn.MSELoss()

    def zero_grad(self):
        self._model.zero_grad()

    def optimizer_step(self):
        self._model_optimizer.step()

    def train_step(self, datum):
        event_predicted_vector = self.estimate(datum)
        relationship_target, org_target = self.relationship_target(datum)

        event_prediction_loss = self._criterion(
            event_predicted_vector, relationship_target
        )
        if self._loss is None:
            self._loss = event_prediction_loss
        else:
            self._loss += event_prediction_loss
        return event_prediction_loss, event_predicted_vector, org_target

    def relationship_target(self, datum):
        annotations = []
        for annotation in datum.annotations():
            annotations.append(annotation.value())
        mean = np.mean(annotations)
        org_target = mean
        relationship_target = np.array([mean])
        relationship_target = torch.from_numpy(relationship_target).to(device)
        relationship_target = relationship_target.unsqueeze(0)
        return relationship_target, org_target

    def estimate(self, datum):
        output = self._model(datum)
        return output

    def train_epoch(self):
        self.zero_grad()
        train_sample = self._data_handler.train_data()
        self._jade_logger.new_train_batch()
        for datum in tqdm(train_sample):
            loss, predicted_label, org_target = self.train_step(datum)
            self._all_losses += [loss.item()]
            self._iteration += 1
            if self._loss is not None and self._iteration % 1 == 0:
                self._loss.backward()
                self.optimizer_step()
                self.zero_grad()
                self._loss = None
            self._jade_logger.new_train_datapoint(org_target, predicted_label.item(), loss.item(), {})
        self._model.save()

    def train(self, run_config):
        self._jade_logger.new_experiment()
        self._jade_logger.set_experiment_type('regression')
        self._jade_logger.set_total_epochs(run_config.epochs())
        for epoch in range(EPOCHS):
            self._jade_logger.new_epoch()
            self._epoch = epoch
            self.train_epoch()
            self.evaluate(run_config)


    def evaluate(self, run_config):
        with torch.no_grad():
            test_sample = self._data_handler.test_data()
            self._jade_logger.new_evaluate_batch()
            for datumi, datum in enumerate(test_sample):
                sentence = datum.text()
                event_string = datum.event_string()

                features_array = self._factuality_categorizer.categorize(sentence, event_string)
                event_predicted_vector = self.estimate(datum)
                relationship_target, org_target = self.relationship_target(datum)
                batch_loss = self._criterion(
                    event_predicted_vector,
                    relationship_target
                )
                loss = batch_loss.item()
                self._jade_logger.new_evaluate_datapoint(org_target, event_predicted_vector.item(), loss, {'features_array': features_array.to_dict()})
