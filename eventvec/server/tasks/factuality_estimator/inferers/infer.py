import numpy as np
import os
from torch import nn
import torch
from torch.optim import Adam

from eventvec.server.config import Config
from eventvec.server.tasks.factuality_estimator.models.factuality_estimator_model import FactualityEstimatorModel  # noqa
from eventvec.server.tasks.factuality_estimator.datahandlers.model_datahandler import FactualityRoBERTaDataHandler  # noqa
from eventvec.server.data.factuality.factuality_datamodels.belief_datum import BeliefDatum


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FactualityClassificationInfer:
    def __init__(self):
        self._config = Config.instance()
        self._data_handler = FactualityRoBERTaDataHandler()

    def load(self, run_config={}):
        self._data_handler.load()
        self._model = FactualityEstimatorModel(run_config)
        self._model.load()

    def estimate(self, datum):
        output = self._model(datum)
        return output

    def infer(self, sentence, event_string):
        with torch.no_grad():
            datum = BeliefDatum()
            datum.set_event_string(event_string)
            datum.set_sentence(sentence)
            event_predicted_vector = self.estimate(datum)
            value = event_predicted_vector.item()
            return value


if __name__ == '__main__':
    fci = FactualityClassificationInfer()
    fci.load()
    label = fci.infer('I am not sure if George went back to school', 'went')
    print(label)
