import numpy as np
import os
from torch import nn
import torch
from tqdm import tqdm
from torch.optim import Adam
from jadelogs import JadeLogger


from eventvec.server.tasks.entailment_classification.models.nli_classifier_model import NLIClassifierModel  # noqa
from eventvec.server.tasks.tense_classification.models.llm_tense_pretrain_model import LLMTensePretrainer  # noqa
from eventvec.server.tasks.event_vectorization.datahandlers.data_handler_registry import DataHandlerRegistry
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer


TRAIN_SAMPLE_SIZE = int(8000 / 5)
TEST_SAMPLE_SIZE = 2000
EPOCHS = 60
LEARNING_RATE = 1e-6  # 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 10000000

labels2idx = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
}

idx2label = {
    v: k for k, v in labels2idx.items()
}

class NLIClassificationTrain:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._data_handler_registry = DataHandlerRegistry()
        self._total_loss = 0
        self._all_losses = []
        self._iteration = 0
        self._last_iteration = 0
        self._loss = None
        self._factuality_categorizer = FactualityCategorizer()

    def load(self, run_config):
        data_handler = 'nli_datahandler'
        self._data_handler = self._data_handler_registry.get_data_handler(data_handler)
        self._data_handler.load(run_config)
        self._model = NLIClassifierModel(run_config)
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
        event_predicted_vector, event_string_1, event_string_2, entropy = self.classify(datum, 'train')
        relationship_target = self.relationship_target(datum)
        event_prediction_loss = self._criterion(
            event_predicted_vector, relationship_target
        )
        predicted = event_predicted_vector.argmax(dim=1).item()
        predicted_label = idx2label[predicted]
        if self._loss is None:
            self._loss = event_prediction_loss
        else:
            self._loss += event_prediction_loss
        return event_prediction_loss, predicted_label, entropy

    def relationship_target(self, datum):
        relationship_target = np.array([0 for i in range(3)]).astype(float)
        target = datum.label()
        label_idx = labels2idx[target]
        relationship_target[label_idx] = 1
        relationship_target = torch.from_numpy(relationship_target).to(device)
        relationship_target = relationship_target.unsqueeze(0)
        return relationship_target

    def classify(self, datum, train_test):
        output, event_string_1, event_string_2, entropy = self._model(datum, train_test)
        return output, event_string_1, event_string_2, entropy

    def train_epoch(self):
        self.zero_grad()
        train_sample = self._data_handler.train_data()
        self._jade_logger.new_train_batch()
        for datum in tqdm(train_sample):
            if datum.label() not in labels2idx:
                continue
            loss, predicted_label, entropy = self.train_step(datum)
            self._all_losses += [loss.item()]
            self._iteration += 1
            if self._loss is not None and self._iteration % 10 == 0:
                self._loss.backward()
                self.optimizer_step()
                self.zero_grad()
                self._loss = None
            self._jade_logger.new_train_datapoint(datum.label(), predicted_label, loss.item(), {})
            

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
            test_sample = self._data_handler.test_data()
            self._jade_logger.new_evaluate_batch()
            for datumi, datum in enumerate(test_sample):
                event_predicted_vector, event_string_1, event_string_2, entropy = self.classify(datum, 'test')
                features_array_1 = self._factuality_categorizer.categorize(datum.sentence_1(), event_string_1)
                features_array_2 = self._factuality_categorizer.categorize(datum.sentence_2(), event_string_2)

                relationship_target = self.relationship_target(datum)
                batch_loss = self._criterion(
                    event_predicted_vector,
                    relationship_target
                )
                loss = batch_loss.item()
                predicted = event_predicted_vector.argmax(dim=1).item()
                predicted_label = idx2label[predicted]

                self._jade_logger.new_evaluate_datapoint(
                    datum.label(),
                    predicted_label,
                    loss,
                    {
                        'entropy': datum.entropy(),
                        'predicted_distribution': event_predicted_vector.tolist()[0],
                        'distribution': datum.label_dist(),
                        'features_array': features_array_1.to_dict(),
                        'features_array_2': features_array_2.to_dict(),
                        'model_attention_entropy': float(entropy),
                        'uid': datum.uid(),
                    }
                )