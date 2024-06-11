import numpy as np
import os
from torch import nn
import torch
from tqdm import tqdm
from torch.optim import Adam
from jadelogs import JadeLogger
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from collections import defaultdict

from eventvec.server.tasks.event_vectorization.datahandlers.data_handler_registry import DataHandlerRegistry
from eventvec.server.tasks.factuality_estimator.datahandlers.model_datahandler import FactualityRoBERTaDataHandler  # noqa
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer

categories = [
    'is_negated',
    'has_modal',
    'is_subordinate_of_said',
    'is_subordinate_of_believe',
    'is_subordinate_of_if',
    'is_speech_act',
    'is_belief_act',
    'is_subordinate_of_expects',
    'is_subordinate_of_then',
    'has_modal_adjective',
    'has_negation_words',
    'is_infinive_sub_neg',
]

class FactualityEstimationTrainML:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._total_loss = 0
        self._all_losses = []
        self._data_handler_registry = DataHandlerRegistry()
        self._factuality_categorizer = FactualityCategorizer()
        self._iteration = 0
        self._last_iteration = 0
        self._loss = None
        self._regr = MLPRegressor(random_state=1, max_iter=6)
        #self._regr = RandomForestRegressor(max_depth=20, random_state=0)
        self._model_location = self._jade_logger.file_manager.data_filepath('ml_factuality_model.pkl')
        self._counter = 0

    def load(self, run_config):
        self._data_handler = FactualityRoBERTaDataHandler()
        self._data_handler.load()

    def fix_datum(self, datum):
        sentence = datum.text()
        event_string = datum.event_string()
        category = self._factuality_categorizer.categorize(sentence, event_string)
        if category.has_modal_adjective() is True:
            self._counter += 1
        x = []
        for key, value in sorted(category.to_dict().items(), key=lambda x: x[0]):
            if value is True:
                x.append(1)
            else:
                x.append(0)
        return x, category

    def relationship_target(self, datum):
        annotations = []
        for annotation in datum.annotations():
            annotations.append(annotation.value())
        mean = np.mean(annotations)
        target = mean
        return target

    def estimate(self, datum):
        output = self._model(datum)
        return output

    def train_epoch(self):
        self._counter = 0
        train_sample = self._data_handler.train_data()
        self._jade_logger.new_train_batch()
        X = []
        Y = []
        for datum in tqdm(train_sample):
            x, category = self.fix_datum(datum)
            y = self.relationship_target(datum)
            X.append(x)
            Y.append(y)
        print('counter is', self._counter)
        self._regr.fit(X, Y)
        pickle.dump(self._regr, open(self._model_location, 'wb'))

    def train(self, run_config):
        self._jade_logger.new_experiment()
        self._jade_logger.set_experiment_type('regression')
        self._jade_logger.set_total_epochs(run_config.epochs())
        #for epoch in range(EPOCHS):
        self._jade_logger.new_epoch()
        #    self._epoch = epoch
        self.train_epoch()
        self.dev(run_config)
        self.evaluate(run_config)

    def dev(self, run_config):
        test_sample = self._data_handler.dev_data()
        self._jade_logger.new_evaluate_batch()
        X = []
        Y = []
        counter = defaultdict(int)
        for datum in test_sample:
            x, category = self.fix_datum(datum)
            y = self.relationship_target(datum)
            X.append(x)
            Y.append(y)
            prediction = self._regr.predict([x])
            prediction = prediction[0]
            #if diff > 2:
            #    continue
            diff = prediction * y / abs(prediction * y)
            counter[diff] += 1
            if 2 <= diff <= 2.5:
                print('{} | event: {} | prediction: {} | label: {} | {}'.format(datum.text(), datum.event_string(), prediction, y, category.to_dict()))
                print()
                print('\n' + '-' * 80 + '\n')
        x = self._regr.predict(X)
        mse = mean_squared_error(Y, x)
        print(mse)
        print(counter)


    def evaluate(self, run_config):
        test_sample = self._data_handler.dev_data() + self._data_handler.test_data()
        self._jade_logger.new_evaluate_batch()

        counter = 0
        for category in categories:
            X = []
            Y = []
            print(category)
            for datum in tqdm(test_sample):
                x, category_obj = self.fix_datum(datum)
                y = self.relationship_target(datum)
                if category_obj.to_dict()[category] is True:
                    X.append(x)
                    Y.append(y)
                counter += 1
            print(counter)
            x = self._regr.predict(X)
            mae = mean_absolute_error(Y, x)
            mse = mean_squared_error(Y, x)
            print('dev_test_mse', mse)
            print('dev_test_mae', mae)
            print(counter)
