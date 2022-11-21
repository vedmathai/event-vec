from collections import defaultdict
import random


class ModelInputData:
    def __init__(self):
        self._train_data = []
        self._validation_data = []
        self._test_data = []
        self._classes = set()

    def train_datum(self, i):
        self._train_data[i]

    def train_data(self):
        return self._train_data

    def add_train_datum(self, datum):
        self._train_data.append(datum)

    def validation_datum(self, i):
        self._validation_data[i]

    def validation_data(self):
        return self._validation_data

    def add_validation_datum(self, datum):
        self._validation_data.append(datum)

    def test_datum(self, i):
        self._test_data[i]

    def test_data(self):
        return self._test_data

    def add_test_datum(self, datum):
        self._test_data.append(datum)

    def classes(self):
        return self._classes

    def num_classes(self):
        return len(self._classes)

    def add_class(self, label):
        self._classes.add(label)

    def set_classes(self, classes):
        self._classes = classes

    def set_train_data(self, train_data):
        self._train_data = train_data

    def set_test_data(self, test_data):
        self._test_data = test_data

    def sample_train_data(self, sample_number):
        return self._sample_data(self._train_data, sample_number, True)

    def sample_test_data(self, sample_number):
        return self._sample_data(self._test_data, sample_number, True)

    def _sample_data(self, original_data, sample_number, check_trainable):
        random.seed(10)
        data = defaultdict(list)
        for datum in original_data:
            if check_trainable is True and datum.is_trainable() is True:
                data[datum.target()].append(datum)
            if check_trainable is False:
                data[datum.target()].append(datum)
        for target in data:
            data[target] = random.choices(data[target], k=sample_number)
        data = sum([list(i) for i in data.values()], [])
        random.shuffle(data)
        return data
