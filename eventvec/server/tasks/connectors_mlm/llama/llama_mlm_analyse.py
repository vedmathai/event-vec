import re
import csv
from jadelogs import JadeLogger
import random
import numpy as np
from collections import defaultdict
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ConnectorMLMDatum:
    def __init__(self):
        self._uid = None
        self._text = None
        self._label = None


connectors = ['and', 'because', 'but', 'so', 'though']
class LlamaMLMAnalysis:
    def __init__(self):
        self._jade_logger = JadeLogger()
        self._filepath = self._jade_logger.file_manager.data_filepath('llama_mlm.csv')
        self._results_file_helped = '/home/lalady6977/oerc/projects/data/roc_masked_connectors_llama_helped.csv'
        self._results_file_plain = '/home/lalady6977/oerc/projects/data/roc_masked_connectors_llama_base.csv'

    def read_results(self, filename):
        llm_answers = {}
        with open(filename, 'rt') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                llm_answers[row[0]] = row[1]
        return llm_answers

    def read_data(self):
        data = {}
        with open('/home/lalady6977/oerc/projects/data/roc_masked_connectors.csv') as f:
            reader = csv.reader(f, delimiter='\t')
            for r in reader:
                datum = ConnectorMLMDatum()
                datum._uid = r[0]
                datum._label = r[1]
                datum._text = r[2].replace('[MASK]', '<mask>')
                data[datum._uid] = datum
        return data
    
    def mask(self, train_test='train'):
        required_answers = {}
        data = self.read_data()
        confusion_plain = defaultdict(lambda: defaultdict(list))
        confusion_helped = defaultdict(lambda: defaultdict(list))



        for datum_i, datum in enumerate(data.values()):
            required_answers[datum._uid] = datum._label
            llm_answers_plain = self.read_results(self._results_file_plain)
            llm_answers_helped = self.read_results(self._results_file_helped)
            self.check_metric(llm_answers_plain, required_answers)
            f1s = []

        y_test = []
        predictions = []
        classes = ['and', 'but', 'though', 'so', 'because', 'none']
        for key in llm_answers_helped.keys():
            if key in required_answers:
                y_test.append(required_answers[key])
                used = False
                for c in connectors:
                    if c in llm_answers_helped[key].split():
                        confusion_helped[required_answers[key]][c].append(key)
                        used = True
                        predictions.append(c)
                if not used:
                    confusion_helped[required_answers[key]]['none'].append(key)
                    predictions.append('none')

        for key in confusion_helped['because']['but']:
            print()
            print(key, data[key]._text)

        cm = confusion_matrix(y_test, predictions, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        plt.savefig('/home/lalady6977/Downloads/confusion_llama_mlm_helped.png', bbox_inches='tight')


    def check_metric(self, llm_answers, required_answers):
        true_p = {'and': 0, 'but': 0, 'though':0, 'so': 0, 'because': 0}
        false_n = {'and': 0, 'but': 0, 'though':0, 'so': 0, 'because': 0}
        false_p = {'and': 0, 'but': 0, 'though':0, 'so': 0, 'because': 0}

        for key in llm_answers.keys():
            if key in required_answers:
                if required_answers[key] == 'but':
                    if 'but' in llm_answers[key].split():
                        true_p['but'] += 1
                    else:
                        false_n['but'] += 1
                        for c in connectors:
                            if c in llm_answers[key].split():
                                false_p[c] += 1

                if required_answers[key] == 'and':
                    if 'and' in llm_answers[key].split():
                        true_p['and'] += 1
                    else:
                        false_n['and'] += 1
                        for c in connectors:
                            if c in llm_answers[key].split():
                                false_p[c] += 1

                if required_answers[key] == 'because':
                    if 'because' in llm_answers[key].split():
                        true_p['because'] += 1
                    else:
                        false_n['because'] += 1
                        for c in connectors:
                            if c in llm_answers[key].split():
                                false_p[c] += 1

                if required_answers[key] in ['so', 'therefore']:
                    if any(i in llm_answers[key].split() for i in ['so', 'therefore']):
                        true_p['so'] += 1
                    else:
                        false_n['so'] += 1
                        for c in connectors:
                            if c in llm_answers[key].split():
                                false_p[c] += 1

                if required_answers[key] in ['though']:
                    if 'though' in llm_answers[key].split():
                        true_p['though'] += 1
                    else:
                        false_n['though'] += 1
                        for c in connectors:
                            if c in llm_answers[key].split():
                                false_p[c] += 1
        f1s = []
        for key in connectors:
            f1 = 0
            precision = 0
            if true_p[key] + false_p[key] > 0:
                precision = true_p[key] / (true_p[key] + false_p[key])
            recall = 0
            if true_p[key] + false_n[key] > 0:
                recall = true_p[key] / (true_p[key] + false_n[key])
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            f1s.append(f1)
            print(key, precision, recall, f1)
        print(np.mean(f1s))


if __name__ == '__main__':
    LlamaMLMAnalysis().mask('train')