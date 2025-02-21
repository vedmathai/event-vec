import numpy as np
from collections import defaultdict
from transformers import BertTokenizer, RobertaTokenizer
from collections import defaultdict
from jadelogs import JadeLogger
import json
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import matplotlib

from eventvec.server.config import Config
from eventvec.server.tasks.event_ordering_nli.datareader.temporal_datareader import TemporalDatareader

label_map = {
    'True': 'True',
    'False': 'False',
    'Impossible': 'Undefined',
    'Contradictory': 'Undefined',
    'Undefined': 'Undefined',
    'Contrastive': 'Undefined',
}

class GPTAnalyse():
    def __init__(self):
        self._data_readers = {
            'temporal': TemporalDatareader(),
        } 


    def load(self):
        k = 0
        self._jl = JadeLogger()
        data_reader = self._data_readers['temporal']
        data = data_reader.data('temporal_nli_test')[:4800]
        files = [
            #'temporal/gpt_temporal_helped_1.json',
            #'temporal/gpt_temporal_8b_plain_1.json',
            #'temporal/gpt_spatial_1.json',
            #'temporal/llama_104B_temporal.json',
            #'temporal/gpt_before_only_ordered.json',
            #'temporal/gpt_temporal_nli_test.json',
            'temporal/gpt_temporal_plain_1.json',

            #'temporal/llama_3_temporal_70b_plain_6.json',
            #'temporal/llama_3_temporal_70b_helped_1.json',
            #'temporal/llama_3_temporal_simple_70b_1.json',
        ]

        for filename in files:
            print(filename)

            true_answers = {}
            location = self._jl.file_manager.data_filepath(filename)

            with open(location, 'rt') as f:
                gpt_answer = json.loads(f.read())
                print(len(gpt_answer))
                for features in [1]:
                    uid2data = {}
                    for d in data:
                        if d.relationship_number() / d.event_number() < 1:
                            uid2data[d.uid()] = label_map.get(d)
                            true_answers[d.uid()] = label_map.get(d.label())
                            
                    f1_score = self.f1_score(true_answers, gpt_answer)

                    print(' ' * 4, features, '{:.3f}'.format(f1_score))
            self.print_confusion(true_answers, gpt_answer)
        
    def print_confusion(self, true_answers, gpt_answers):
        y_test = []
        predictions = []
        for uid, label in true_answers.items():
            if uid not in gpt_answers:
                continue
            mapped = {'True': 'True', 'False': 'False', 'Contradictory': 'Undefined'}
            if label in mapped and gpt_answers[uid][0] in mapped:
                y_test.append(mapped[label])
                predictions.append(mapped[gpt_answers[uid][0]])
        classes = ['True', 'False', 'Undefined']
        cm = confusion_matrix(y_test, predictions, labels=classes)
        matplotlib.rcParams.update({'font.size': 18})
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        plt.savefig('/home/lalady6977/Downloads/confusion_llama_gpt.png', bbox_inches='tight')

    def f1_score(self, true_answers, gpt_answers):
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        f1s = []
        for uid, label in true_answers.items():
            if uid not in gpt_answers:

                continue
            if gpt_answers[uid][0] == '':
                fn[label] += 1
            
            gpt_answer = label_map[gpt_answers[uid][0]]
            if label is None:
                continue
            if label.lower() in gpt_answer.lower():
                tp[gpt_answer] += 1
            else:
                fp[label] += 1
                fn[gpt_answer] += 1

        for key in ['Undefined', 'True', 'False']:
            f1 = 0
            precision = 0
            recall = 0
            if tp[key] + fp[key] != 0:
                precision = tp[key] / (tp[key] + fp[key])
                #print(key, precision)
            if tp[key] + fn[key] != 0:
                recall = tp[key] / (tp[key] + fn[key])
                #print(key, recall)
            if precision + recall != 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                #print(key, f1)
            f1s.append(f1)
        return np.mean(f1s)

if __name__ == '__main__':
    Config.instance()
    data_preparer = GPTAnalyse()
    data_preparer.load()