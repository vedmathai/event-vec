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

from eventvec.server.config import Config
from eventvec.server.tasks.event_ordering_nli.datareader.temporal_datareader import TemporalDatareader

label_map = {
    'True': 'True',
    'False': 'False',
    'Impossible': 'Contradictory'
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
        data = data_reader.data('test')[:4800]
        files = [
            'temporal/llama_3_temporal_70b_helped_1.json',
            'temporal/llama_3_temporal_70b_plain_1.json',
            'temporal/llama_3_temporal_70b_plain_2.json',
            'temporal/llama_3_temporal_70b_plain_3.json',


            #'temporal/llama_3_temporal_70b_plain_6.json',
            #'temporal/llama_3_temporal_70b_helped_1.json',
            #'temporal/llama_3_temporal_simple_70b_1.json',
        ]
        uid2data = {}

        for filename in files:
            print(filename)

            true_answers = {}
            location = self._jl.file_manager.data_filepath(filename)

            with open(location, 'rt') as f:
                gpt_answer = json.loads(f.read())
                print(len(gpt_answer))
                for features in [lambda x: x < 1, lambda x: x==1, lambda x: x>1]:
                    for d in data:
                        if True: ####features(d._relationship_number / d._event_number):
                            uid2data[d.uid()] = d
                            true_answers[d.uid()] = label_map.get(d.label())
                            
                    f1_score = self.f1_score(true_answers, gpt_answer)

                    print(' ' * 4, features, '{:.3f}'.format(f1_score))
        
    def print_confusion(self, file2correct, uid2data, gpt_answers):
        credence_only = file2correct['gpt_answers_credence_output_2.json'] & file2correct['gpt_answers_credence_output.json'] - (file2correct['gpt_answers_few_shot.json'] | file2correct['gpt_answers_few_shot_2.json'])
        non_credence_only = (file2correct['gpt_answers_few_shot.json'] & file2correct['gpt_answers_few_shot_2.json']) - (file2correct['gpt_answers_credence_output_2.json'] | file2correct['gpt_answers_credence_output.json'])
        both_correct = (file2correct['gpt_answers_few_shot.json'] & file2correct['gpt_answers_few_shot_2.json']) & (file2correct['gpt_answers_credence_output_2.json'] & file2correct['gpt_answers_credence_output.json'])
        both_wrong = set(uid2data.keys()) - (file2correct['gpt_answers_few_shot.json'] | file2correct['gpt_answers_few_shot_2.json'] | file2correct['gpt_answers_credence_output_2.json'] | file2correct['gpt_answers_credence_output.json'])
        classes = {
            'credence_only': credence_only,
            'non_credence_only': non_credence_only,
            'both_correct': both_correct,
            'both_wrong': both_wrong
        }
        confusion_data = [['uid', 'class', 'premise', 'hypothesis', 'true_label', 'gpt_answer_few_shot', 'gpt_answer_credence']]
        for key, value in classes.items():
            for uidi, uid in enumerate(value):
                if uidi > min(max(len(credence_only), len(non_credence_only)), len(both_correct), len(both_wrong)):
                    break
                confusion_data.append([
                    uid,
                    key,
                    uid2data[uid].sentence_1(),
                    uid2data[uid].sentence_2(),
                    uid2data[uid].label()[0],
                    gpt_answers['gpt_answers_few_shot.json'].get(uid),
                    gpt_answers['gpt_answers_credence_output_2.json'].get(uid),
                ])
        with open(self._jl.file_manager.data_filepath('gpt_confusion_data.csv'), 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(confusion_data[0])
            for row in sorted(confusion_data[1:], key=lambda x: (x[1], x[4], str([5]), str(x[6]))):
                writer.writerow(row)

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
            
            gpt_answer = gpt_answers[uid][0]
            if label is None:
                continue
            if label.lower() in gpt_answer.lower():
                tp[gpt_answer] += 1
            else:
                fp[label] += 1
                fn[gpt_answer] += 1
        for key in ['Contradictory', 'True', 'False']:
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