import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
from collections import defaultdict
from jadelogs import JadeLogger
import json
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from eventvec.server.config import Config
from eventvec.server.tasks.subordinate.datareader.subordinate_datareader import SubordinateTemporalDatareader

aspects = ['perfect', 'simple', 'continuous', 'perfect-continuous']
tenses = ['past', 'present', 'future']
is_quote = ['yes', 'no']
temporal_marker = ['no_marker', 'yesterday', 'today', 'tomorrow', 'now', 'everyday']


class GPTAnalyse():
    def __init__(self):
        self._data_readers = {
            'subordinate': SubordinateTemporalDatareader(),
        }


    def load(self):
        k = 0
        self._jl = JadeLogger()
        data_reader = self._data_readers['subordinate']
        data = data_reader.data()[:4800]
        files = [
            #'subordinate/gpt_subordinate_plain_dct_is_matrix.json',
            #'subordinate/gpt_subordinate_plain_dct_is_sub.json',
            #'subordinate/gpt_subordinate_plain_matrix_is_sub.json',
            #'subordinate/llama_405B_subordinate_plain_dct_is_sub.json',
            'subordinate/llama_405B_subordinate_plain_matrix_is_sub.json',
            #'subordinate/llama_405B_subordinate_plain_dct_is_matrix.json',
        ]
        uid2data = {}

        for d in data:
            uid2data[d.key().lower()] = d

        for filename in files:
            print(filename)

            location = self._jl.file_manager.data_filepath(filename)
            data_len = 0
            with open(location, 'rt') as f:
                gpt_answer = json.loads(f.read())
                true_answers = {}
                new_gpt_answer = {}
                for feature_1 in temporal_marker:
                    for feature_2 in tenses:
                        for d in data:
                            if d.key() in gpt_answer:
                                if d.temporal_marker().lower() == feature_1 and d.sub_tense().lower() == feature_2:
                                    #new_gpt_answer[d.key()] = ['x', random.choice(['before', 'after', 'during', 'iduring'])] #gpt_answer[d.key()]
                                    new_gpt_answer[d.key()] = gpt_answer[d.key()]
                                    true_answers[d.key()] = d.matrix_is_sub()
                                    data_len += 1

                        f1_score = self.f1_score(true_answers, new_gpt_answer)
                        print(' ' * 4, feature_1, feature_2, '{:.3f}'.format(f1_score))
        
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
        f1s = []

        for uid, label in true_answers.items():
            tp = defaultdict(int)
            fp = defaultdict(int)
            fn = defaultdict(int)
            if uid not in gpt_answers:
                continue
            #if gpt_answers[uid][1] == '':
            #    fn[] += 1
            gpt_answer = gpt_answers[uid][1]
            new_gpt_answer = []
            if 'after' in gpt_answer:
                new_gpt_answer += ['after']
            if 'before' in gpt_answer:
                new_gpt_answer += ['before']
            if 'during' in gpt_answer:
                new_gpt_answer += ['during']
            for i in new_gpt_answer:
                if i in label:
                    tp[i] += 1
                else:
                    fp[i] += 1
            for i in label:
                if i not in new_gpt_answer:
                    fn[i] += 1
        
            sub_f1 = []
            for key in ['after', 'before', 'during']:
                if key in tp or key in fp or key in fn:
                    precision = 0
                    recall = 0
                    if tp[key] + fp[key] != 0:
                        precision = tp[key] / (tp[key] + fp[key])
                        #print(key, precision)
                    if tp[key] + fn[key] != 0:
                        recall = tp[key] / (tp[key] + fn[key])
                        #print(key, recall)
                    if precision + recall != 0:
                        sub_f1 += [2 * (precision * recall) / (precision + recall)]
                    else:
                        sub_f1 += [0]
            if len(sub_f1) > 0:
                f1s.append(np.mean(sub_f1))
        return np.mean(f1s)

if __name__ == '__main__':
    Config.instance()
    data_preparer = GPTAnalyse()
    data_preparer.load()