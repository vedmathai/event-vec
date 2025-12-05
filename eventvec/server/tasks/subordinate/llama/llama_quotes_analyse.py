import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
from collections import defaultdict
from jadelogs import JadeLogger
import json
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from eventvec.server.config import Config
from eventvec.server.tasks.subordinate.datareader.datareader import SubordinateTemporalDatareader


class GPTAnalyse():
    def __init__(self):
        self._data_readers = {
            'subordinate': SubordinateTemporalDatareader(),
        }

    def read_data(self):
        with open('/home/lalady6977/oerc/projects/local_jade/jade_front/event-vec/data/temporal_subordinate/pronoun_subordinate.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            data = []
            for ri, r in enumerate(reader):
                if r[0] != '' and ri > 0:
                    data.append(r)
        return data

    def load(self):
        k = 0
        self._jl = JadeLogger()
        data_reader = self._data_readers['subordinate']
        files = [
            'subordinate/pronoun_subordinate_gpt_oss.json',
            #'subordinate/pronoun_subordinate_gpt5_2.json',
        ]
        uid2data = {}
        data = self.read_data()
        for d in data:
            key = '_'.join(d[0:5])
            uid2data[key.lower()] = d
        for filenamei, filename in enumerate(files):
            print(filename)
            location = self._jl.file_manager.data_filepath(filename)
            with open(location, 'rt') as f:
                gpt_answer = json.loads(f.read())
                for feature in ['direct', 'indirect']:
                    data_len = 0
                    new_gpt_answer = {}
                    true_answers = {}
                    for d in gpt_answer:
                        if uid2data[d.lower()][4].strip() == feature.strip():
                            data_len += 1
                            dp = uid2data[d.lower()]
                            direct_indirect = dp[4].strip()
                            indirect_key = ('_'.join(dp[0:4] + ['indirect'])).lower()
                            direct_key = ('_'.join(dp[0:4] + ['direct'])).lower()
                            
                            new_gpt_answer[d] = gpt_answer[d]
                            if direct_indirect == 'indirect':
                                if False and gpt_answer[d][1].strip().strip('.').lower() != uid2data[indirect_key][7].strip().lower():
                                    true_answers[d] = uid2data[direct_key][7].strip().lower()
                                else:
                                    true_answers[d] = uid2data[indirect_key][7].strip().lower()       
                            else:
                                true_answers[d] = uid2data[direct_key][7].strip().lower()

                    
                    f1_score = self.f1_score(feature, uid2data, true_answers, new_gpt_answer)
                    print(' ' * 4, '{:.3f}'.format(f1_score), feature, data_len)
        

    def f1_score(self, feature, uid2data, true_answers, gpt_answers):
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        total = 0
        match = 0
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        for uid, label in true_answers.items():
            
            if uid not in gpt_answers:
                continue
            total += 1
            gpt_answer = gpt_answers[uid][1].lower().strip().strip('.')
            if gpt_answer == label.strip():
                tp[gpt_answer] += 1
                match += 1
            else:
                fp[gpt_answer] += 1
                fn[label] += 1
                print(uid2data[uid.lower()][5], '&', uid2data[uid.lower()][6], '&', gpt_answer, '&', label, '\\\\')
            confusion_matrix[label][gpt_answer] += 1
        print(confusion_matrix)
        
        sub_f1 = []
        for key in ['alice', 'speaker a', 'listener b', 'unknown third person']:
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
        matrix = []
        keys = ['alice', 'speaker a', 'listener b', 'unknown third person']
        for key1 in keys:
            row = []
            for key2 in keys:
                row += [confusion_matrix[key1][key2]]
            matrix += [row]
        matrix = np.array(matrix)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=keys)
        disp.plot(xticks_rotation='vertical')
        plt.savefig(feature, bbox_inches='tight')
        return np.mean(sub_f1)

if __name__ == '__main__':
    Config.instance()
    data_preparer = GPTAnalyse()
    data_preparer.load()