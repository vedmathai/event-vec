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

aspects = ['perfect', 'simple', 'continuous', 'perfect-continuous']
tenses = ['past', 'present', 'future']
is_quote = ['yes', 'no']
temporal_marker = ['no_marker', 'yesterday', 'today', 'tomorrow', 'now', 'everyday']

aspect_mapper = {
    'perfect': 'Perf',
    'continuous': 'Prog',
    'simple': 'Simple',
    'perfect-continuous': 'Prog-Perf'
}

tense_mapper = {
    'past': 'Past',
    'present': 'Pres',
    'future': 'Future',
}

# Inter model aggreement [Done]
# GPT-5 vs GPT-open vs Llama vs Qwen [Done]
# NLI setup, QA setup, Missing word setup [Done last 2]
# Reasoning of the model chain of thought matches our reasoning [Done]
# try to find a linguistics resource for each phenomenon [Done]
# Qualitative analysis of the errors [Done]
# ate versus fasted versus jogged versus dative [Done]
# it is easier to see matrix-sub in quotes cause the tense relative to only matrix with quotes 
# it is harder to see dct-sub in quotes
#John has been saying that Mary would eat yesterday. versus John had been saying that Mary would eat yesterday.
# Habitual versus single event (graduated vs eaten vs fasted vs jogged) [Done]
# John will say 'Mary ate yesterday.' It is unknown when exactly. But John will say, 'Mary will eat tomorrow'

class GPTAnalyse():
    def __init__(self):
        self._data_readers = {
            'subordinate': SubordinateTemporalDatareader(),
        }


    def load(self):
        k = 0
        self._jl = JadeLogger()
        data_reader = self._data_readers['subordinate']
        data = data_reader.data('temporal_subordinate_said')[:4800]
        files = [
            #'subordinate/gpt_subordinate_plain_dct_is_matrix.json',
            #'subordinate/gpt_subordinate_plain_dct_is_sub.json',
            #'subordinate/gpt_subordinate_plain_matrix_is_sub.json',
            #'subordinate/llama_405B_subordinate_plain_dct_is_sub.json',
            #'subordinate/llama_405B_subordinate_plain_matrix_is_sub.json',
            #'subordinate/llama_405B_subordinate_plain_dct_is_matrix.json',
            #'subordinate/gpt_subordinate_question_plain_matrix_is_sub.json',
            #'subordinate/gpt_oss_subordinate_question_plain_matrix_is_sub.json',
            'subordinate/gpt_oss_subordinate_question_plain_dct_is_sub.json',
            'subordinate/qwen3_said_question_dct_is_sub.json',


            'subordinate/gpt_oss_subordinate_question_plain_dct_is_sub_2.json',
            'subordinate/gpt_5_said_cloze_dct_is_sub.json',

            'subordinate/gpt_5_subordinate_question_plain_dct_is_sub.json',
            'subordinate/gpt5_retiring_question_dct_is_sub.json',
            
            #'subordinate/llama70_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/llama8_subordinate_question_plain_dct_is_sub.json',
            'subordinate/deepseek_subordinate_question_plain_dct_is_sub.json',

            #'subordinate/gpt_open_jogged_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/gpt_open_fasting_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/gpt_open_graduating_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/gpt_open_retiring_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/gpt_open_retiring_subordinate_question_plain_dct_is_matrix.json',
            'subordinate/gpt_open_said_cloze_dct_is_sub.json',

            'subordinate/temporal_subordinate_jogging_unembedded_2.json',
            'subordinate/gpt_5_temporal_subordinate_jogging_unembedded.json',
            'subordinate/qwen3_said_question_dct_is_sub.json',
            'subordinate/qwen3_said_cloze_dct_is_sub.json',
            'subordinate/deepseek_said_cloze_dct_is_sub.json',
            #'subordinate/llama_70_said_cloze_dct_is_sub.json',
            #'subordinate/llama_8_said_cloze_dct_is_sub.json',
            #'subordinate/deepseek_retiring_question_dct_is_sub.json'



        ]
        uid2data = {}
        total_scores = defaultdict(lambda: defaultdict(int))

        for d in data:
            uid2data[d.key().lower()] = d



        for feature_1 in tenses[:]:
            for feature_2 in aspects[:]:
                for feature_3 in tenses:
                    for feature_4 in aspects:
                        new_gpt_answer = {}
                        true_answers = {}
                        data_len = 0

                        for filenamei, filename in enumerate(files):
                            counter = defaultdict(lambda: defaultdict(int))
                            location = self._jl.file_manager.data_filepath(filename)
                            with open(location, 'rt') as f:
                                gpt_answer = json.loads(f.read()) 
                                for d in data:
                                    if d.key() in gpt_answer:
                                        #if (d.sub_aspect() == 'perfect' and d.sub_tense() == 'future') or (d.matrix_aspect() == 'perfect' and d.matrix_tense() == 'Future'):
                                        #    continue
                                        #if (d.sub_aspect() == 'perfect-continuous' and d.sub_tense() == 'future') or (d.matrix_aspect() == 'perfect-continuous' and d.matrix_tense() == 'Future'):
                                        #    continue
                                        if (d.sub_tense() == 'simple') or d.dct_is_sub() == ['-'] or d.temporal_marker() != 'no_marker':
                                            pass#
                                        if not((d.matrix_tense() == 'Future') and d.matrix_aspect() == 'simple' and (d.sub_tense() == 'present') and d.sub_aspect() == 'simple'):
                                            continue
                                        if d.matrix_tense().lower() == feature_1 and d.matrix_aspect() == feature_2 and d.sub_tense() == feature_3 and d.sub_aspect() == feature_4:
                                            #new_gpt_answer[d.key()] = ['x', random.choice(['before', 'after', 'during', 'iduring'])] #gpt_answer[d.key()]
                                            new_gpt_answer[d.key()+filename] = gpt_answer[d.key()]
                                            #if filenamei == 0:
                                            #    true_answers[d.key()] = gpt_answer[d.key()][1]
                                            true_answers[d.key()+filename] = d.dct_is_sub()
                                            data_len += 1

                        f1_score = self.f1_score(true_answers, new_gpt_answer)
                        print(' ' * 4, feature_1, feature_2, feature_3, feature_4, '{:.3f}'.format(f1_score), data_len)
                        total_scores[(tense_mapper[feature_1], aspect_mapper[feature_2])][(tense_mapper[feature_3], aspect_mapper[feature_4])] = f1_score

        for key in total_scores:
            total_scores[key] = dict(total_scores[key])
        print(dict(total_scores))

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
            if '-' in gpt_answer:
                new_gpt_answer += ['-']
            new_label = []
            if 'after' in label:
                new_label += ['after']
            if 'before' in label:
                new_label += ['before']
            if 'during' in label:
                new_label += ['during']
            if '-' in label:
                new_label += ['-']
            for i in new_gpt_answer:
                if i in new_label:
                    tp[i] += 1
                else:
                    fp[i] += 1
            for i in new_label:
                if i not in new_gpt_answer:
                    fn[i] += 1
            if sum(fp.values()) + sum(fn.values()) > 0:
                print(gpt_answers[uid][0], ';', 'gpt:', gpt_answers[uid][1], ';', 'expected:', label)
                pass
            sub_f1 = []
            for key in ['after', 'before', 'during', '-']:
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
