import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
from collections import defaultdict
from jadelogs import JadeLogger
import json
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random

from eventvec.server.config import Config
from eventvec.server.tasks.subordinate.datareader.datareader import SubordinateTemporalDatareader

aspects = ['perfect', 'simple', 'continuous', 'perfect-continuous']
tenses = ['past', 'present', 'future']
is_quote = ['yes', 'no']
temporal_marker = ['no_marker', 'yesterday', 'today', 'tomorrow', 'now', 'everyday']

rcParams.update({'figure.autolayout': True})

include = [
    ('Future', 'perfect', 'future', 'perfect', 'no_marker', 'no'),
    ('Future', 'simple', 'past', 'continuous', 'no_marker', 'no'),
    ('Future', 'perfect', 'present', 'simple', 'no_marker', 'no'),
    ('Future', 'perfect', 'past', 'simple', 'no_marker', 'no'),
    ('Past', 'simple', 'future', 'simple', 'no_marker', 'yes'),
    ('Past', 'perfect', 'past', 'simple', 'no_marker', 'no'),
    ('Past', 'simple', 'past', 'simple', 'no_marker', 'yes'),
    ('Past', 'simple', 'present', 'simple', 'no_marker', 'yes'),
    ('Present', 'simple', 'future', 'perfect', 'no_marker', 'no'),
    ('Future', 'simple', 'future', 'perfect', 'no_marker', 'no'),
    ('Past', 'simple', 'future', 'continuous', 'no_marker', 'no'),
    ('Present', 'simple', 'past', 'simple', 'no_marker', 'no'),
    ('Past', 'simple', 'future', 'perfect', 'no_marker', 'no'),
    ('Past', 'simple', 'present', 'continuous', 'no_marker', 'no'),
    ('Future', 'simple', 'present', 'continuous', 'no_marker', 'no'),
    ('Present', 'perfect', 'present', 'simple', 'no_marker', 'no'),
    ('Past', 'simple', 'past', 'continuous', 'no_marker', 'no'),
    ('Future', 'simple', 'present', 'perfect', 'no_marker', 'no'),
    ('Future', 'simple', 'present', 'simple', 'no_marker', 'no'),
    ('Past', 'simple', 'past', 'simple', 'no_marker', 'no'),
]

# Inter model aggreement [Done]
# GPT-5 vs GPT-open vs Llama vs Qwen [Done]
# NLI setup, QA setup, Missing word setup [Done last 2]
# Reasoning of the model chain of thought matches our reasoning [Done]
# try to find a linguistics resource for each phenomenon [Done]
# Qualitative analysis of the errors [Done]
# ate versus fasted versus jogged versus dative [Done]
# it is easier to see matrix-sub in quotes cause the tense relative to only matrix with quotes 
# it is harder to see dct-sub in quotes
# John has been saying that Mary would eat yesterday. versus John had been saying that Mary would eat yesterday.
# Habitual versus single event (graduated vs eaten vs fasted vs jogged) [Done]
# John will say 'Mary ate yesterday.' It is unknown when exactly. But John will say, 'Mary will eat tomorrow'

# Show that high frequency low quality numbers (go through the labels)
# Show that F1 does not favour the random check
# Add the figure of frequency x performance
# Update the related work with sentences regarding how we are better (we are looking at semantics of quotes) (we add to the blimp people with temporal reasoning)
# Add small figure explaining the reichenbach
# Talk about the compositional nature of the problem (only backshift spoils it)
# It is not a frequency problem. Because low frequency is also right.
# Update the main examples figure by pulling out the quotes thing.
# Update the number of datapoints to include the unembedded.
# Break the results figure 1,
# Quotes with models
# Add three textual labels for each quandrant showing tense and aspect

def string2list(answer):
    new_answer = []
    if 'after' in answer:
        new_answer += ['after']
    if 'before' in answer:
        new_answer += ['before']
    if 'during' in answer:
        new_answer += ['during']
    if '-' in answer:
        new_answer += ['-']
    return sorted(new_answer)

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


            #'subordinate/gpt_oss_subordinate_question_plain_dct_is_sub_2.json',
            #'subordinate/gpt_5_said_cloze_dct_is_sub.json',

            #'subordinate/gpt_5_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/gpt5_retiring_question_dct_is_sub.json',
            
            #'subordinate/llama70_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/llama8_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/deepseek_subordinate_question_plain_dct_is_sub.json',

            #'subordinate/gpt_open_jogged_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/gpt_open_fasting_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/gpt_open_graduating_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/gpt_open_retiring_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/gpt_open_retiring_subordinate_question_plain_dct_is_matrix.json',
            #'subordinate/gpt_open_said_cloze_dct_is_sub.json',

            #'subordinate/temporal_subordinate_jogging_unembedded_2.json',
            #'subordinate/gpt_5_temporal_subordinate_jogging_unembedded.json',
            #'subordinate/qwen3_said_question_dct_is_sub.json',
            #'subordinate/qwen3_said_cloze_dct_is_sub.json',
            #'subordinate/deepseek_said_cloze_dct_is_sub.json',
            #'subordinate/llama_70_said_cloze_dct_is_sub.json',
            #'subordinate/llama_8_said_cloze_dct_is_sub.json',
            #'subordinate/deepseek_retiring_question_dct_is_sub.json'
            #'subordinate/deepseek_said_question_dct_is_sub_unembedded.json',
            #'subordinate/qwen3_said_question_dct_is_sub_unembedded.json',
            #'subordinate/llama8_said_question_dct_is_sub_unembedded.json',
            #'subordinate/llama70_said_question_dct_is_sub_unembedded.json',
            #'subordinate/gpt_5_said_cloze_dct_is_sub_unembedded.json',
            #'subordinate/gpt_open_said_cloze_dct_is_sub_unembedded.json',
            #'subordinate/deepseek_said_cloze_dct_is_sub_unembedded.json',
            #'subordinate/qwen3_said_cloze_dct_is_sub_unembedded.json',
            #'subordinate/deepseek_fasted_question_dct_is_sub.json',
            #'subordinate/deepseek_suggested_question_dct_is_sub.json',
            #'subordinate/deepseek_graduated_question_dct_is_sub.json',
            #'subordinate/gpt_5_graduate_question_dct_is_sub.json',
            #'subordinate/gpt_open_dying_question_dct_is_sub.json',
            #'subordinate/deepseek_dying_question_dct_is_sub.json',
            #'subordinate/gpt_5_dying_question_dct_is_sub.json',
            #'subordinate/gpt_5_jogging_question_dct_is_sub.json',
            #'subordinate/gpt_5_stating_question_dct_is_sub.json'
        ]

        uid2data = {}

        true_answers = {}

        for d in data:
            uid2data[d.key().lower()] = d
            true_answers[d.key()] = d.dct_is_sub()

        for filenamei, filename in enumerate(files):
            print(filename)
            location = self._jl.file_manager.data_filepath(filename)
            with open(location, 'rt') as f:
                gpt_answer = json.loads(f.read())
    
                data_len = 0
                for feature_1 in is_quote[:]:
                    for feature_2 in aspects[:1]:
                        new_gpt_answer = {}
                        data_len = 0
                        counter = defaultdict(lambda: defaultdict(int))
                        location = self._jl.file_manager.data_filepath(filename)

                        for d in data:
                            if d.key() in gpt_answer:
                                #if (d.sub_aspect() == 'perfect' and d.sub_tense() == 'future') or (d.matrix_aspect() == 'perfect' and d.matrix_tense() == 'future'):
                                #    continue
                                #if (d.sub_aspect() == 'perfect-continuous' and d.sub_tense() == 'future') or (d.matrix_aspect() == 'perfect-continuous' and d.matrix_tense() == 'future'):
                                #    continue
                                if not tuple([d.matrix_tense(), d.matrix_aspect(), d.sub_tense(), d.sub_aspect(), d.temporal_marker(), d.is_quote()]) in include:
                                    pass
                                if d.temporal_marker() != 'no_marker':
                                    continue
                                if True or d.is_quote().lower() == feature_1: #d.matrix_tense().lower() == feature_1 and d.temporal_marker().lower() == feature_2:
                                    #new_gpt_answer[d.key()] = ['x', random.choice(['before', 'after', 'during'])] #gpt_answer[d.key()]
                                    new_gpt_answer[d.key()] = gpt_answer[d.key()]
                                    #if filenamei == 0:
                                    #    true_answers[d.key()] = gpt_answer[d.key()][1]
                                    #if True and string2list(gpt_answer[d.key()][1]) != string2list(d.dct_is_sub()):
                                    #if d.opposite_key().lower() in uid2data:
                                    #    true_answers[d.key()] = uid2data[d.opposite_key().lower()].dct_is_sub()
                                    data_len += 1
                        
                        f1_score = self.f1_score(true_answers, new_gpt_answer)
                        print(' ' * 4, feature_1, feature_2, '{:.3f}'.format(f1_score), data_len)
        

    def f1_score(self, true_answers, gpt_answers):
        f1s = []
        confusion_matrix = defaultdict(lambda: defaultdict(int))

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

            new_gpt_answer = string2list(gpt_answer)
            new_label = string2list(label)
            for i in new_gpt_answer:
                if i in new_label:
                    tp[i] += 1
                else:
                    fp[i] += 1
            for i in new_label:
                if i not in new_gpt_answer:
                    fn[i] += 1
            confusion_matrix[tuple(new_label)][tuple(new_gpt_answer)] += 1
            if sum(fp.values()) + sum(fn.values())> 0 and len(new_label) == 1: 
                print(gpt_answers[uid][0], '\t', 'gpt: \t', gpt_answers[uid][1], '\t', 'expected:\t', label)
                #pass
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
        matrix = []
        keys = [('-',), ('after',), ('after', 'before'), ('after', 'before', 'during'), ('after', 'during'), ('before',), ('before', 'during'), ('during',)]
        for key1 in keys:
            row = []
            for key2 in keys:
                row += [confusion_matrix[key1][key2]]
            matrix += [row]
        matrix = np.array(matrix)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=keys)
        disp.plot(xticks_rotation='vertical')
        plt.savefig('/home/lalady6977/Downloads/durative_confusion_no_marker.png', bbox_inches='tight')


        return np.mean(f1s)

if __name__ == '__main__':
    Config.instance()
    data_preparer = GPTAnalyse()
    data_preparer.load()
