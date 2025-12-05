import numpy as np
import random
from collections import defaultdict
from jadelogs import JadeLogger
import json
import os
import matplotlib.pyplot as plt  
import matplotlib
import ast

from eventvec.server.config import Config
from eventvec.server.tasks.subordinate.datareader.datareader import SubordinateTemporalDatareader



class NLIDataPreparer():
    def __init__(self):
        self._data_readers = {
            'subordinate': SubordinateTemporalDatareader(),
        } 

    def load(self):
        self._jl = JadeLogger()
        data_reader = self._data_readers['subordinate']
        data = data_reader.data('temporal_subordinate_retiring')[:4800]
        gpt_answer_files = [   
            #'subordinate/gpt_oss_subordinate_question_plain_dct_is_sub.json',
            #'subordinate/gpt_oss_subordinate_question_plain_dct_is_sub_2.json',
            'subordinate/gpt_open_retiring_subordinate_question_plain_dct_is_sub.json'
        ]
             
        for gpt_answer_file in gpt_answer_files:
            location = self._jl.file_manager.data_filepath(gpt_answer_file)
            with open(location) as f:
                gpt_answer = json.loads(f.read())

            random.shuffle(data)
            counter_direct = defaultdict(lambda: defaultdict(int))
            counter_indirect = defaultdict(lambda: defaultdict(int))
            counter_gpt_direct = defaultdict(lambda: defaultdict(int))
            counter_gpt_indirect = defaultdict(lambda: defaultdict(int))
            counter_ambiguous_direct = defaultdict(int)
            counter_ambiguous_indirect = defaultdict(int)
            yes_count = defaultdict(int)

            relationship_keys = set()
            uid2datum = {}
            for datumi, datum in enumerate(data, start=1):
                relationships = datum.dct_is_sub()
                key = tuple(sorted(datum.dct_is_sub()))
                uid2datum[datum.key()] = datum
                relationship_keys.add(key)
                if datum.temporal_marker() != 'no_marker':
                    continue
                if datum.dct_is_sub() == ['-']:
                    continue
                if datum.is_quote() == 'yes':
                    counter_direct[(datum.matrix_tense().lower(), datum.sub_tense().lower())][key] += 1
                else:
                    counter_indirect[(datum.matrix_tense().lower(), datum.sub_tense().lower())][key] += 1

            for datumi, (key, value) in enumerate(gpt_answer.items(), start=1):
                matrix_tense, matrix_aspect, sub_tense, sub_aspect, is_quote, adverb = key.split('_')[:6]
                value = ast.literal_eval(value[1])
                value = tuple(sorted(value))
                if not(adverb == 'no'):
                    continue
                if is_quote == 'yes':
                    counter_gpt_direct[(matrix_tense.lower(), sub_tense.lower())][value] += 1
                else:
                    counter_gpt_indirect[(matrix_tense.lower(), sub_tense.lower())][value] += 1

            for datumi, (key, value) in enumerate(sorted(gpt_answer.items(), key=lambda x: x[0])):
                matrix_tense, matrix_aspect, sub_tense, sub_aspect, is_quote, adverb = key.split('_')[:6]
                value = ast.literal_eval(value[1])
                value = tuple(sorted(value))
                if not(adverb == 'no' and uid2datum[key].dct_is_sub() == ['after', 'before']):
                    continue
                print('\t'.join([matrix_tense, sub_tense, sub_aspect, is_quote, uid2datum[key].example(), str(uid2datum[key].dct_is_sub()), str(value)])) 
                for rel in ['after', 'before', 'during']: ## For the ambiguous cases find patterns in why the model is choosing certain things
                    ## make an error decision tree (what falls under backshifting, what falls under temporal anchoring)
                    if is_quote == 'yes':
                        if rel in value:
                            counter_ambiguous_direct[rel] += 1
                    else:
                        if rel in value:
                            counter_ambiguous_indirect[rel] += 1
            print(sum(counter_ambiguous_direct.values()))
            print(sum(counter_ambiguous_indirect.values()))

        ax = plt.gca()

        X = [('before',), ('during',), ('after',), ('before', 'during'), ('after', 'during'), ('after', 'before', 'during'), ('-')]
        X_axis = range(len(X))

        #points = [('past', 'future'), ('future', 'past')][0:1]
        #points = [('past', 'future'), ('future', 'past'), ('past', 'past'), ('future', 'future')][2:3]
        #points = [('past', 'past')][0:1]
        points = counter_gpt_direct.keys()
        #points = [('present', 'past'), ('present', 'present'), ('present', 'future')]

        colour_map = {('past', 'future'): 'C0', ('future', 'past'): 'C1', ('past', 'past'): 'C1', ('future', 'future'): 'C1'}
        colour_map = {
            ('present', 'past'): 'C0', ('present', 'present'): 'C1', ('present', 'future'): 'C2',
            ('past', 'past'): 'C0', ('past', 'present'): 'C1', ('past', 'future'): 'C2',
            ('future', 'past'): 'C0', ('future', 'present'): 'C1', ('future', 'future'): 'C2',
        }
        fig, ax = plt.subplots(layout='constrained')
        for key_i, key1 in enumerate(sorted(points)):
            graph = []
            for key2 in X:
                graph += [counter_gpt_direct[key1][key2]]
            ax.plot(X_axis, graph, color=colour_map[key1], linestyle='solid', label = '{}_direct_gpt'.format(key1), alpha=0.5)

        for key_i, key1 in enumerate(sorted(points)):
            graph = []
            for key2 in X:
                graph += [counter_gpt_indirect[key1][key2]]
            ax.plot(X_axis, graph, color=colour_map[key1], linestyle='solid', label = '{}_indirect_gpt'.format(key1), alpha=1)

        for key_i, key1 in enumerate(sorted(points)):
            graph = []
            for key2 in X:
                graph += [counter_direct[key1][key2]]
            ax.plot(X_axis, graph, color=colour_map[key1], linestyle='dashed', label = '{}_direct_true'.format(key1), alpha=0.5)

        for key_i, key1 in enumerate(sorted(points)):
            graph = []
            for key2 in X:
                graph += [counter_indirect[key1][key2]]
            ax.plot(X_axis, graph, color=colour_map[key1], linestyle='dashed', label = '{}_indirect_true'.format(key1), alpha=1)

        ax.set_ylim([0, 70])
        #ax.set_xlim([0.2, 0.5])

        plt.xticks(X_axis, X, rotation=45) 
        plt.xlabel("Relationship") 
        plt.ylabel("Counts") 
        #plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
        plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.55), ncol=2) 


        plt.savefig('/home/lalady6977/Downloads/tense_graph_past_present_punctual.png', bbox_inches='tight')

        fig, ax = plt.subplots(layout='constrained')
        X_axis= [0, 1, 2]
        X = ['before', 'during', 'after']
        graph = []
        for key2 in X:
            graph += [counter_ambiguous_direct[key2]]
        print(graph)
        
        ax.plot(X_axis, graph, color=colour_map[key1], linestyle='dashed', label = '_indirect_true'.format(key1), alpha=1)
        graph = []
        for key2 in X:
            graph += [counter_ambiguous_indirect[key2]]
        print(graph)
        ax.plot(X_axis, graph, color=colour_map[key1], linestyle='dashed', label = '_indirect_true'.format(key1), alpha=0.5)
        ax.set_ylim([0, 20])
        #ax.set_xlim([0.2, 0.5])

        plt.xticks(X_axis, X, rotation=45) 
        plt.xlabel("Relationship") 
        plt.ylabel("Counts") 
        #plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
        plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.55), ncol=2) 
        plt.savefig('/home/lalady6977/Downloads/confusion.png', bbox_inches='tight')

            
if __name__ == '__main__':
    Config.instance()
    data_preparer = NLIDataPreparer()
    data_preparer.load()