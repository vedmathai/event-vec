import numpy as np
import pprint
import re
from collections import defaultdict
from transformers import BertTokenizer, RobertaTokenizer
import pprint
import random
from collections import defaultdict
from jadelogs import JadeLogger
import csv
import time


from eventvec.server.config import Config
from eventvec.server.tasks.entailment_classification.gpt_4.llama_3_api import llama_3
from eventvec.server.tasks.entailment_classification.gpt_4.sambanova import sambanova
from eventvec.server.tasks.entailment_classification.gpt_4.qwen import qwen
from eventvec.server.tasks.entailment_classification.gpt_4.gpt_4_api import gpt_4
from eventvec.server.tasks.event_ordering_nli.datareader.temporal_datareader import TemporalDatareader
import os


prompt_preamble = """
    Given below is a premise and a hypothesis. The task is to say how true the hypothesis is given the information in the premise.
    Provide a label from {True, False}.
    Do not provide your reasoning, just provide the answer.

    Provide the answer in the form of
    Answer: index: label

"""

para_type = 'single'
model = 'llama70b'
round_count = '1'
results_sheet = f'/home/lalady6977/oerc/projects/data/credenceNLI/results_single/{model}_{para_type}_{round_count}.csv'

class NLIDataPreparer():
    def load(self):
        results = self.load_sheet()
        result_indices = [r[0] for r in results]
        contexts = {}
        with open('/home/lalady6977/oerc/projects/data/credenceNLI/contexts.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            for l in reader:
                if len(l) < 3:
                    continue
                contexts[l[0]] = [l[1].strip(), l[2].strip()]
        with open('/home/lalady6977/oerc/projects/data/credenceNLI/credence_nli.csv') as f:
            reader = csv.reader(f, delimiter=',')
            for l in reader:
                if l[0] in result_indices:
                    continue
                prompt = str(prompt_preamble)
                #prompt += f"Index: {l[0]}\n"
                if para_type == 'para':
                    premise = contexts[l[1]][0] + ' ' + l[4] + ' ' + contexts[l[1]][1]
                elif para_type == 'single':
                    premise = l[4]
                prompt += f"Premise: {premise}\n"
                prompt += f"Hypothesis: {l[6]}\n"


                answer = gpt_4(prompt, '')
                for line in answer.split('\n'):
                    if ':' in line:
                        try:
                            if len(line.split(':')) == 3:
                                _, index, label = line.split(':')
                            elif len(line.split(':')) == 2:
                                _, label = line.split(':')
                            print( 'GPT:', label.strip(), 'expected:', l[7])
                            results.append([l[0], l[1], l[2], l[3], l[4], l[6], l[7], label.strip()])
                        except ValueError:
                            print(line)
                            continue
                    self.save_sheet(results)
                    time.sleep(3)
        self.save_sheet(results)
        

    def load_sheet(self):
        results = []
        if os.path.exists(results_sheet):
            with open(results_sheet) as f:
                reader = csv.reader(f, delimiter=',')
                for l in reader:
                    if len(l) < 7:
                        continue
                    results.append(l)
        return results

    def save_sheet(self, results):
        results = sorted(results, key=lambda x: int(x[0]))
        with open(results_sheet, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            for result in results:
                writer.writerow(result)


if __name__ == '__main__':
    Config.instance()
    data_preparer = NLIDataPreparer()
    data_preparer.load()