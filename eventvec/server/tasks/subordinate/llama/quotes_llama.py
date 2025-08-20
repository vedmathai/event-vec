import numpy as np
from collections import defaultdict
from transformers import BertTokenizer, RobertaTokenizer
from collections import defaultdict
import time
import os
import csv


from eventvec.server.config import Config
from eventvec.server.tasks.entailment_classification.gpt_4.llama_3_api import llama_3
from eventvec.server.tasks.entailment_classification.gpt_4.gpt_4_api import gpt_4
from eventvec.server.tasks.entailment_classification.gpt_4.sambanova import sambanova

from eventvec.server.tasks.event_ordering_nli.datareader.temporal_datareader import TemporalDatareader
from eventvec.server.tasks.subordinate.datareader.datareader import SubordinateTemporalDatareader



prompt_preamble = """
[INST] <<SYS>>

    This is a modified NLI task. You will be given a premise written by speaker A and read by listener B. You will also be given a hypothesis. 
    If the hypothesis is true given the premise answer true. 
    If the hypothesis is false given the hypothesis answer false. 

    Example 1:
    Key: 5647:
    Premise: Alice said, 'her house is in Oxford.'
    Hypothesis: According to Alice, someone else's house, not Alice's is in Oxford.
    Answer: 5647: true

    key: 7624:
    Premise: Alice said that your house is in Oxford.
    Hypothesis: According to Alice, Speaker A’s house is in Oxford.
    Answer: 7624: false

    The query is in the form:
    key: <key>: 
    Premise: <premise> 
    Hypothesis: <hypothesis>

    Response format:
    Answer: <key>: <Answer>


    """


para_type = 'single'
model = 'gpt-4o'
round_count = '1'
results_sheet = f'/home/lalady6977/oerc/projects/data/subordinate/results_quotes/{model}_{para_type}_{round_count}.csv'

class NLIDataPreparer():
    def load(self):
        results = self.load_sheet()
        result_indices = [r[0] for r in results]
        contexts = {}
        with open('/home/lalady6977/oerc/projects/data/subordinate/pronouns.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            for li, l in enumerate(reader):
                if li == 0:
                    continue
                if int(l[0]) * 2 in result_indices and (int(l[0]) * 2) + 1 in result_indices:
                    continue
                prompt = str(prompt_preamble)
                #prompt += f"Index: {l[0]}\n"
                premise = l[6]
                prompt += f"Key: {str(int(l[0]) * 2)}\n"
                prompt += f"Premise: {premise}\n"
                prompt += f"Hypothesis: {l[7]}\n"


                answer = gpt_4(prompt, '')
                for line in answer.split('\n'):
                    if ':' in line:
                        try:
                            if len(line.split(':')) == 3:
                                _, index, label = line.split(':')
                            elif len(line.split(':')) == 2:
                                _, label = line.split(':')
                            print(premise, '....', l[7])
                            print( 'GPT:', label.strip(), 'expected:', 'true')
                            if label.strip() in ['true', 'false']:
                                results.append([int(l[0]) * 2, l[1], l[2], l[3], l[4], l[5], l[6], l[7], 'true', label.strip()])
                        except ValueError:
                            print(line)

                    self.save_sheet(results)
                    time.sleep(3)

                if (int(l[0]) * 2) + 1 in result_indices:
                    continue

                prompt = str(prompt_preamble)
                #prompt += f"Index: {l[0]}\n"
                premise = l[6]
                prompt += f'Key: {(int(l[0]) * 2) + 1}\n'
                prompt += f"Premise: {premise}\n"
                prompt += f"Hypothesis: {l[8]}\n"


                answer = gpt_4(prompt, '')
                for line in answer.split('\n'):
                    if ':' in line:
                        try:
                            if len(line.split(':')) == 3:
                                _, index, label = line.split(':')
                            elif len(line.split(':')) == 2:
                                _, label = line.split(':')
                            print(premise, '....', l[8])
                            print( 'GPT:', label.strip(), 'expected:', 'false')
                            if label.strip() in ['true', 'false']:
                                results.append([(int(l[0]) * 2) + 1, l[1], l[2], l[3], l[4], l[5], l[6], l[8], 'false', label.strip()])
                        except ValueError:
                            print(line)
                            continue
                    self.save_sheet(results)
                    time.sleep(3)
        if not results:
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