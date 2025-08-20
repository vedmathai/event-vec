import csv
from collections import defaultdict
import numpy as np
import os

label_map = {
    'contradiction': 'False',
    'entailment': 'True',
    'True': 'True',
    'False': 'False',
    'true': 'True',
    'false': 'False',
    'T': 'True',
    'F': 'False',
}

check_outside = True

class Scorer:
    def load(self):
        idx2label = defaultdict(lambda: dict())
        with open('/home/lalady6977/oerc/projects/data/subordinate/pronouns.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            for li, l in enumerate(reader):
                if li == 0:
                    continue
                key = tuple([l[1], l[2], l[3], l[4]])
                idx2label[key][l[5]] = (l[7], l[8])
        foldername = '/home/lalady6977/oerc/projects/data/subordinate/results_quotes'
        for filename in sorted(os.listdir(foldername)):
            fullfilename = os.path.join(foldername, filename)
            print(filename)
            idx2choice_dissimilar = {}
            idx2choice_similar = {}
            idx2choice_dissimilar_examples = {}
            idx2choice_similar_examples = {}
            mismatch_confused = defaultdict(int)
            mismatch_total = defaultdict(int)

            with open(fullfilename) as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    if int(row[0]) % 2 == 1:
                        key = tuple([row[1], row[2], row[3], row[4], row[5]])
                        idx2choice_dissimilar[key] = label_map[row[9]]
                        idx2choice_dissimilar_examples[key] = row[7]
                    if int(row[0]) % 2 == 0:
                        key = tuple([row[1], row[2], row[3], row[4], row[5]])
                        idx2choice_similar[key] = label_map[row[9]]
                        idx2choice_similar_examples[key] = row[7]
            for check_outside in [True, False]:
                idx2score = defaultdict(list)
                seen = set()

                if check_outside is True:
                    print('  Checking normal')
                else:
                    print('  Checking substituted')
                with open(fullfilename) as f:
                    reader = csv.reader(f, delimiter=',')
                    for row in reader:
                        idx = row[0]
                        if idx in seen:
                            continue
                        seen.add(idx)
                        key = tuple([row[1], row[2], row[3], row[4]])
                        if int(idx) % 2 == 0:
                            if row[5] == 'direct':
                                mismatch_total[(row[1], row[2])] += 1
                                if check_outside is True:
                                    if label_map[row[8]] == label_map[row[9]]:
                                        idx2score[row[5]] += [1]
                                    else:
                                        mismatch_confused[(row[1], row[2])] += 1
                                        idx2score[row[5]] += [0]
                                else:
                                    if idx2label[key]['direct'][0] == idx2label[key]['indirect'][1]:
                                        mismatch_total[row[2]] += 1 
                                        key2 = tuple([row[1], row[2], row[3], row[4], 'direct'])
                                        if label_map[row[8]] != label_map[row[9]]:
                                            if key2 in idx2choice_dissimilar:
                                                if idx2choice_dissimilar[key2] == label_map['true']:
                                                    idx2score[row[5]] += [1]
                                                else:
                                                    idx2score[row[5]] += [0]
                                            else:
                                                idx2score[row[5]] += [0]
                                        else:
                                            idx2score[row[5]] += [1]

                                    elif idx2label[key]['direct'][0] == idx2label[key]['indirect'][0]:
                                        key2 = tuple([row[1], row[2], row[3], row[4], 'direct'])
                                        if label_map[row[8]] != label_map[row[9]]:
                                            if key2 in idx2choice_similar:
                                                if idx2choice_similar[key2] == label_map['true']:
                                                    idx2score[row[5]] += [1]
                                                else:
                                                    idx2score[row[5]] += [0]
                                            else:
                                                idx2score[row[5]] += [0]
                                        else:
                                            idx2score[row[5]] += [1]
                                    
                                    else:
                                        if label_map[row[8]] == label_map[row[9]]:
                                            idx2score[row[5]] += [1]
                                        else:
                                            idx2score[row[5]] += [0]

                            if row[5] == 'indirect':
                                if label_map[row[8]] == label_map[row[9]]:
                                    idx2score[row[5]] += [1]
                                else:

                                    idx2score[row[5]] += [0]
                    total = []
                    for key in idx2score:
                        score = sum(idx2score[key]) / len(idx2score[key])
                        total += [score]
                        print(' '* 4, key, score)
                    for key in mismatch_confused:
                        if mismatch_total[key] != 0:
                            print(' '* 4, key, 'mismatch confused:', mismatch_confused[key], 'total:', mismatch_total[key], 'confused:', mismatch_confused[key] / mismatch_total[key])


if __name__ == '__main__':
    scorer = Scorer()
    scorer.load()