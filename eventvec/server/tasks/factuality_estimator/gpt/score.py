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



class Scorer:
    def load(self):
        idx2label = {}
        filename = '/home/lalady6977/oerc/projects/data/credenceNLI/filled/aashna_filled.csv'
        #with open(filename) as f:
        #    reader = csv.reader(f, delimiter=',')
        #    for rowi, row in enumerate(reader):
        #        if rowi == 0:
        #            continue
        #        idx2label[row[1]] = row[4]

        with open('/home/lalady6977/oerc/projects/data/credenceNLI/credence_nli.csv') as f:
            reader = csv.reader(f, delimiter=',')
            for l in reader:
                idx2label[l[0]] = l[7]
        foldername = '/home/lalady6977/oerc/projects/data/credenceNLI/results'
        below_average_counter = defaultdict(int)
        above_average_counter = defaultdict(int)
        for filename in sorted(os.listdir(foldername)):
            idx2score = defaultdict(list)
            fullfilename = os.path.join(foldername, filename)
            print(filename)
            with open(fullfilename) as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    idx = row[0]
                    if idx not in idx2label:
                        continue
                    if  row[7] not in label_map:
                        continue
                    if label_map[row[7]] == label_map[idx2label[row[0]]]:
                        idx2score[row[2]] += [1]
                    else:
                        idx2score[row[2]] += [0]
                        #print(row[0], row[3], row[4], row[5], row[6], row[7])
            total = []
            for key in idx2score:
                score = sum(idx2score[key]) / len(idx2score[key])
                total += [score]
                #print(f'Index {key} score: {score}')
            mean = np.mean(total)
            print('mean:', mean)
            print('std:', np.std(total))
            for key, value in sorted(idx2score.items(), key=lambda x: np.mean(x[1]), reverse=True):
                score = np.mean(value)
                if score > mean:
                    print(f'Index {key} score: {score}')
                    above_average_counter[key] += 1
            print('-' * 50)
            for key, value in sorted(idx2score.items(), key=lambda x: np.mean(x[1]), reverse=True):
                score = np.mean(value)
                if score <= mean:
                    print(f'Index {key} score: {score}')
                    above_average_counter[key] -= 1
            print('\n' * 5)

        for key, value in sorted(above_average_counter.items(), key=lambda x: (x[1], x[0]), reverse=True):
            print(key, value)

if __name__ == '__main__':
    scorer = Scorer()
    scorer.load()