import csv
import os
import uuid
import random

class NLIDataPreparer():
    def load(self):
        lines = []
        ids = [str(uuid.uuid4())[-15:] for _ in range(210)]
        ids = sorted(ids)
        with open('/home/lalady6977/oerc/projects/data/credenceNLI/credence_nli.csv') as f:
            reader = csv.reader(f, delimiter=',')
            counter = 0
            for line in reader:
                if line[1] in ['5', '6']:
                    counter += 1
                    lines.append([ids[counter]] + line)

        random.shuffle(lines)
        with open('/home/lalady6977/oerc/projects/data/credenceNLI/credence_nli_vijay.csv', 'wt') as f:
            writer = csv.writer(f, delimiter=',')
            for l in lines:
                writer.writerow(l)


if __name__ == '__main__':
    data_preparer = NLIDataPreparer()
    data_preparer.load()