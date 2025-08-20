import csv

label_map = {
    'contradiction': 'False',
    'entailment': 'True',
    'True': 'True',
    'False': 'False',
    'true': 'True',
    'false': 'False',
    'TRUE': 'True',
    'FALSE': 'False',
    'T': 'True',
    'F': 'False',
}

class Scorer:
    def load(self):
        scores = []
        incorrrects = []
        id2human = {}
        id2mine = {}
        id2item = {}
        filename = '/home/lalady6977/oerc/projects/data/credenceNLI/filled/aashna_filled.csv'
        with open(filename) as f:
            reader = csv.reader(f, delimiter=',')
            for rowi, row in enumerate(reader):
                if rowi == 0:
                    continue
                id2human[row[1]] = row[4]
                id2item[row[1]] = row
        with open('/home/lalady6977/oerc/projects/data/credenceNLI/credence_nli.csv') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                id2mine[row[0]] = row[7]
        for id in id2human:
            if label_map[id2human[id]] == label_map[id2mine[id]]:
                scores += [1]
            else:
                print(id, id2human[id], id2mine[id], id2item[id])
                scores += [0]
        print('scores', sum(scores)/len(scores))


if __name__ == '__main__':
    scorer = Scorer()
    scorer.load()