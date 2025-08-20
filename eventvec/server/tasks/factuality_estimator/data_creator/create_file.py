import csv
import random

modal_type = [
    'none',
    'none_not',
    'if',
    'if_not',
    'then',
    'then_not',
    'modal',
    'modal_not',
    'sub_speech',
    'sub_speech_inside_not',
    'sub_speech_outside_not',
    'sub_speech_modal',
    'sub_speech_modal_not',
    'sub_belief',
    'sub_belief_inside_not',
    'sub_belief_outside_not',
    'sub_belief_modal',
    'sub_belief_modal_not',
    'sub_infinitive',
    'sub_infinitive_inside_not',
    'sub_infinitive_outside_not',
    'sub_possible_that',
    'sub_possible_that_inside_not',
    'sub_possible_that_outside_not',
    'sub_or',
    'sub_nor',
]

second_sentence = [
    'none',
    'modal',
    'modal_not',
    'not',
]

class CreateCells:
    def __init__(self):
        self._data = []

    def create_data(self):
        iter = 0
        for count in range(20):
            for mtype in modal_type:
                for ssentence in second_sentence:
                        self._data.append([iter+1, count+1, mtype, ssentence])
                        iter += 1


    def write_data(self):
        with open('/home/lalady6977/oerc/projects/data/credenceNLI/credence_nli.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in self._data:
               writer.writerow(row)

if __name__ == '__main__':
    creator = CreateCells()
    creator.create_data()
    creator.write_data()