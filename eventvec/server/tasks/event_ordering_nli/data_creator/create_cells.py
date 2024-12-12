import csv
import random

class CreateCells:
    def __init__(self):
        self._data = []

    def create_data(self):
        count = 1
        total_data = 2000
        per_factor_set = int(total_data/2/3/4/3) + 1
        for possible in ['possible', 'impossible']:
            for relationship in ['after', 'before', 'overlap']:
                for event_number_power in range(2, 6):
                    event_number = 2 ** event_number_power
                    for relationship_multiplier in [0.5, 1, 2]:
                        relationship_number = int(event_number * relationship_multiplier)
                        relationship_number = min(max(relationship_number, 3), 32)
                        for i in range(per_factor_set):
                            hops = 0
                            self._data.append([count, possible, relationship, event_number, relationship_number, hops, '', '', ''])
                            count += 1


    def write_data(self):
        with open('/home/lalady6977/oerc/projects/data/temporal_nli/temporal_nli_sort_relationships_test.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in self._data:
               writer.writerow(row)

if __name__ == '__main__':
    creator = CreateCells()
    creator.create_data()
    creator.write_data()