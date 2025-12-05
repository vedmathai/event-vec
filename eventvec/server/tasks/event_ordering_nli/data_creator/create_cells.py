import csv
import random
import os

from eventvec.server.tasks.event_ordering_nli.data_creator.parameters import parameters


class CreateCells:
    def __init__(self, parameter):
        self._data = []
        self._parameter_name = parameter

    def create_data(self):
        count = 0
        per_factor_set = parameters[self._parameter_name]['count']
        for possible in ['possible']:
            for relationship in ['after', 'before', 'overlap']:
                for event_number_power in range(2, 6):
                    event_number = 2 ** event_number_power
                    for relationship_multiplier in [0.5, 1, 2]:
                        relationship_number = int(event_number * relationship_multiplier)
                        relationship_number = min(max(relationship_number, 3), 32)
                        for i in range(per_factor_set):
                            hops = 0
                            self._data.append([count, possible, 'false', relationship, event_number, relationship_number, hops, '', '', ''])
                            count += 1
        for possible in ['impossible']:
            for relationship in ['after', 'before']:
                for event_number_power in range(2, 6):
                    event_number = 2 ** event_number_power
                    for relationship_multiplier in [0.5, 1, 2]:
                        relationship_number = int(event_number * relationship_multiplier)
                        relationship_number = min(max(relationship_number, 3), 32)
                        for i in range(per_factor_set):
                            hops = 0
                            force_impossible_label = 'true' if random.random() < 0.7 else 'false'
                            self._data.append([count, possible, force_impossible_label, relationship, event_number, relationship_number, hops, '', '', ''])
                            count += 1

    
    def write_data(self):
        filename = parameters[self._parameter_name]['filename']
        filename = '/home/lalady6977/oerc/projects/data/temporal_nli/{}'.format(filename)
        if os.path.exists(filename) is False:
            with open(filename, 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                for row in self._data:
                    writer.writerow(row)



