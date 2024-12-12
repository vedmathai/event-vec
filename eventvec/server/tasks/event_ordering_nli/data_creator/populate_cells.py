import csv
import random
import time
import os

from eventvec.server.tasks.event_ordering_nli.data_creator.create import Creator, parameters, parameter_name


if parameters[parameter_name]['random_seed']:
    random.seed(0)

class Row:
    def __init__(self):
        self._count = 0
        self._possible = None
        self._relationship = None
        self._event_number = 0
        self._relationship_number = 0
        self._hops = 0
        self._premise = ''
        self._hypothesis = ''
        self._label = ''

    @classmethod
    def from_row(self, datum):
        row = Row()
        row._count = datum[0]
        row._possible = datum[1]
        row._relationship = datum[2]
        row._event_number = int(datum[3])
        row._relationship_number = int(datum[4])
        row._hops = int(datum[5])
        row._premise = datum[6]
        row._hypothesis = datum[7]
        row._label = datum[8]
        return row
    
    def to_row(self):
        return [self._count, self._possible, self._relationship, self._event_number, self._relationship_number, self._hops, self._premise, self._hypothesis, self._label]
    

class PopulateCells:
    def __init__(self):
        self._data = []

    def populate_data(self):
        row_i = 0
        previous_row_i = 0
        count = 100
        while row_i < len(self._data):
            if self._data[row_i]._label != '':
                row_i += 1
                count = 100
                continue
            else:
                if count > 0:
                    count -= 1
                if count == 0:
                    row_i += 1
                    count = 100
                
            row = self._data[row_i]
            print(row_i, row._event_number, row._relationship_number)

            use = False
            while use == False:
                creator = Creator()
                event_count = 0
                relationship_count = 0
                while event_count <= row._event_number:
                    creator.create_event()
                    event_count += 1
                used = True
                while relationship_count <= row._relationship_number:
                    creator.create_random_relationship()
                    relationship_count += 1
                impossible_event_points = creator.find_all_impossible_event_points()
                if row._possible == 'possible' and len(impossible_event_points) == 0:
                    use = True
                elif row._possible == 'impossible' and len(impossible_event_points) > 0:
                    use = True

            distances2events = creator.distances2events()
            for distance in sorted(distances2events.keys(), reverse=True):
                if row._label != '':
                    break
                for (event_1, event_2) in distances2events[distance]:
                    if row._label != '':
                        break
                    if parameters[parameter_name]['sort_relationships']:
                        creator.sort_relationships()
                    relationships = '. '.join([str(r) for r in creator._relationships if r.event_point_1().event() != r.event_point_2().event()])
                    row._premise = relationships
                    is_impossible = creator.is_impossible_event_pair(event_1, event_2)
                    if row._relationship == 'overlap':
                        if creator.is_overlap_events(event_1, event_2):
                            row._hops = distance
                            if random.random() > 0:
                                row._hypothesis = f'{event_1.event_name()} {row._relationship}s with {event_2.event_name()}'
                                if is_impossible:
                                    row._label = 'Impossible'
                                else:
                                    continue
                                    row._label = 'True'
                            else:
                                relationship = random.choice(['before', 'after'])
                                row._hypothesis = f'{event_1.event_name()} happens {relationship} {event_2.event_name()}'
                                row._label = 'False'
                        else:
                            continue

                    if row._relationship == 'before':
                        if creator.is_strictly_before(event_1, event_2) and not creator.is_overlap_events(event_1, event_2):
                            row._premise = relationships
                            row._hops = distance
                            if random.random() > 0:
                                row._hypothesis = f'{event_1.event_name()} happens {row._relationship} {event_2.event_name()}'
                                if is_impossible:
                                    row._label = 'Impossible'
                                else:
                                    
                                    continue
                                    row._label = 'True'
                            else:
                                relationship = random.choice(['overlaps', 'after'])
                                if relationship == 'overlaps':
                                    row._hypothesis = f'{event_1.event_name()} overlaps with {event_2.event_name()}'
                                else:
                                    row._hypothesis = f'{event_1.event_name()} happens after {event_2.event_name()}'
                                if is_impossible:
                                    row._label = 'Impossible'
                                else:
                                    row._label = 'False'

                    if row._relationship == 'after':
                        if creator.is_strictly_before(event_2, event_1) and not creator.is_overlap_events(event_1, event_2):
                            row._hops = distance
                            row._premise = relationships
                            if random.random() > 0:
                                row._hypothesis = f'{event_1.event_name()} happens {row._relationship} {event_2.event_name()}'
                                if is_impossible:
                                    row._label = 'Impossible'
                                else:
                                    continue
                                    row._label = 'True'
                            else:
                                relationship = random.choice(['overlaps', 'before'])
                                if relationship == 'overlaps':
                                    row._hypothesis = f'{event_1.event_name()} overlaps with {event_2.event_name()}'
                                else:
                                    row._hypothesis = f'{event_1.event_name()} happens before {event_2.event_name()}'
                                if is_impossible:
                                    row._label = 'Impossible'
                                else:
                                    row._label = 'False'

            if row._label != '':
                self.write_data()


    def write_data(self):
        folder_name = '/home/lalady6977/oerc/projects/data/temporal_nli'
        filename = os.path.join(folder_name, parameters[parameter_name]['filename'])
        with open(filename, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in self._data:
               writer.writerow(row.to_row())

    def read_data(self):
        folder_name = '/home/lalady6977/oerc/projects/data/temporal_nli'
        filename = os.path.join(folder_name, parameters[parameter_name]['filename'])
        with open(filename) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self._data.append(Row.from_row(row))


if __name__ == '__main__':
    creator = PopulateCells()
    creator.read_data()
    creator.populate_data()
    creator.write_data()