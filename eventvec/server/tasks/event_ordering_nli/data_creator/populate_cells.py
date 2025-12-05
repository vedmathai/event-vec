import csv
import random
import time
import os

from eventvec.server.tasks.event_ordering_nli.data_creator.create import Creator
from eventvec.server.tasks.event_ordering_nli.data_creator.parameters import parameters
from eventvec.server.tasks.event_ordering_nli.datamodel.row import Row
from eventvec.server.tasks.event_ordering_nli.data_creator.hypothesis_templates import hypothesis_templates


class PopulateCells:
    def __init__(self, parameter_name):
        self._data = []
        self._parameter_name = parameter_name
        self._domain = parameters[self._parameter_name]['domain']

    def populate_data(self):
        row_i = 0
        previous_row_i = 0
        count = 40
        while row_i < len(self._data):
            if self._data[row_i]._label != '':
                row_i += 1
                count = 40
                continue
            else:
                if count > 0:
                    count -= 1
                if count == 0:
                    row_i += 1
                    count = 40
                
            row = self._data[row_i]
            print(row_i, row._event_number, row._relationship_number)

            use = False
            while use == False:
                creator = Creator(self._parameter_name)
                event_count = 0
                relationship_count = 0
                while event_count <= row._event_number:
                    creator.create_event()
                    event_count += 1
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
                    if parameters[self._parameter_name]['sort_relationships']:
                        creator.sort_relationships()
                    relationships = '. '.join([str(r) for r in creator._relationships if r.event_point_1().event() != r.event_point_2().event()])
                    row._premise = relationships
                    is_impossible = creator.is_impossible_event_pair(event_1, event_2)
                    if row._relationship == 'overlap':
                        if creator.is_overlap_events(event_1, event_2):
                            row._hops = distance
                            if row._force_impossible is True:
                                threshold = 0.0
                            else:
                                threshold = 0.5
                            if random.random() > threshold:
                                row._hypothesis = hypothesis_templates[self._domain]['overlaps'][0].format(event_1.event_name(), event_2.event_name())

                                if is_impossible:
                                    row._label = 'Impossible'
                                else:
                                    if row._force_impossible:
                                        continue
                                    row._label = 'True'
                            else:
                                relationship = random.choice(['before', 'after'])
                                if relationship == 'before':
                                    row._hypothesis = hypothesis_templates[self._domain]['before'][0].format(event_1.event_name(), event_2.event_name())
                                else:
                                    row._hypothesis = hypothesis_templates[self._domain]['after'][0].format(event_1.event_name(), event_2.event_name())
                                row._label = 'False'
                        else:
                            continue

                    if row._relationship == 'before':
                        if creator.is_strictly_before(event_1, event_2) and not creator.is_overlap_events(event_1, event_2):
                            row._premise = relationships
                            row._hops = distance
                            if row._force_impossible is True:
                                threshold = 0.0
                            else:
                                threshold = 0.5
                            if random.random() > threshold:
                                row._hypothesis = hypothesis_templates[self._domain]['before'][0].format(event_1.event_name(), event_2.event_name())
                                if is_impossible:
                                    row._label = 'Impossible'
                                else:
                                    if row._force_impossible:
                                        continue
                                    row._label = 'True'
                            else:
                                relationship = random.choice(['overlaps', 'after'])
                                if relationship == 'overlaps':
                                    row._hypothesis = hypothesis_templates[self._domain]['overlaps'][0].format(event_1.event_name(), event_2.event_name())
                                else:
                                    row._hypothesis = hypothesis_templates[self._domain]['after'][0].format(event_1.event_name(), event_2.event_name())

                                if is_impossible:
                                    row._label = 'Impossible'
                                else:
                                    row._label = 'False'

                    if row._relationship == 'after':
                        if creator.is_strictly_before(event_2, event_1) and not creator.is_overlap_events(event_1, event_2):
                            row._hops = distance
                            row._premise = relationships
                            if row._force_impossible is True:
                                threshold = 0.0
                            else:
                                threshold = 0.5
                            if random.random() > threshold:
                                row._hypothesis = hypothesis_templates[self._domain]['after'][0].format(event_1.event_name(), event_2.event_name())
                                if is_impossible:
                                    row._label = 'Impossible'
                                else:
                                    if row._force_impossible:
                                        continue
                                    row._label = 'True'
                            else:
                                relationship = random.choice(['overlaps', 'before'])
                                if relationship == 'overlaps':
                                    row._hypothesis = hypothesis_templates[self._domain]['overlaps'][0].format(event_1.event_name(), event_2.event_name())
                                else:
                                    row._hypothesis = hypothesis_templates[self._domain]['before'][0].format(event_1.event_name(), event_2.event_name())


                                if is_impossible:
                                    row._label = 'Impossible'
                                else:
                                    row._label = 'False'
            if row._label != '':
                self.write_data()


    def write_data(self):
        folder_name = '/home/lalady6977/oerc/projects/data/temporal_nli'
        filename = os.path.join(folder_name, parameters[self._parameter_name]['filename'])
        with open(filename, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in self._data:
               writer.writerow(row.to_row())

    def read_data(self):
        folder_name = '/home/lalady6977/oerc/projects/data/temporal_nli'
        filename = os.path.join(folder_name, parameters[self._parameter_name]['filename'])
        with open(filename) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self._data.append(Row.from_row(row))
