from typing import Dict
import csv
import os
from jadelogs import JadeLogger


from eventvec.server.tasks.event_ordering_nli.datareader.datamodel import TemporalNLIRow

from eventvec.server.data.abstract import AbstractDatareader

files = {
    'temporal_nli_all_diff_train': 'temporal_nli_all_diff_train.csv',
    'temporal_nli_same_english_train': 'temporal_nli_same_english_train.csv',
    'temporal_nli_same_names': 'temporal_nli_same_names.csv',
    'temporal_nli_same_structures_train': 'temporal_nli_same_structures_train.csv',
    'temporal_nli_same_structures_and_templates_train': 'temporal_nli_same_structures_and_templates_train.csv',
    'temporal_nli_test': 'temporal_nli_test.csv',
    'temporal_nli_simple_event_names': 'temporal_nli_simple_event_names.csv',
    'temporal_nli_before_after_train': 'temporal_nli_before_after_train.csv',
    'temporal_nli_before_after_test': 'temporal_nli_before_after_test.csv',
    'temporal_nli_before_simultaneous_train': 'temporal_nli_before_simultaneous_train.csv',
    'temporal_nli_before_simultaneous_test': 'temporal_nli_before_simultaneous_test.csv',
    'temporal_nli_after_simultaneous_train': 'temporal_nli_after_simultaneous_train.csv',
    'temporal_nli_after_simultaneous_test': 'temporal_nli_#after_simultaneous_test.csv',
    'temporal_nli_after_simultaneous_train': 'temporal_nli_after_simultaneous_train.csv',
    'temporal_nli_after_simultaneous_test': 'temporal_nli_after_simultaneous_test.csv',
    'temporal_nli_simple_event_names_train': 'temporal_nli_simple_event_names_train.csv',
    'temporal_nli_simple_event_names_test': 'temporal_nli_simple_event_names_test.csv',
    'temporal_nli_sort_relationships_train': 'temporal_nli_sort_relationships_train.csv',
    'temporal_nli_sort_relationships_test': 'temporal_nli_sort_relationships_test.csv',
    'temporal_nli_only_before_sort_train': 'temporal_nli_only_before_sort_train.csv',
    'temporal_nli_only_before_sort_test': 'temporal_nli_only_before_sort_test.csv',
    'spatial_nli_relationships_test': 'spatial_nli_relationships_test.csv',
    'spatial_nli_relationships_train': 'spatial_nli_relationships_train.csv',
    'logical_nli_relationships_test': 'logical_nli_relationships_test.csv',
    'logical_nli_relationships_train': 'logical_nli_relationships_train.csv',
}

class TemporalDatareader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self.folder = self._config.temporal_nli_data_location()
        self._jade_logger = JadeLogger()

    def data(self, train_test='train') -> Dict:
        filename = files[train_test]
        filepath = os.path.join(self.folder, filename)

        data = []
        with open(filepath) as f:
            reader = csv.reader(f, delimiter='\t')
            for r in reader:
                datum = TemporalNLIRow.from_array(r)
                data.append(datum)
        return data
    

if __name__ == '__main__':
    reader = TemporalDatareader()
    d = reader.data()
    for r in d:
        print(r._label)