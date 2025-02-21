import csv
from collections import defaultdict


class Row():
    def __init__(self):
        self._matrix_tense = None
        self._matrix_aspect = None
        self._sub_tense = None
        self._sub_aspect = None
        self._is_quote = None
        self._is_future_looking_temporal_adverb = None
        self._is_past_looking_temporal_adverb = None
        self._temporal_marker = None
        self._example = None
        self._is_possible = None
        self._sub_is_repeated_process = None
        self._matrix_is_sub = []
        self._sub_is_dct = []
        self._matrix_is_dct = []

    def set_matrix_tense(self, matrix_tense):
        self._matrix_tense = matrix_tense

    def set_matrix_aspect(self, matrix_aspect):
        self._matrix_aspect = matrix_aspect

    def set_sub_tense(self, sub_tense):
        self._sub_tense = sub_tense

    def set_sub_aspect(self, sub_aspect):
        self._sub_aspect = sub_aspect

    def set_is_quote(self, is_quote):
        self._is_quote = is_quote

    def set_is_future_looking_temporal_adverb(self, is_future_looking_temporal_adverb):
        self._is_future_looking_temporal_adverb = is_future_looking_temporal_adverb

    def set_is_past_looking_temporal_adverb(self, is_past_looking_temporal_adverb):
        self._is_past_looking_temporal_adverb = is_past_looking_temporal_adverb

    def set_temporal_marker(self, temporal_marker):
        self._temporal_marker = temporal_marker

    def set_example(self, example):
        self._example = example

    def set_is_possible(self, is_possible):
        self._is_possible = is_possible

    def set_sub_is_repeated_process(self, sub_is_repeated_process):
        self._sub_is_repeated_process = sub_is_repeated_process

    def set_matrix_is_sub(self, matrix_is_sub):
        self._matrix_is_sub = matrix_is_sub.split('/')

    def set_sub_is_dct(self, sub_is_dct):
        self._sub_is_dct = sub_is_dct.split('/')

    def set_matrix_is_dct(self, matrix_is_dct):
        self._matrix_is_dct = matrix_is_dct.split('/')

    def get_matrix_tense(self):
        return self._matrix_tense
    
    def key(self):
        return '{}_{}_{}_{}_{}'.format(self._matrix_tense, self._matrix_aspect, self._sub_tense, self._sub_aspect, self._temporal_marker)

    @staticmethod
    def from_csv_row(row):
        r = Row()
        r.set_matrix_tense(row[0])
        r.set_matrix_aspect(row[1])
        r.set_sub_tense(row[2])
        r.set_sub_aspect(row[3])
        r.set_is_quote(row[4])
        r.set_is_future_looking_temporal_adverb(row[5])
        r.set_is_past_looking_temporal_adverb(row[6])
        r.set_temporal_marker(row[7])
        r.set_example(row[8])
        r.set_is_possible(row[9])
        r.set_sub_is_repeated_process(row[10])
        r.set_matrix_is_sub(row[11])
        r.set_sub_is_dct(row[12])
        r.set_matrix_is_dct(row[13])
        return r
    

class SubordinateAnalysis:

    def load_csv(self):
        path = '/home/lalady6977/oerc/projects/data/subordinate/subordinate_event_said.csv'
        data = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for r in reader:
                if r[0] != '':
                    data.append(Row.from_csv_row(r))
        return data

    def analyse(self):
        data = self.load_csv()
        data_dict = defaultdict(lambda: defaultdict(Row))

        for d in data:
            data_dict[d.key()][d._is_quote] = d
        
        for k in data_dict:
            item = data_dict[k]
            same_order = item['yes']._sub_is_dct == item['no']._sub_is_dct
            same_temporal_marker = item['yes']._temporal_marker == item['no']._temporal_marker
            if same_order and same_temporal_marker and item['yes']._temporal_marker == 'no_marker':
                print()
                print(k)
                print(same_order, '|', item['yes']._sub_is_dct, '|', item['no']._sub_is_dct)
                print(item['yes']._example)
                print(item['no']._example)

        for k in data_dict:
            item = data_dict[k]
            sub_is_dct = item['yes']._sub_is_dct
            if False and len(sub_is_dct) >= 2 and item['yes']._temporal_marker == 'no_marker':
                print()
                print(k)
                print(same_order, '|', item['yes']._sub_is_dct, '|', item['no']._sub_is_dct)
                print(item['yes']._example)
                print(item['no']._example)


if __name__ == '__main__':
    sa = SubordinateAnalysis()
    sa.analyse()


# If the text is cancellable by the temporal maker then the no marker should account for both.