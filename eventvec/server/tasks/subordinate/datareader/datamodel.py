
class SubordinateRow():
    def __init__(self):
        self._matrix_tense = None
        self._matrix_aspect = None
        self._sub_tense = None
        self._sub_aspect = None
        self._is_quote = None
        self._temporal_marker = None
        self._example = None
        self._is_possible = None
        self._sub_is_repeated_process = None
        self._matrix_is_sub = []
        self._dct_is_sub = []
        self._dct_is_matrix = []

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

    def set_dct_is_sub(self, dct_is_sub):
        self._dct_is_sub = dct_is_sub.split('/')

    def set_dct_is_matrix(self, dct_is_matrix):
        self._dct_is_matrix = dct_is_matrix.split('/')

    def matrix_tense(self):
        return self._matrix_tense
    
    def matrix_aspect(self):
        return self._matrix_aspect
    
    def sub_tense(self):
        return self._sub_tense
    
    def sub_aspect(self):
        return self._sub_aspect
    
    def is_quote(self):
        return self._is_quote
    
    def opposite_is_quote(self):
        if self._is_quote == 'yes':
            return 'no'
        return 'yes'
    
    def temporal_marker(self):
        return self._temporal_marker
    
    def example(self):
        return self._example
    
    def is_possible(self):
        return self._is_possible
    
    def sub_is_repeated_process(self):
        return self._sub_is_repeated_process
    
    def matrix_is_sub(self):
        return self._matrix_is_sub
    
    def dct_is_sub(self):
        return self._dct_is_sub
    
    def dct_is_matrix(self):
        return self._dct_is_matrix
    
    def key(self):
        return '{}_{}_{}_{}_{}_{}'.format(self._matrix_tense, self._matrix_aspect, self._sub_tense, self._sub_aspect, self.is_quote(), self._temporal_marker)

    def opposite_key(self):
        return '{}_{}_{}_{}_{}_{}'.format(self._matrix_tense, self._matrix_aspect, self._sub_tense, self._sub_aspect, self.opposite_is_quote(), self._temporal_marker)

    @staticmethod
    def from_csv_row(row):
        r = SubordinateRow()
        r.set_matrix_tense(row[0])
        r.set_matrix_aspect(row[1])
        r.set_sub_tense(row[2])
        r.set_sub_aspect(row[3])
        r.set_is_quote(row[4])
        r.set_temporal_marker(row[5])
        r.set_example(row[6])
        r.set_is_possible(row[7])
        r.set_sub_is_repeated_process(row[8])
        r.set_matrix_is_sub(row[9])
        r.set_dct_is_sub(row[10])
        r.set_dct_is_matrix(row[11])
        return r
    