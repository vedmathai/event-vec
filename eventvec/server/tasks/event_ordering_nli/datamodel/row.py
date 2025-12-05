class Row:
    def __init__(self):
        self._count = 0
        self._possible = None
        self._force_impossible = None
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
        if datum[2] == 'true':
            row._force_impossible = True
        else:
            row._force_impossible = False
        row._relationship = datum[3]
        row._event_number = int(datum[4])
        row._relationship_number = int(datum[5])
        row._hops = int(datum[6])
        row._premise = datum[7]
        row._hypothesis = datum[8]
        row._label = datum[9]
        return row
    
    def to_row(self):
        return [self._count, self._possible, self._force_impossible, self._relationship, self._event_number,
                self._relationship_number, self._hops, self._premise, self._hypothesis,
                self._label]
    